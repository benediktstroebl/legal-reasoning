import os
import json
import argparse
import asyncio
import random
from tqdm import tqdm
from datasets import load_dataset, Dataset
from openai import AsyncOpenAI, AzureOpenAI
from litellm import acompletion
from dotenv import load_dotenv
from asyncio import Semaphore
from prompts import BAREXAM_SYSTEM_PROMPT, BAREXAM_MULTIPLE_CHOICE_SYSTEM_PROMPT

load_dotenv(override=True)

instruct_model = "deepseek-ai/DeepSeek-R1"
instruct_client = AsyncOpenAI(
    # api_key="token-abc123", 
    # base_url="http://localhost:8000/v1"
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
    )
scoring_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint="https://api-ai-sandbox.princeton.edu/",
    api_version="2024-02-01"
)
MAX_TOKENS = 7000
DEFAULT_NUM_SAMPLES = 1
CONCURRENCY_LIMIT = 5
TEMPERATURE = 0.7
TOP_P = 0.7

async def perform_reasoning_trace_inference(prompt):
    chat_completion = await instruct_client.chat.completions.create(
        model=instruct_model,
        messages=prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    generated = chat_completion.choices[0].message.content
    return generated

async def extract_solution(generated_text):
    begin_tag = "[begin_of_solution]"
    end_tag = "[end_of_solution]"
    start_index = generated_text.find(begin_tag)
    if start_index == -1:
        return generated_text
    start_index += len(begin_tag)
    end_index = generated_text.find(end_tag, start_index)
    if end_index == -1:
        solution = generated_text[start_index:].strip()
    else:
        solution = generated_text[start_index:end_index].strip()
    return solution

async def score_answer(solution, correct_answer, semaphore):
    async with semaphore:
        prompt = (
            "You are a scoring assistant. Determine if the following solution answer matches the correct answer logically.\n\n"
            "Respond with only 'Correct' if the solution matches the correct answer logically, or 'Incorrect' if it does not.\n\n"
            f"Correct answer: {correct_answer}\n"
            f"Solution answer: {solution}\n\n"
        )
        response = await asyncio.to_thread(
            scoring_client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0
        )
        return "Correct" in response.choices[0].message.content

def load_bar_exam_dataset():
    dataset = load_dataset("data/barexamqa-mbe", "qa")
    raw_data = dataset["test"] if "test" in dataset else dataset["train"]
    prompt_list = []
    for row in raw_data:
        answers_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate([row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]])])
        user_text = BAREXAM_MULTIPLE_CHOICE_SYSTEM_PROMPT.format(
            question=f"Question: {row['question']}",
            prompt=f"Prompt: {row['prompt']}\n" if row["prompt"] != "nan" else "",
            answers=answers_text
        )
        prompt = [
            {'role': 'system', 'content': BAREXAM_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_text}
        ]
        prompt_list.append(prompt)
    answers = [str(row["answer"]).strip() for row in raw_data]
    dataset = Dataset.from_dict({"prompt": prompt_list, "answer": answers, "raw": raw_data})
    return dataset

async def process_sample(idx, sample, sample_num, prompt, output_filename, score_semaphore):
    generated = await perform_reasoning_trace_inference(prompt)
    solution = await extract_solution(generated)
    correct_answer = str(sample["answer"]).strip()
    is_correct = await score_answer(solution, correct_answer, score_semaphore)
    output_data = {
        "prompt": prompt,
        "generated_text": generated,
        "metadata": sample,
        "question_id": idx,
        "sample_index": sample_num,
        "scored_correct": is_correct,
        "generation_config": {
            "instruct_model": instruct_model,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        },
    }
    with open(output_filename, "w") as f:
        json.dump(output_data, f, indent=2)

async def limited_process(semaphore, idx, sample, sample_num, prompt, output_filename, score_semaphore):
    async with semaphore:
        await process_sample(idx, sample, sample_num, prompt, output_filename, score_semaphore)

async def main(num_samples, begin_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    semaphore = Semaphore(CONCURRENCY_LIMIT)
    score_semaphore = Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    dataset = load_bar_exam_dataset()
    for idx, sample in enumerate(tqdm(dataset, desc="Processing questions")):
        if idx < begin_idx:
            continue
        prompt = sample["prompt"]
        samples_for_question = random.randint(1, num_samples)
        for sample_num in range(samples_for_question):
            output_filename = os.path.join(output_dir, f"question_{idx}_sample_{sample_num}.json")
            if os.path.exists(output_filename):
                print(f"Skipping existing file: {output_filename}")
                continue
            tasks.append(asyncio.create_task(limited_process(semaphore, idx, sample, sample_num, prompt, output_filename, score_semaphore)))
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Waiting for tasks"):
        await task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on the Bar Exam test set")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Max number of samples per question")
    parser.add_argument("--begin_idx", type=int, default=0, help="Index of the first question to process")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store evaluation outputs")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name for inference")
    args = parser.parse_args()
    instruct_model = args.model
    asyncio.run(main(args.num_samples, args.begin_idx, args.output_dir)) 