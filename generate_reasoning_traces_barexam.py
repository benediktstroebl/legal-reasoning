import os
import json
import asyncio
import argparse
import random

from datasets import load_dataset, Dataset
from tqdm import tqdm
from openai import AsyncOpenAI
from prompts import BAREXAM_SYSTEM_PROMPT, BAREXAM_MULTIPLE_CHOICE_SYSTEM_PROMPT

instruct_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
instruct_client = AsyncOpenAI(api_key="token-abc123", base_url="http://localhost:8000/v1")
# Parameters
MAX_TOKENS = 8192
DEFAULT_NUM_SAMPLES = 8  # maximum samples per question
CONCURRENCY_LIMIT = 100
TEMPERATURE = 0.7
TOP_P = 0.7

def format_user_prompt(q, a):
    prompt = "<question> " + q + " </question>\n" + "<answer_choices>\n" + "\n".join([str(i) for i in a]) + "\n</answer_choices>"
    return prompt

def load_barexam_dataset():
    questions = load_dataset("data/barexamqa-mbe", "qa")
    train = questions["train"]
    prompt_list = []
    for row in train:
        answers_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate([row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]])])
        # print(row)
        user_text = BAREXAM_MULTIPLE_CHOICE_SYSTEM_PROMPT.format(question=f"Question: {row['question']}", prompt=f"Prompt: {row['prompt']}\n" if row["prompt"] != "nan" else "", answers=answers_text)
        prompt = [
            {'role': 'system', 'content': BAREXAM_SYSTEM_PROMPT},
            {'role': 'user', 'content': user_text}
        ]
        # print prompt in a nice format
        # print(prompt[0]["content"])
        # print(prompt[1]["content"])
        prompt_list.append(prompt)
    answers = [str(ans) for ans in train["answer"]]
    dataset = Dataset.from_dict({"prompt": prompt_list, "answer": answers})
    return dataset

async def perform_reasoning_trace_inference(conversation):
    chat_completion = await instruct_client.chat.completions.create(
        model=instruct_model,
        messages=conversation,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    generated = chat_completion.choices[0].message.content.replace("</think>", "[end_of_thought]").replace("<think>", "[begin_of_thought]")
    return generated

async def process_reasoning_trace_sample(idx, sample, sample_num, prompt, output_filename):
    generated = await perform_reasoning_trace_inference(prompt)
    output_data = {
        "prompt": prompt,
        "generated_text": generated,
        "metadata": sample,
        "question_id": idx,
        "sample_index": sample_num,
        "generation_config": {
            "instruct_model": instruct_model,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "max_tokens": MAX_TOKENS,
        },
    }
    with open(output_filename, "w") as f:
        json.dump(output_data, f, indent=2)

async def limited_process(semaphore, idx, sample, sample_num, prompt, output_filename):
    async with semaphore:
        await process_reasoning_trace_sample(idx, sample, sample_num, prompt, output_filename)

async def main(num_samples, begin_idx=0):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    dataset = load_barexam_dataset()
    for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
        if idx < begin_idx:
            continue
        prompt = sample["prompt"]
        
        for sample_num in range(num_samples):
            output_filename = os.path.join(OUTPUT_DIR, f"question_{idx}_sample_{sample_num}.json")
            if os.path.exists(output_filename):
                print(f"Skipping existing file: {output_filename}")
                continue
            tasks.append(asyncio.create_task(limited_process(semaphore, idx, sample, sample_num, prompt, output_filename)))
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Waiting for tasks"):
        await task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning traces on the barexam dataset")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Max number of samples per question (actual number is variable between 1 and this value)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name to use for inference")
    parser.add_argument("--begin_idx", type=int, default=0, help="Index of the first question to process")
    args = parser.parse_args()
    instruct_model = args.model
    OUTPUT_DIR = f"reasoning_traces/barexam/{args.model.replace('/', '_')}/"
    asyncio.run(main(args.num_samples, args.begin_idx)) 