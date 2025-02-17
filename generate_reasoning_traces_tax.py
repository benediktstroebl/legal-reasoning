import os
import json
import asyncio
import argparse
import random

from datasets import load_from_disk
from tqdm import tqdm
from prompts import TAX_PROBLEMS_SYSTEM_PROMPT, TAX_PROBLEMS_USER_PROMPT, TAX_PROBLEMS_MULTIPLE_CHOICE_USER_PROMPT
from openai import AsyncOpenAI


instruct_model = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
instruct_client = AsyncOpenAI(api_key="token-abc123", base_url="http://localhost:8000/v1")
# Parameters
MAX_TOKENS = 8192
DEFAULT_NUM_SAMPLES = 8  # maximum samples per question
CONCURRENCY_LIMIT = 100
TEMPERATURE = 0.7
TOP_P = 0.7


def get_prompt(sample, mc=False):
    if mc:
        answers = "\n".join([f"{i+1}. {answer}" for i, answer in enumerate(sample["answers"])])
        return [
            {"role": "system", "content": TAX_PROBLEMS_SYSTEM_PROMPT},
            {"role": "user", "content": TAX_PROBLEMS_MULTIPLE_CHOICE_USER_PROMPT.format(question=sample["question"], answers=answers)},
        ]
    else:
        return [
            {"role": "system", "content": TAX_PROBLEMS_SYSTEM_PROMPT},
            {"role": "user", "content": TAX_PROBLEMS_USER_PROMPT.format(question=sample["question"])},
        ]


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


async def main(num_samples, use_mc, begin_idx=0):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = []
    ds = load_from_disk("data/tax_problems_dataset")["train"]
    for idx, sample in enumerate(tqdm(ds, desc="Processing samples")):
        prompt = get_prompt(sample, mc=use_mc)
        samples_for_question = random.randint(1, num_samples)
        for sample_num in range(samples_for_question):
            output_filename = os.path.join(OUTPUT_DIR, f"question_{idx}_sample_{sample_num}.json")
            if os.path.exists(output_filename):
                print(f"Skipping existing file: {output_filename}")
                continue
            tasks.append(asyncio.create_task(limited_process(semaphore, idx, sample, sample_num, prompt, output_filename)))
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Waiting for tasks"):
        await task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reasoning traces on the train set")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Max number of samples per question (actual number is variable between 1 and this value)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", help="Model name to use for inference")
    parser.add_argument("--mc", action="store_true", help="Use multiple-choice mode")
    parser.add_argument("--begin_idx", type=int, default=0, help="Index of the first question to process")
    args = parser.parse_args()
    OUTPUT_DIR = f"reasoning_traces/mc_{args.mc}/{args.model.replace('/', '_')}/"
    asyncio.run(main(args.num_samples, args.mc, args.begin_idx)) 
