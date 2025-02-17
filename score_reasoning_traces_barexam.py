import os
import json
import argparse
import asyncio
from tqdm import tqdm
from asyncio import Semaphore
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint="https://api-ai-sandbox.princeton.edu/",
    api_version="2024-02-01"
)

async def extract_solution(generated_text):
    begin_tag = "[begin_of_solution]"
    end_tag = "[end_of_solution]"
    start_index = generated_text.find(begin_tag)
    if start_index == -1:
        return generated_text.strip()
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
            "You are a scoring assistant for multiple choice questions. "
            "Determine if the final answer provided in the reasoning trace matches the correct answer.\n\n"
            "Respond with only 'Correct' if they match, or 'Incorrect' if they don't match.\n\n"
            f"Correct answer: {correct_answer}\n"
            f"Reasoning trace answer: {solution}\n\n"
        )
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        return "Correct" in response.choices[0].message.content

async def process_file(file_path, semaphore):
    file_name = os.path.basename(file_path)
    with open(file_path, "r") as f:
        try:
            data = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"Skipping {file_path} due to JSONDecodeError: {e}")
            return
    if "scored_correct" in data:
        print(f"Skipping {file_path} because it has already been scored")
        return
    # if "sample_0" not in file_name:
    #     print(f"Skipping {file_path} because it is not a sample 0 file")
    #     return
    generated_text = data.get("generated_text", None)
    if generated_text is None:
        print(f"Skipping {file_path} because it has no reformatted text")
        return
    solution = await extract_solution(generated_text)
    correct_answer = data.get("metadata", {}).get("answer", "").strip()
    is_correct = await score_answer(solution, correct_answer, semaphore)
    data["scored_correct"] = is_correct
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

async def main(args):
    input_dir = args.input_dir
    semaphore = Semaphore(30)
    json_files = [file for file in os.listdir(input_dir) if file.endswith(".json")]
    tasks = [asyncio.create_task(process_file(os.path.join(input_dir, file), semaphore)) for file in json_files]
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing files"):
        await task

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score reasoning traces for bare exam questions")
    parser.add_argument("--input_dir", required=True, help="Directory containing generated sample JSON files")
    args = parser.parse_args()
    asyncio.run(main(args)) 