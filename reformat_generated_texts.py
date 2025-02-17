import argparse
import json
import os
import time
import asyncio
from tqdm import tqdm

from dotenv import load_dotenv
from prompts import convert_prompt, convert_prompt_example
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint="https://api-ai-sandbox.princeton.edu/",
    api_version="2024-02-01"
)

def process_content(content):
    prompt = convert_prompt.format(example=convert_prompt_example, content=content)
    retries = 3
    while retries > 0:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a solution format converter."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=16384,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error processing content: {e}. Retrying...")
            retries -= 1
            if retries == 0:
                return f"Error processing content: {e}"
            time.sleep(5)

def process_file(file_path):
    if "reformatted_" in os.path.basename(file_path):
        print(f"Skipping {file_path} because it is already reformatted")
        return file_path, file_path

    output_filename = file_path

    print(f"Processing {file_path}")
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            
        if 'reformatted_text' in data:
            print(f"Skipping {file_path} because it has already been reformatted")
            return file_path, file_path
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return file_path, file_path
    content = data.get("generated_text", "")
    reformatted = process_content(content)
    data["reformatted_text"] = reformatted
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    return file_path, output_filename

async def process_file_async(file_path, semaphore):
    async with semaphore:
        return await asyncio.to_thread(process_file, file_path)

async def main():
    parser = argparse.ArgumentParser(
        description="Reformat generated texts for tax and barexam questions using Azure OpenAI"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing sample JSON files.")
    parser.add_argument("--max_concurrency", type=int, default=80, help="Maximum number of concurrent tasks")
    args = parser.parse_args()
    file_paths = [
        os.path.join(args.input_dir, filename)
        for filename in os.listdir(args.input_dir) if filename.endswith(".json")
    ]
    semaphore = asyncio.Semaphore(args.max_concurrency)
    tasks = [asyncio.create_task(process_file_async(fp, semaphore)) for fp in file_paths]
    results = []
    pbar = tqdm(total=len(tasks))
    for future in asyncio.as_completed(tasks):
        result = await future
        results.append(result)
        pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    asyncio.run(main()) 