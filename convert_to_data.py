import argparse
import json
import os
import random
from tqdm import tqdm
import re
from prompts import BAREXAM_SYSTEM_PROMPT, BAREXAM_MULTIPLE_CHOICE_SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Convert JSON data for processing.")
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing input JSON files."
    )
    parser.add_argument("--output", type=str, help="Output JSON file.")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of correct samples per question (0 for no limit)")
    parser.add_argument("--base", action="store_true", help="Use base model")
    args = parser.parse_args()
    
    data_by_question = {}
    for filename in tqdm(os.listdir(args.input_dir), desc="Processing files"):
        if filename.endswith(".json"):
            regex = re.compile(r"question_(?P<question_id>\d+)_sample_(?P<sample_id>\d+)(?P<suffix>.*)\.json")
            match = regex.match(filename)
            question_id = int(match.group("question_id"))
            sample_id = int(match.group("sample_id"))
            suffix = match.group("suffix")
            
            filepath = os.path.join(args.input_dir, filename)
            with open(filepath, "r") as f:
                sample = json.load(f)

            if args.base:
                system_prompt = BAREXAM_SYSTEM_PROMPT
                metadata = sample["metadata"]
                answers_text = "\n".join(
                    [f"{i+1}. {choice}" for i, choice in enumerate([metadata["choice_a"], metadata["choice_b"], metadata["choice_c"], metadata["choice_d"]])]
                )
                user_prompt = BAREXAM_MULTIPLE_CHOICE_SYSTEM_PROMPT.format(
                    question=f"Question: {metadata['question']}",
                    prompt=f"Prompt: {metadata['prompt']}\n" if metadata["prompt"] != "nan" else "",
                    answers=answers_text
                )
            else:
                system_prompt = sample["prompt"][0]["content"]
                user_prompt = sample["prompt"][1]["content"]
            assistant_response = sample["reformatted_text"]

            # Accept this data
            if sample["scored_correct"]:
                # Create the conversation format
                conversations = [
                    {"from": "user", "value": user_prompt},
                    {"from": "assistant", "value": assistant_response},
                ]

                # Prepare the final structure
                cur_data = {
                    "system": system_prompt,
                    "conversations": conversations,
                }
                if question_id not in data_by_question:
                    data_by_question[question_id] = []
                data_by_question[question_id].append(cur_data)

    all_data = []
    # Randomly select up to k correct samples per question
    for samples in data_by_question.values():
        if args.max_samples > 0 and len(samples) > args.max_samples:
            selected = random.sample(samples, args.max_samples)
        else:
            selected = samples
        all_data.extend(selected)

    # Save the converted data to the output file
    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=4)

    # Print number of unique questions based on unique user prompts
    unique_questions = len({data["conversations"][0]["value"] for data in all_data})
    print(f"Number of unique questions: {unique_questions}")

    print(
        f"Conversion completed. The data has been saved to {args.output} with {len(all_data)} data."
    )

if __name__ == "__main__":
    main()