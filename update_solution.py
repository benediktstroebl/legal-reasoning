import os
import glob
import re
import json

def convert_solution(text):
    cleaned = text.strip().replace('\n', '').replace('\\n', '')
    if cleaned.isdigit():
        num = int(cleaned)
        if 1 <= num <= 26:
            return chr(64 + num)
    return cleaned

def update_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if "reformatted_text" in data:
        text = data["reformatted_text"]
        pattern = re.compile(r'(<\|begin_of_solution\|>)(.*?)(<\|end_of_solution\|>)', re.DOTALL)
        def replacer(match):
            return f"{match.group(1)}{convert_solution(match.group(2))}{match.group(3)}"
        data["reformatted_text"] = pattern.sub(replacer, text)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    directory = "reasoning_traces/barexam/deepseek-ai_DeepSeek-R1-Distill-Qwen-32B"
    for filepath in glob.glob(os.path.join(directory, "*.json")):
        update_file(filepath)

if __name__ == '__main__':
    main() 