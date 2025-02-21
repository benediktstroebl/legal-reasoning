import re
import torch
import os
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# Load and prep dataset

SYSTEM_PROMPT = """
Respond in the following format:
<|begin_of_thought|>
...
<|end_of_thought|>
<|begin_of_solution|>
...
<|end_of_solution|>
"""

BAREXAM_SYSTEM_PROMPT = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|>. Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:
"""

XML_COT_FORMAT = """\
<|begin_of_thought|>
{reasoning}
<|end_of_thought|>
<|begin_of_solution|>
{answer}
<|end_of_solution|>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<|begin_of_solution|>")[-1]
    answer = answer.split("<|end_of_solution|>")[0]
    return answer.strip()

# uncomment middle messages for 1-shot prompting
def get_tax_questions(split = "train") -> Dataset:
    data = load_from_disk("data/tax_problems_dataset")[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': x["correct_answer"]
    }) # type: ignore
    return data # type: ignore

def format_user_prompt(q, a):
    prompt = "<question> " + q + " </question>\n" + "<answer_choices>\n" + "\n".join([str(i) for i in a]) + "\n</answer_choices>"
    return prompt

def load_barexam_dataset():
    questions = load_dataset("data/barexamqa-mbe", "qa")
    train = questions["train"]
    prompts = [[
        {'role': 'system', 'content': BAREXAM_SYSTEM_PROMPT},
        {'role': 'user', 'content': format_user_prompt(row["prompt"] + " " + row["question"], [row["choice_a"], row["choice_b"], row["choice_c"], row["choice_d"]])}
    ] for row in train]
    answers = [str(i) for i in train["answer"]]
    dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})
    return dataset

dataset = load_barexam_dataset()


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r.strip().lower() == a.strip().lower() else 0.0 for r, a in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<|begin_of_thought|>\n.*?\n<|end_of_thought|>\n<|begin_of_solution|>\n.*?\n<|end_of_solution|>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<|begin_of_thought|>.*?<|end_of_thought|>\s*<|begin_of_solution|>.*?<|end_of_solution|>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<|begin_of_thought|>\n") == 1:
        count += 0.125
    if text.count("\n<|end_of_thought|>\n") == 1:
        count += 0.125
    if text.count("\n<|begin_of_solution|>\n") == 1:
        count += 0.125
        count -= len(text.split("\n<|end_of_solution|>\n")[-1])*0.001
    if text.count("\n<|end_of_solution|>\n") == 1:
        count += 0.125
        count -= (len(text.split("\n<|end_of_solution|>\n")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def main(args):
    model_name = args.model_name
    output_dir = args.output_dir
    run_name = args.run_name
    
    # Load dataset
    dataset = load_barexam_dataset()

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        beta=0.02,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=6,
        num_generations=6,
        max_prompt_length=1024,
        max_completion_length=3072,
        num_train_epochs=args.epochs,
        save_steps=50,        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=True,
        vllm_device="cuda:0",
        vllm_gpu_memory_utilization=.5,
        report_to="wandb" #I'm disabling Wandb.
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model and tokenizer
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
    from accelerate import infer_auto_device_map, init_empty_weights

    
    # with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="balanced_low_0"
    )
    # device_map = infer_auto_device_map(model, max_memory={
    #     0: "0GiB", 
    #     1: "76GiB", 
    #     2: "76GiB", 
    #     3: "76GiB", 
    #     4: "76GiB", 
    #     5: "76GiB", 
    #     6: "76GiB", 
    #     7: "76GiB"})
    
    # from huggingface_hub import snapshot_download
    # # checkpoint = "marcsun13/gpt2-xl-linear-sharded"
    # weights_location = snapshot_download(repo_id=model_name)
    
    # from accelerate import load_checkpoint_and_dispatch

    # model = load_checkpoint_and_dispatch(
    #     model, device_map=device_map, checkpoint=model_name
    # )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # unset cuda visible devices
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    
    # use peft at your own risk; not working for me with multi-GPU training
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        #peft_config=peft_config
    )
    trainer.train()
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/Qwen-7B-GRPO-barexam")
    parser.add_argument("--run_name", type=str, default="Qwen-7B-GRPO-barexam")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)
