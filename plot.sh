python plot_pass_at_k.py --input_dirs \
    eval/barexam_r1_qwen_14b \
    eval/barexam_qwen_7b_instruct \
    eval/barexam_r1_qwen_7b \
    eval/barexam_r1_qwen_32b \
    eval/barexam_r1_qwen_32b_qwen25_full_cleaned_data \
    eval/grpo_qwen_7b_full_ckpt2300 \
    eval/barexam_r1_qwen_32b_qwen25_14b_full_cleaned_data \
    eval/Qwen-7B-SFT-GRPO-barexam \
    --output barexam_grpo_pass_at_k.pdf
