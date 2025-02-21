CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python grpo_barexam_trl.py \
    --model_name /home/legal-reasoning/LLaMA-Factory/saves/barexam_r1_qwen_32b_qwen25_full_cleaned_data/checkpoint-58 \
    --output_dir outputs/Qwen-7B-SFT-GRPO-barexam \
    --epochs 2 \
    --run_name Qwen-7B-SFT-GRPO-barexam