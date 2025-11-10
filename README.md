# Generalizable and Efficient Automated Scoring with a Knowledge-Distilled Multi-Task Mixture-of-Experts

**This repository contains code for the paper "Generalizable and Efficient Automated Scoring with a Knowledge-Distilled Multi-Task Mixture-of-Experts".**


## Usage Example
### Example Teacher Fine-tuning Usage:
python finetune_gptoss20b_lora_jsonl_cls.py \
    --model_name "openai/gpt-oss-20b" \
    --train_file ./data/task1_train.csv \
    --test_file ./data/task1_test.csv \
    --max_length 512 \
    --batch_size 4 \
    --accum_steps 8 \
    --epochs 10 \
    --lr 2.0e-5 \
    --warmup_ratio 0.10 \
    --weight_decay 0.02 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --gradient_checkpointing \
    --bf16 \
    --log_every_steps 200 \
    --max_grad_norm 0.3 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --threshold 0.5 \
    --early_stop_patience 4 \
    --embed_dim 768 \
    --output_dir ./results/task1 \
    --probs_out ./results/task1/train_probs.csv \
    --include_text_in_csv \
    > ./results/task1/train.log \
    2> ./results/task1/train.err

### MTL_MOE Example:
python mtl_moe_trainer.py --num_experts 4 --kd_weight 0.5 --load_balance_weight 0.01 --seed 42 --max_epochs 20 --patience 3 --data_dir '../PASTA_data/new_processed_data/' --output_dir './PASTA-Models/'
