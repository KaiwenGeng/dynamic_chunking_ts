#!/bin/bash

model_name=LatentTransformer

echo "LatentTransformer ETTh1 Experiments (Dual Encoder Architecture - Compression Study)"
echo "=========================================="
echo "GPU 0: No compression (1x) - Dual encoder baseline"
echo "GPU 1: 4x compression - Dual encoder"
echo "GPU 2: 8x compression - Dual encoder"
echo "GPU 3: 16x compression - Dual encoder"
echo "Architecture: Encoder uses regular conv, Decoder uses causal conv"
echo "=========================================="

# 创建日志目录
mkdir -p logs

# ========== 并行执行4个实验 ==========

# GPU 0: No compression (1x) - Dual encoder baseline
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_latent_dual1x \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 2048 \
  --des 'Exp' \
  --itr 1 \
  --patience 100 \
  --train_epochs 50 \
  --latent_config dual_no_compression \
  --result_file result_compression_study.txt > logs/ETTh1_latent_dual1x.log 2>&1 &

# GPU 1: 4x compression - Dual encoder
CUDA_VISIBLE_DEVICES=1 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_latent_dual4x \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 2048 \
  --des 'Exp' \
  --itr 1 \
  --patience 100 \
  --train_epochs 50 \
  --latent_config dual_4x \
  --result_file result_compression_study.txt > logs/ETTh1_latent_dual4x.log 2>&1 &

# GPU 2: 8x compression - Dual encoder
CUDA_VISIBLE_DEVICES=2 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_latent_dual8x \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 2048 \
  --des 'Exp' \
  --itr 1 \
  --patience 100 \
  --train_epochs 50 \
  --latent_config dual_8x \
  --result_file result_compression_study.txt > logs/ETTh1_latent_dual8x.log 2>&1 &

# GPU 3: 16x compression - Dual encoder
CUDA_VISIBLE_DEVICES=3 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96_latent_dual16x \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --d_ff 2048 \
  --des 'Exp' \
  --itr 1 \
  --patience 100 \
  --train_epochs 50 \
  --latent_config dual_16x \
  --result_file result_compression_study.txt > logs/ETTh1_latent_dual16x.log 2>&1 &

# ========== 等待所有后台任务完成 ==========
echo "All LatentTransformer ETTh1 experiments started in parallel. Waiting for completion..."
wait

echo "=========================================="
echo "LatentTransformer ETTh1 compression study completed!"
echo "Check the following log files for details:"
echo "- logs/ETTh1_latent_dual1x.log (No compression baseline)"
echo "- logs/ETTh1_latent_dual4x.log (4x compression)"
echo "- logs/ETTh1_latent_dual8x.log (8x compression)"
echo "- logs/ETTh1_latent_dual16x.log (16x compression)"
echo "=========================================="

