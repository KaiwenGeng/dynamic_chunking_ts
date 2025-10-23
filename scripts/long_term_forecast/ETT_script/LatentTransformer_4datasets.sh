#!/bin/bash

model_name=LatentTransformer

echo "LatentTransformer - 4 ETT Datasets Evaluation (Dual Encoder Architecture)"
echo "=========================================="
echo "GPU 0: ETTh1 (Dual 8x compression)"
echo "GPU 1: ETTh2 (Dual 8x compression)"
echo "GPU 2: ETTm1 (Dual 8x compression)"
echo "GPU 3: ETTm2 (Dual 8x compression)"
echo "Architecture: Encoder uses regular conv, Decoder uses causal conv"
echo "=========================================="

# 创建日志目录
mkdir -p logs

# 共同参数
common_params="--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --model $model_name \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
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
  --result_file result_4datasets.txt"

# ========== 并行执行4个数据集 ==========

# GPU 0: ETTh1
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  $common_params \
  --data ETTh1 \
  --data_path ETTh1.csv \
  --model_id ETTh1_latent_dual8x \
  --factor 3 \
  > logs/LatentTransformer_ETTh1.log 2>&1 &

# GPU 1: ETTh2
CUDA_VISIBLE_DEVICES=1 python -u run.py \
  $common_params \
  --data ETTh2 \
  --data_path ETTh2.csv \
  --model_id ETTh2_latent_dual8x \
  --factor 3 \
  > logs/LatentTransformer_ETTh2.log 2>&1 &

# GPU 2: ETTm1
CUDA_VISIBLE_DEVICES=2 python -u run.py \
  $common_params \
  --data ETTm1 \
  --data_path ETTm1.csv \
  --model_id ETTm1_latent_dual8x \
  > logs/LatentTransformer_ETTm1.log 2>&1 &

# GPU 3: ETTm2
CUDA_VISIBLE_DEVICES=3 python -u run.py \
  $common_params \
  --data ETTm2 \
  --data_path ETTm2.csv \
  --model_id ETTm2_latent_dual8x \
  --factor 1 \
  > logs/LatentTransformer_ETTm2.log 2>&1 &

# ========== 等待所有后台任务完成 ==========
echo "All 4 dataset experiments started in parallel. Waiting for completion..."
wait

echo "=========================================="
echo "All experiments completed!"
echo "Check the following log files for details:"
echo "- logs/LatentTransformer_ETTh1.log"
echo "- logs/LatentTransformer_ETTh2.log"
echo "- logs/LatentTransformer_ETTm1.log"
echo "- logs/LatentTransformer_ETTm2.log"
echo "=========================================="

