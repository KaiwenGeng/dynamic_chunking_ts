#!/bin/bash

model_name=LatentTransformerWithReconstruction

echo "LatentTransformer with VAE - Hyperparameter Search on ETTh1"
echo "=========================================="
echo "GPU 0: kl_loss_weight = 0.001"
echo "GPU 1: kl_loss_weight = 0.01"
echo "GPU 2: kl_loss_weight = 0.05"
echo "GPU 3: kl_loss_weight = 0.1"
echo "=========================================="

# 创建日志目录
mkdir -p logs

# 共同参数
common_params="--task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
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
  --latent_config medium \
  --reconstruction_mode VAE \
  --reconstruction_loss_weight 0.5 \
  --result_file result_VAE_hypersearch.txt"

# ========== 并行执行4个超参数配置 ==========

# GPU 0: kl_loss_weight = 0.001 (very small)
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  $common_params \
  --model_id ETTh1_VAE_kl_0.001 \
  --kl_loss_weight 0.001 \
  > logs/VAE_ETTh1_kl_0.001.log 2>&1 &

# GPU 1: kl_loss_weight = 0.01 (standard)
CUDA_VISIBLE_DEVICES=1 python -u run.py \
  $common_params \
  --model_id ETTh1_VAE_kl_0.01 \
  --kl_loss_weight 0.01 \
  > logs/VAE_ETTh1_kl_0.01.log 2>&1 &

# GPU 2: kl_loss_weight = 0.05
CUDA_VISIBLE_DEVICES=2 python -u run.py \
  $common_params \
  --model_id ETTh1_VAE_kl_0.05 \
  --kl_loss_weight 0.05 \
  > logs/VAE_ETTh1_kl_0.05.log 2>&1 &

# GPU 3: kl_loss_weight = 0.1 (large)
CUDA_VISIBLE_DEVICES=3 python -u run.py \
  $common_params \
  --model_id ETTh1_VAE_kl_0.1 \
  --kl_loss_weight 0.1 \
  > logs/VAE_ETTh1_kl_0.1.log 2>&1 &

# ========== 等待所有后台任务完成 ==========
echo "All VAE hyperparameter search experiments started. Waiting for completion..."
wait

echo "=========================================="
echo "VAE Hyperparameter Search Completed!"
echo "Check the following log files for details:"
echo "- logs/VAE_ETTh1_kl_0.001.log  (kl_weight=0.001, very small)"
echo "- logs/VAE_ETTh1_kl_0.01.log   (kl_weight=0.01, standard)"
echo "- logs/VAE_ETTh1_kl_0.05.log   (kl_weight=0.05)"
echo "- logs/VAE_ETTh1_kl_0.1.log    (kl_weight=0.1, large)"
echo ""
echo "Compare the test results to find the best kl_loss_weight!"
echo "Note: Watch for posterior collapse if KL loss is too small."
echo "=========================================="


