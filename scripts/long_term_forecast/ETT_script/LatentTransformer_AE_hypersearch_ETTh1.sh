#!/bin/bash

model_name=LatentTransformerWithReconstruction

echo "LatentTransformer with AE - Hyperparameter Search on ETTh1"
echo "=========================================="
echo "GPU 0: reconstruction_loss_weight = 0.1"
echo "GPU 1: reconstruction_loss_weight = 0.3"
echo "GPU 2: reconstruction_loss_weight = 0.5"
echo "GPU 3: reconstruction_loss_weight = 1.0"
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
  --reconstruction_mode AE \
  --result_file result_AE_hypersearch.txt"

# ========== 并行执行4个超参数配置 ==========

# GPU 0: reconstruction_loss_weight = 0.1
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  $common_params \
  --model_id ETTh1_AE_recon_0.1 \
  --reconstruction_loss_weight 0.1 \
  > logs/AE_ETTh1_recon_0.1.log 2>&1 &

# GPU 1: reconstruction_loss_weight = 0.3
CUDA_VISIBLE_DEVICES=1 python -u run.py \
  $common_params \
  --model_id ETTh1_AE_recon_0.3 \
  --reconstruction_loss_weight 0.3 \
  > logs/AE_ETTh1_recon_0.3.log 2>&1 &

# GPU 2: reconstruction_loss_weight = 0.5
CUDA_VISIBLE_DEVICES=2 python -u run.py \
  $common_params \
  --model_id ETTh1_AE_recon_0.5 \
  --reconstruction_loss_weight 0.5 \
  > logs/AE_ETTh1_recon_0.5.log 2>&1 &

# GPU 3: reconstruction_loss_weight = 1.0
CUDA_VISIBLE_DEVICES=3 python -u run.py \
  $common_params \
  --model_id ETTh1_AE_recon_1.0 \
  --reconstruction_loss_weight 1.0 \
  > logs/AE_ETTh1_recon_1.0.log 2>&1 &

# ========== 等待所有后台任务完成 ==========
echo "All AE hyperparameter search experiments started. Waiting for completion..."
wait

echo "=========================================="
echo "AE Hyperparameter Search Completed!"
echo "Check the following log files for details:"
echo "- logs/AE_ETTh1_recon_0.1.log  (weight=0.1)"
echo "- logs/AE_ETTh1_recon_0.3.log  (weight=0.3)"
echo "- logs/AE_ETTh1_recon_0.5.log  (weight=0.5)"
echo "- logs/AE_ETTh1_recon_1.0.log  (weight=1.0)"
echo ""
echo "Compare the test results to find the best reconstruction_loss_weight!"
echo "=========================================="

