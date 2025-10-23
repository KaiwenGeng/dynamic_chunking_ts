#!/bin/bash

model_name=LatentTransformerWithReconstruction

echo "LatentTransformer with MAE - Hyperparameter Search on ETTh1"
echo "=========================================="
echo "GPU 0: mask_ratio = 0.15"
echo "GPU 1: mask_ratio = 0.25"
echo "GPU 2: mask_ratio = 0.50"
echo "GPU 3: mask_ratio = 0.75"
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
  --reconstruction_mode MAE \
  --reconstruction_loss_weight 0.5 \
  --result_file result_MAE_hypersearch.txt"

# ========== 并行执行4个超参数配置 ==========

# GPU 0: mask_ratio = 0.15
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  $common_params \
  --model_id ETTh1_MAE_mask_0.15 \
  --mask_ratio 0.15 \
  > logs/MAE_ETTh1_mask_0.15.log 2>&1 &

# GPU 1: mask_ratio = 0.25 (BERT-style)
CUDA_VISIBLE_DEVICES=1 python -u run.py \
  $common_params \
  --model_id ETTh1_MAE_mask_0.25 \
  --mask_ratio 0.25 \
  > logs/MAE_ETTh1_mask_0.25.log 2>&1 &

# GPU 2: mask_ratio = 0.50
CUDA_VISIBLE_DEVICES=2 python -u run.py \
  $common_params \
  --model_id ETTh1_MAE_mask_0.50 \
  --mask_ratio 0.50 \
  > logs/MAE_ETTh1_mask_0.50.log 2>&1 &

# GPU 3: mask_ratio = 0.75 (aggressive masking)
CUDA_VISIBLE_DEVICES=3 python -u run.py \
  $common_params \
  --model_id ETTh1_MAE_mask_0.75 \
  --mask_ratio 0.75 \
  > logs/MAE_ETTh1_mask_0.75.log 2>&1 &

# ========== 等待所有后台任务完成 ==========
echo "All MAE hyperparameter search experiments started. Waiting for completion..."
wait

echo "=========================================="
echo "MAE Hyperparameter Search Completed!"
echo "Check the following log files for details:"
echo "- logs/MAE_ETTh1_mask_0.15.log  (mask=15%)"
echo "- logs/MAE_ETTh1_mask_0.25.log  (mask=25%, BERT-style)"
echo "- logs/MAE_ETTh1_mask_0.50.log  (mask=50%)"
echo "- logs/MAE_ETTh1_mask_0.75.log  (mask=75%, aggressive)"
echo ""
echo "Compare the test results to find the best mask_ratio!"
echo "=========================================="


