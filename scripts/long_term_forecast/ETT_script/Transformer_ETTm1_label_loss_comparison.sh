#!/bin/bash

model_name=Transformer

echo "ETTm1 Label Loss Comparison Experiments"
echo "=========================================="
echo "GPU 0: No label loss (baseline)"
echo "GPU 1: Label loss weight 0.1"
echo "GPU 2: Label loss weight 0.5" 
echo "GPU 3: Label loss weight 1.0"
echo "=========================================="

# 创建日志目录
mkdir -p logs

# ========== 并行执行4个实验 ==========

# GPU 0: 无label loss (基线)
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_no_label_loss \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patience 100 \
  --train_epochs 100 > logs/ETTm1_no_label_loss.log 2>&1 &

# GPU 1: 权重 0.1
CUDA_VISIBLE_DEVICES=1 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_label_loss_0.1 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patience 100 \
  --train_epochs 100 \
  --use_label_loss \
  --label_loss_weight 0.1 > logs/ETTm1_label_loss_0.1.log 2>&1 &

# GPU 2: 权重 0.5
CUDA_VISIBLE_DEVICES=2 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_label_loss_0.5 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patience 100 \
  --train_epochs 100 \
  --use_label_loss \
  --label_loss_weight 0.5 > logs/ETTm1_label_loss_0.5.log 2>&1 &

# GPU 3: 权重 1.0
CUDA_VISIBLE_DEVICES=3 python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96_label_loss_1.0 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --patience 100 \
  --train_epochs 100 \
  --use_label_loss \
  --label_loss_weight 1.0 > logs/ETTm1_label_loss_1.0.log 2>&1 &

# ========== 等待所有后台任务完成 ==========
echo "All ETTm1 experiments started in parallel. Waiting for completion..."
wait

echo "=========================================="
echo "ETTm1 experiments completed!"
echo "Check the following log files for details:"
echo "- logs/ETTm1_no_label_loss.log"
echo "- logs/ETTm1_label_loss_0.1.log" 
echo "- logs/ETTm1_label_loss_0.5.log"
echo "- logs/ETTm1_label_loss_1.0.log"
echo "=========================================="
