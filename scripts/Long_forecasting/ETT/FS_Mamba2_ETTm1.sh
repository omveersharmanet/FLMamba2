export CUDA_VISIBLE_DEVICES=0

model_name=FS_Mamba2
# d state 2
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_1024 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 1024 \
  --e_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1