export CUDA_VISIBLE_DEVICES=0

model_name=FS_Mamba2

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_1024 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 1024 \
  --e_layers 1 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --d_state 32 \
  --train_epochs 5 \
  --batch_size 16 \
  --learning_rate 0.0005 \
  --itr 1