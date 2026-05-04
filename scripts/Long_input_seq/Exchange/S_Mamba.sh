export CUDA_VISIBLE_DEVICES=0

model_name=S_Mamba

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_48_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 16\
  --learning_rate 0.0001 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_192_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 192 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 16\
  --learning_rate 0.0001 \
  --d_ff 128 \
  --itr 1

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_336_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 16\
  --learning_rate 0.0001 \
  --d_ff 128 \
  --itr 1

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_720_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 128 \
  --batch_size 16\
  --learning_rate 0.0001 \
  --d_ff 128 \
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_48_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_192_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 192 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_336_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

  python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_720_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --learning_rate 0.00005 \
  --d_model 128 \
  --d_ff 128 \
  --itr 1

