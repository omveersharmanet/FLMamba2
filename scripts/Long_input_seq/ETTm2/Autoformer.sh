export CUDA_VISIBLE_DEVICES=0

model_name=Autoformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_48_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_192_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 192 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 720 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --itr 1



python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_48_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_192_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 192 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 720 \
  --pred_len 720 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_48_1024 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 48 \
  --pred_len 1024 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_192_1024 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 192 \
  --pred_len 1024 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_336_1024 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 336 \
  --pred_len 1024 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_720_1024 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 720 \
  --pred_len 1024 \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_state 2 \
  --learning_rate 0.00005 \
  --d_ff 128 \
  --itr 1


