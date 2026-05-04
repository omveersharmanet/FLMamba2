export CUDA_VISIBLE_DEVICES=0

model_name=FS_Mamba2


python -u run.py \
  --is_training 1 \
  --model_id ALL_96_96 \
  --model $model_name \
  --data ALL \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --seq_len_case 1 \
  --label_len_case 1 \
  --pred_len_case 1 \
  --e_layers 1 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 8\
  --learning_rate 0.0001 \
  --d_ff 512 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --model_id ALL_96_192 \
  --model $model_name \
  --data ALL \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --seq_len_case 1 \
  --label_len_case 1 \
  --pred_len_case 2 \
  --e_layers 1 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 8\
  --learning_rate 0.0001 \
  --d_ff 512 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --model_id ALL_96_336 \
  --model $model_name \
  --data ALL \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --seq_len_case 1 \
  --label_len_case 1 \
  --pred_len_case 3 \
  --e_layers 1 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 8\
  --learning_rate 0.0001 \
  --d_ff 512 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --model_id ALL_96_720 \
  --model $model_name \
  --data ALL \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --seq_len_case 1 \
  --label_len_case 1 \
  --pred_len_case 4 \
  --e_layers 1 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --d_model 512 \
  --batch_size 8\
  --learning_rate 0.0001 \
  --d_ff 512 \
  --itr 1
