python -m src.common_path
python -m src.prepare
python -m src.lgb.lgb_prepare

python -m src.lgb.lgb_train \
  --learning_rate 0.01 \
  --n_estimators 5000 \
  --num_leaves 49 \
  --subsample 0.65 \
  --colsample_bytree 0.65 \
  --random_state 2024

python -m src.lgb.lgb_train \
  --learning_rate 0.02 \
  --n_estimators 6000 \
  --num_leaves 49 \
  --subsample 0.65 \
  --colsample_bytree 0.65 \
  --random_state 3000
  

  #####################################################################
  #
  #                 nn training shell
  #
  #####################################################################

for item_seed in 12 13 14 15 16
do 
  python -m src.nn.train \
    --num_epochs 3 \
    --lr 0.01 \
    --bs 512 \
    --ema_start_step 1000 \
    --logging_step 1000 \
    --action '' \
    --l2_reg_embedding 0.1 \
    --l2 0.0001 \
    --multi_modal_hidden_size 128  \
    --seed $item_seed
done