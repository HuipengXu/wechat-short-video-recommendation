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
