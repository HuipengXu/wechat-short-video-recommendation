echo "LGB inference ..."

python -m src.lgb.lgb_infer

echo "NN inference ..."
for item_seed in 12 13 14 15 16
do
  python -m src.nn.test \
    --bs 512 \
    --action '' \
    --l2_reg_embedding 0.1 \
    --l2 0.0001 \
    --multi_modal_hidden_size 128 \
    --seed $item_seed
done

python -m src.ensemble
