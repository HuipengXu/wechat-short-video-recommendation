import os
import gc
import numpy as np
from tqdm import tqdm, trange
from src.common_path import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import (
    seed_everything, roc_score, load_embedding_matrix, EMA, ProcessedData,
    batch2cuda, WXDataset, Collator, roc_score_one_act, reset_args, undersampling_neg_sample, read_data,
    undersampling_neg_sample_all_actions
)

import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from .multideepfm4wx import MultiDeepFM
from .args import get_args

ACTIONS = ['read_comment', 'like', 'click_avatar', 'forward']

def prepare_data(args):
    data_df = read_data(args.merge_data_path, args.debug_data)
    processed_data = ProcessedData(args)
    collate_fn = Collator(args)
    return collate_fn, data_df, processed_data


def args_setup():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        args.n_gpus = torch.cuda.device_count()
        args.bs *= args.n_gpus
    args.multi_modal_emb_matrix = load_embedding_matrix(
        filepath=args.multi_modal_emb_matrix_path,
        max_vocab_size=args.vocab_size
    )
    args.multi_modal_emb_char_matrix = load_embedding_matrix(
        filepath=args.multi_modal_emb_char_matrix_path,
        max_vocab_size=args.char_vocab_size
    )
    return args

def predict(config, model, test_dataloader):
    test_iterator = tqdm(test_dataloader, desc='Predicting', total=len(test_dataloader))
    test_preds = []

    model.eval()
    with torch.no_grad():
        for batch in test_iterator:
            batch_cuda, _ = batch2cuda(batch, config)
            logits = model(**batch_cuda)[4]
            probs = torch.sigmoid(logits)
            test_preds.append(probs.detach().cpu())

    test_preds = torch.cat(test_preds).numpy()
    test_df = pd.read_csv(config.submit_test_path)
    if config.debug_data:
        test_df = test_df.head(1000)
    test_df[['read_comment', 'like', 'click_avatar', 'forward']] = test_preds
    test_df.drop('device', axis=1, inplace=True)
    submission_path = os.path.join(SUBMISSION_PATH, "nn{}.csv".format(config.seed))
    test_df.to_csv(submission_path, index=False, encoding='utf8', sep=',')


def main():
    args = args_setup()

    collate_fn, data_df, processed_data = prepare_data(args)

    test_df = read_data(args.test_data_path, args.debug_data)
    mms = MinMaxScaler(feature_range=(0, 1))
    data_df[['videoplayseconds']] = mms.fit_transform(data_df[['videoplayseconds']])
    test_df[['videoplayseconds']] = mms.transform(test_df[['videoplayseconds']])

    merge_all_data = pd.concat([data_df, test_df], axis=0, ignore_index=True)
    merge_all_data["user_seq"] = merge_all_data.groupby(["userid"]).cumcount() + 1

    data_df = merge_all_data[~merge_all_data['read_comment'].isna()].reset_index(drop=True)
    test_df = merge_all_data[merge_all_data['read_comment'].isna()].reset_index(drop=True)
    test_df = test_df.dropna(axis=1, how="all")
    test_dataloader = DataLoader(WXDataset(test_df, processed_data), batch_size=args.bs, shuffle=False,
                                    collate_fn=collate_fn, num_workers=4, pin_memory=False)
    model = MultiDeepFM(args)
    model.to(args.device)
    args.best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    model.load_state_dict(torch.load(args.best_model_path))
    predict(args, model, test_dataloader)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
