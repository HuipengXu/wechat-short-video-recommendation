import os
import gc
import time
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import roc_auc_score

from src.common_path import *

pd.set_option('display.max_columns', None)


# 从官方baseline里面抽出来的评测函数
def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(list)
    user_truth = defaultdict(list)
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc
            size += 1.0
    user_auc = float(total_auc) / size
    return user_auc


parser = argparse.ArgumentParser(description='For lgb')
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--n_estimators', default=5000, type=int)
parser.add_argument('--num_leaves', default=49, type=int)
parser.add_argument('--subsample', default=0.65, type=float)
parser.add_argument('--colsample_bytree', default=0.65, type=float)
parser.add_argument('--random_state', default=2024, type=int)
args = parser.parse_args()

y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]

print('Reading data ...')
train = pd.read_csv(os.path.join(TRAIN_TEST_DATA_PATH, 'lgb_train.csv')).sample(frac=0.2)
test = pd.read_csv(os.path.join(TRAIN_TEST_DATA_PATH, 'lgb_test.csv'), nrows=1000)

cols = [f for f in train.columns if f not in ['date_'] + play_cols + y_list]
print(train[cols].shape)

trn_x = train[train['date_'] < 14].reset_index(drop=True)
val_x = train[train['date_'] == 14].reset_index(drop=True)

params = {
    'learning_rate': args.learning_rate,
    'n_estimators': args.n_estimators,
    'num_leaves': args.num_leaves,
    'subsample': args.subsample,
    'colsample_bytree': args.colsample_bytree,
    'random_state': args.random_state,
    'metric': None,
}

fit_params = {
    'eval_metric': 'auc',
    'early_stopping_rounds': 100
}

##################### 线下验证 #####################
uauc_list = []
r_list = []

print('Starting offline training ...')
for y in y_list[:4]:
    print('=========', y, '=========')
    t = time.time()
    clf = LGBMClassifier(**params)

    clf.fit(
        trn_x[cols], trn_x[y],
        eval_set=[(val_x[cols], val_x[y])],
        eval_metric=fit_params['eval_metric'],
        early_stopping_rounds=fit_params['early_stopping_rounds'],
        verbose=50
    )

    val_x[y + '_score'] = clf.predict_proba(val_x[cols])[:, 1]
    val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])
    print(f'{y}: {val_uauc:.6f}')
    uauc_list.append(val_uauc)
    r_list.append(clf.best_iteration_)
    print('runtime: {}\n'.format(time.time() - t))

    del clf
    gc.collect()

weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]

print(uauc_list)
print(weighted_uauc)

del trn_x, val_x
gc.collect()

models_dir = os.path.join(MODEL_PATH, f'lgb_{args.random_state}')
os.makedirs(models_dir, exist_ok=True)
r_dict = dict(zip(y_list[:4], r_list))
print('Starting online training ...')
for y in y_list[:4]:
    print('=========', y, '=========')
    t = time.time()
    params['n_estimators'] = r_dict[y]
    clf = LGBMClassifier(**params)

    clf.fit(
        train[cols], train[y],
        eval_set=[(train[cols], train[y])],
        early_stopping_rounds=r_dict[y],
        verbose=100
    )

    clf.booster_.save_model(os.path.join(models_dir, f'{y}.txt'), num_iteration=r_dict[y])

    print('runtime: {}\n'.format(time.time() - t))

    del clf
    gc.collect()
