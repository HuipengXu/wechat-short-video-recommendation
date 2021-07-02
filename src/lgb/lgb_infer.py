import pandas as pd
import lightgbm as lgb

from src.common_path import *

y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]

test_df = pd.read_csv(os.path.join(TRAIN_TEST_DATA_PATH, 'lgb_test.csv'))
cols = [f for f in test_df.columns if f not in ['date_'] + play_cols + y_list]

model1_path = os.path.join(MODEL_PATH, 'lgb_2024')
model2_path = os.path.join(MODEL_PATH, 'lgb_3000')

results1 = {}
results2 = {}

for y in y_list[:4]:
    print(f'Starting predicting {y} ...')
    act_model_path1 = os.path.join(model1_path, f'{y}.txt')
    act_model_path2 = os.path.join(model2_path, f'{y}.txt')
    clf1 = lgb.Booster(model_file=act_model_path1)
    clf2 = lgb.Booster(model_file=act_model_path2)
    results1[y] = clf1.predict(test_df[cols])
    results2[y] = clf2.predict(test_df[cols])

result1_df = test_df[['userid', 'feedid']]
result2_df = test_df[['userid', 'feedid']]

result1_df = pd.concat([result1_df, pd.DataFrame(data=results1)], axis=1)
result2_df = pd.concat([result2_df, pd.DataFrame(data=results2)], axis=1)

result1_df.to_csv(os.path.join(SUBMISSION_PATH, 'lgb1.csv'), index=False, encoding='utf8')
result2_df.to_csv(os.path.join(SUBMISSION_PATH, 'lgb2.csv'), index=False, encoding='utf8')
