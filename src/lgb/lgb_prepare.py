import gc
import json
import pickle
from tqdm import tqdm, trange
import pickle5 as pk5

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from src.common_path import *

pd.set_option('display.max_columns', None)

# feed embedding PCA 降维
feed_embedding = pd.read_csv(os.path.join(COMPETITION_DATA_PATH, 'feed_embeddings.csv'))

embeddings = []
for _, row in tqdm(feed_embedding.iterrows(), total=len(feed_embedding)):
    embeddings.append(np.fromstring(row.feed_embedding, dtype=float, sep=' '))
embeddings = np.stack(embeddings, axis=0)

pca = PCA(n_components=64, random_state=2021)
embeddings_pca = pca.fit_transform(embeddings)

embeddings_name = [f'emb_{i}' for i in range(64)]
feed_embedding[embeddings_name] = embeddings_pca
feed_embedding.drop('feed_embedding', axis=1, inplace=True)


def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


# 加载奇异值向量
user_token2id = load_json(os.path.join(UID_FID_SVD_PATH, 'user_token2id.json'))
uid_fid_svd = np.load(os.path.join(UID_FID_SVD_PATH, 'uid_svd_64.npy'))
feed_token2id = load_json(os.path.join(UID_FID_SVD_PATH, 'feed_token2id.json'))
fid_svd = np.load(os.path.join(UID_FID_SVD_PATH, 'fid_svd_64.npy')).T

author_token2id = load_json(os.path.join(UID_AID_SVD_PATH, 'author_token2id.json'))
uid_aid_svd = np.load(os.path.join(UID_AID_SVD_PATH, 'uid_svd_16.npy'))
aid_svd = np.load(os.path.join(UID_AID_SVD_PATH, 'aid_svd_16.npy')).T

keyword_token2id = load_json(os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'keyword_list_token2id.json'))
uid_keyword_list_svd = np.load(os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'uid_svd_16.npy'))
keyword_list_svd = np.load(os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'keyword_list_svd_16.npy')).T

tag_token2id = load_json(os.path.join(UID_TAG_LIST_SVD_PATH, 'tag_list_token2id.json'))
uid_tag_list_svd = np.load(os.path.join(UID_TAG_LIST_SVD_PATH, 'uid_svd_16.npy'))
tag_list_svd = np.load(os.path.join(UID_TAG_LIST_SVD_PATH, 'tag_list_svd_16.npy')).T

# 加载随机游走向量
userid_feedid_userid_deepwalk = pk5.load(
    open(os.path.join(DEEPWALK_PATH, 'userid_feedid_userid_dev_deepwalk_64.pkl'), 'rb')
)
userid_feedid_feedid_deepwalk = pk5.load(
    open(os.path.join(DEEPWALK_PATH, 'userid_feedid_feedid_dev_deepwalk_64.pkl'), 'rb')
)
userid_authorid_userid_deepwalk = pk5.load(
    open(os.path.join(DEEPWALK_PATH, 'userid_authorid_userid_dev_deepwalk_64.pkl'), 'rb')
)
userid_authorid_authorid_deepwalk = pk5.load(
    open(os.path.join(DEEPWALK_PATH, 'userid_authorid_authorid_dev_deepwalk_64.pkl'), 'rb')
)
userid_keyword_list_keyword_list_deepwalk = pk5.load(
    open(os.path.join(DEEPWALK_PATH, 'userid_keyword_list_keyword_list_dev_deepwalk_64.pkl'), 'rb')
)
userid_tag_list_tag_list_deepwalk = pk5.load(
    open(os.path.join(DEEPWALK_PATH, 'userid_tag_list_tag_list_dev_deepwalk_64.pkl'), 'rb')
)


# 加载 desc 等关键字相关
def load_embedding_matrix(filepath='', max_vocab_size=50000):
    embedding_matrix = np.load(filepath)
    flag_matrix = np.zeros_like(embedding_matrix[:2])
    return np.concatenate([flag_matrix, embedding_matrix])[:max_vocab_size]


class Vocab:
    PAD_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    MASKS = [PAD_TOKEN, UNKNOWN_TOKEN]
    MASK_COUNT = len(MASKS)

    PAD_TOKEN_INDEX = MASKS.index(PAD_TOKEN)
    UNKNOWN_TOKEN_INDEX = MASKS.index(UNKNOWN_TOKEN)

    def __init__(self, vocab_file, max_vocab_size=None):
        """Vocab 对象,vocab基本操作封装
        :param vocab_file: Vocab 存储路径
        :param max_vocab_size: 最大字典数量
        """
        self.word2id, self.id2word = self.load_vocab(vocab_file, max_vocab_size)
        self.count = len(self.word2id)

    @staticmethod
    def load_vocab(file_path, vocab_max_size=None):
        """读取字典
        :param file_path: 文件路径
        :param vocab_max_size:
        :return: 返回读取后的字典
        """
        vocab = {mask: index
                 for index, mask in enumerate(Vocab.MASKS)}

        reverse_vocab = {index: mask
                         for index, mask in enumerate(Vocab.MASKS)}

        if isinstance(file_path, str):
            token2id = json.load(open(file_path, 'r', encoding='utf8'))
        else:
            token2id = file_path

        for word, index in token2id.items():
            index = int(index)
            # 如果vocab 超过了指定大小
            # 跳出循环 截断
            if vocab_max_size and index >= vocab_max_size - Vocab.MASK_COUNT:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    vocab_max_size, index))
                break
            vocab[word] = index + Vocab.MASK_COUNT
            reverse_vocab[index + Vocab.MASK_COUNT] = word
        return vocab, reverse_vocab

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count


def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


# 读取训练集
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']
max_day = 15

train = pd.read_csv(os.path.join(COMPETITION_DATA_PATH, 'user_action.csv'))
print(train.shape)
for y in y_list:
    print(y, train[y].mean())
# 读取测试集
test = pd.read_csv(os.path.join(COMPETITION_DATA_PATH, 'test_b.csv'))
test['date_'] = max_day
print(test.shape)

# 合并处理
df = pd.concat([train, test], axis=0, ignore_index=True)
print(df.head(3))

# 读取视频信息表
feed_info = pd.read_csv(os.path.join(COMPETITION_DATA_PATH, 'feed_info.csv'))

feed_info = feed_info[[
    'feedid', 'authorid', 'videoplayseconds', 'manual_keyword_list',
    'manual_tag_list'
]]

df = df.merge(feed_info, on='feedid', how='left')
# 视频时长是秒，转换成毫秒，才能与play、stay做运算
df['videoplayseconds'] *= 1000
# 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
df['play_times'] = df['play'] / df['videoplayseconds']
play_cols = [
    'is_finish', 'play_times', 'play', 'stay'
]

print(df.head(3))

multi_modal_emb_matrix = load_embedding_matrix(
    filepath=os.path.join(W2V_PATH, 'embedding.npy'),
    max_vocab_size=70000
)
vocab = Vocab(os.path.join(W2V_PATH, 'token2id.json'), max_vocab_size=70000)
desc = []
kws = pickle.load(open(os.path.join(TFIDF_KWS_PATH, 'kws.pkl'), 'rb'))
for i in trange(len(kws['description'])):
    w2v = np.zeros(200)
    if kws['description'][i]:
        for token in kws['description'][i].split():
            id_ = vocab.word_to_id(token)
            w2v += multi_modal_emb_matrix[id_]
    elif kws['asr'][i]:
        for token in kws['asr'][i].split():
            id_ = vocab.word_to_id(token)
            w2v += multi_modal_emb_matrix[id_]
    else:
        for token in kws['ocr'][i].split():
            id_ = vocab.word_to_id(token)
            w2v += multi_modal_emb_matrix[id_]
    desc.append(w2v)
desc = np.stack(desc, axis=0)
pca = PCA(n_components=16, random_state=2021)
desc = pca.fit_transform(desc)
desc_name = [f'desc_{i}' for i in range(16)]
feed_info[desc_name] = desc
feed_desc_emb = feed_info[['feedid'] + [col for col in feed_info.columns if 'desc' in col]]

# 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
n_day = 5
for stat_cols in tqdm([
    ['userid'],
    ['feedid'],
    ['authorid'],
    ['userid', 'authorid'],
]):
    f = '_'.join(stat_cols)
    stat_df = pd.DataFrame()
    for target_day in range(2, max_day + 1):
        left, right = max(target_day - n_day, 1), target_day - 1
        tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
        tmp['date_'] = target_day
        tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')

        g = tmp.groupby(stat_cols)
        tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
        feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]

        for x in play_cols[1:]:
            for stat in ['max', 'mean']:
                tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))

        for y in y_list[:4]:
            tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
            tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
            feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])

        tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
        stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
        del g, tmp
    df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
    del stat_df
    gc.collect()

# 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行
feats = ['userid', 'feedid', 'authorid']
for f in tqdm(feats):
    df[f + '_count'] = df[f].map(df[f].value_counts())

for f1, f2 in tqdm([
    ['userid', 'feedid'],
    ['userid', 'authorid'],
]):
    df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')
    df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')

for f1, f2 in tqdm([
    ['userid', 'authorid'],
]):
    df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')
    df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
    df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)

df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')
df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')
df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')

total_svd_features = []
total_deepwalk_features = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    svd_features = []
    svd_features.append(uid_fid_svd[user_token2id[str(int(row.userid))]])
    svd_features.append(fid_svd[feed_token2id[str(int(row.feedid))]])
    svd_features.append(uid_aid_svd[user_token2id[str(int(row.userid))]])
    svd_features.append(aid_svd[author_token2id[str(int(row.authorid))]])

    kw_svd = np.zeros(16)
    if isinstance(row.manual_keyword_list, str):
        for kw in row.manual_keyword_list.split(';'):
            kw_svd += keyword_list_svd[keyword_token2id[kw]]
    svd_features.append(kw_svd)

    tag_svd = np.zeros(16)
    if isinstance(row.manual_tag_list, str):
        for tag in row.manual_tag_list.split(';'):
            tag_svd += tag_list_svd[tag_token2id[tag]]
    svd_features.append(tag_svd)

    svd_features = np.concatenate(svd_features)
    total_svd_features.append(svd_features)

    deepwalk_features = []
    deepwalk_features.append(userid_feedid_userid_deepwalk[int(row.userid)])
    deepwalk_features.append(userid_feedid_feedid_deepwalk[int(row.feedid)])
    deepwalk_features.append(userid_authorid_userid_deepwalk[int(row.userid)])
    deepwalk_features.append(userid_authorid_authorid_deepwalk[int(row.authorid)])

    deepwalk_features = np.concatenate(deepwalk_features)
    total_deepwalk_features.append(deepwalk_features)

total_svd_features = np.stack(total_svd_features, axis=0)
total_deepwalk_features = np.stack(total_deepwalk_features, axis=0)

svd_names = [f'svd_{i}' for i in range(total_svd_features.shape[1])]
svd_df = pd.DataFrame(data=total_svd_features, columns=svd_names)
deepwalk_names = [f'deepwalk_{i}' for i in range(total_deepwalk_features.shape[1])]
deepwalk_df = pd.DataFrame(data=total_deepwalk_features, columns=deepwalk_names)
del total_svd_features, total_deepwalk_features
gc.collect()

# 内存够用的不需要做这一步
df.drop(['manual_keyword_list', 'manual_tag_list'], axis=1, inplace=True)
df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])
feed_embedding = reduce_mem(feed_embedding, feed_embedding.columns)
feed_desc_emb = reduce_mem(feed_desc_emb, feed_desc_emb.columns)
svd_df = reduce_mem(svd_df, svd_df.columns)
deepwalk_df = reduce_mem(deepwalk_df, deepwalk_df.columns)

df = pd.concat([df, svd_df, deepwalk_df], axis=1)
df = pd.merge(df, feed_embedding, on='feedid', how='left')
df = pd.merge(df, feed_desc_emb, how='left', on='feedid')
del feed_embedding, feed_desc_emb, svd_df, deepwalk_df
gc.collect()

train = df[~df['read_comment'].isna()].reset_index(drop=True)
test = df[df['read_comment'].isna()].reset_index(drop=True)

train.to_csv(os.path.join(TRAIN_TEST_DATA_PATH, 'lgb_train.csv'), index=False, encoding='utf8')
test.to_csv(os.path.join(TRAIN_TEST_DATA_PATH, 'lgb_test.csv'), index=False, encoding='utf8')

print('Finished saving lgb data ...')
