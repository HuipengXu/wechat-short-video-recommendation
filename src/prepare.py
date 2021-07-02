import os
import gc
import json
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from collections import Counter, defaultdict
from sklearn.utils.extmath import randomized_svd
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import LineSentence, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

from src.common_path import *

# 缺失值唯一编码
no_bgm_song_id = 25159
no_bgm_singer_id = 17500


def save_json(path, content):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def merge_keyword(row):
    kws = set()
    if isinstance(row[-6], str):
        kws |= set(row[-6].split(';'))
    if isinstance(row[-7], str):
        kws |= set(row[-7].split(';'))
    return ';'.join(kws)


def merge_tag(row):
    tgs = set()
    if isinstance(row[-5], str):
        tgs |= set([tg.split()[0] for tg in row[-5].split(';') if float(tg.split()[-1]) > 0.8])
    if isinstance(row[-6], str):
        tgs |= set(row[-6].split(';'))
    return ';'.join(tgs)


def fillna_bgm(df):
    for authorid in tqdm(df.authorid.unique()):
        cur_author_df = df.loc[df.authorid == authorid]

        num_song_id_null = cur_author_df.bgm_song_id.isna().sum()
        if num_song_id_null == len(cur_author_df):
            df.loc[df.authorid == authorid, 'bgm_song_id'] = no_bgm_song_id
        else:
            max_freq_song_id = cur_author_df.bgm_song_id.value_counts().index[0]
            df.loc[df.authorid == authorid, 'bgm_song_id'] = df.loc[df.authorid == authorid, 'bgm_song_id'].fillna(
                max_freq_song_id)

        num_singer_id_null = cur_author_df.bgm_singer_id.isna().sum()
        if num_singer_id_null == len(cur_author_df):
            df.loc[df.authorid == authorid, 'bgm_singer_id'] = no_bgm_singer_id
        else:
            max_freq_singer_id = cur_author_df.bgm_singer_id.value_counts().index[0]
            df.loc[df.authorid == authorid, 'bgm_singer_id'] = df.loc[df.authorid == authorid, 'bgm_singer_id'].fillna(
                max_freq_singer_id)


def build_inter_matrix(total_df, row_item, col_item, row_vocab, col_vocab):
    matrix = np.zeros((len(row_vocab), len(col_vocab)))
    for _, row in tqdm(total_df.iterrows(), total=len(total_df), desc='Building interaction matrix'):
        row_val, col_val = row[row_item], row[col_item]
        matrix[row_vocab[row_val], col_vocab[col_val]] += 1
    return matrix


def svd(inter_matrix, dim):
    u, sigma, vt = randomized_svd(inter_matrix, n_components=dim, n_iter=20, random_state=2021)
    return u, vt


def build_tag_and_kw_vocab(data, col):
    counter = Counter()
    for row in tqdm(data):
        tokens = row.split(';')
        counter.update(tokens)
    unique_tokens = [token for token, _ in counter.most_common()]
    token2id = {token: i for i, token in enumerate(unique_tokens)}
    save_json(f'./data/uid_{col}_svd/{col}_token2id.json', token2id)


def build_kw_tag_inter_matrix(total_df, row_item, col_item, row_vocab, col_vocab):
    matrix = np.zeros((len(row_vocab), len(col_vocab)))
    for _, row in tqdm(total_df.iterrows(), total=len(total_df), desc='Building interaction matrix'):
        row_val, col_vals = row[row_item], row[col_item]
        for val in col_vals.split(';'):
            matrix[row_vocab[row_val], col_vocab[val]] += 1
    return matrix


testa_df = pd.read_csv(os.path.join(COMPETITION_DATA_PATH, 'test_a.csv'))
testb_df = pd.read_csv(os.path.join(COMPETITION_DATA_PATH, 'test_b.csv'))

user_df = pd.read_csv(os.path.join(COMPETITION_DATA_PATH, 'user_action.csv'))
feed_df = pd.read_csv(os.path.join(COMPETITION_DATA_PATH, 'feed_info.csv'))
fillna_bgm(feed_df)
feed_df['keyword_list'] = feed_df.apply(merge_keyword, axis=1)
feed_df['tag_list'] = feed_df.apply(merge_tag, axis=1)

merge_df = pd.merge(user_df, feed_df, how='left', on='feedid')
testa_merge_df = pd.merge(testa_df, feed_df, how='left', on='feedid')
testb_merge_df = pd.merge(testb_df, feed_df, how='left', on='feedid')

merge_df.to_csv(os.path.join(TRAIN_TEST_DATA_PATH, 'nn_train.csv'), index=False, encoding='utf8', sep=',')
testb_merge_df.to_csv(os.path.join(TRAIN_TEST_DATA_PATH, 'nn_test.csv'), index=False, encoding='utf8', sep=',')

total_df = pd.concat([merge_df, testa_merge_df, testb_merge_df], axis=0, ignore_index=True)

# uid-fid
user_token2id = {int(token): i for i, token in enumerate(user_df.userid.unique())}
user_id2token = {i: int(token) for i, token in enumerate(user_df.userid.unique())}

feed_token2id = {int(token): i for i, token in enumerate(feed_df.feedid.unique())}
feed_id2token = {i: int(token) for i, token in enumerate(feed_df.feedid.unique())}

uid_fid_matrix = build_inter_matrix(total_df, 'userid', 'feedid', user_token2id, feed_token2id)
uid_fid_sparse = sparse.csr_matrix(uid_fid_matrix)
u_svd_vector, f_svd_vector = svd(uid_fid_sparse, dim=64)

np.save(os.path.join(UID_FID_SVD_PATH, 'uid_svd_64.npy'), u_svd_vector)
np.save(os.path.join(UID_FID_SVD_PATH, 'fid_svd_64.npy'), f_svd_vector)
save_json(os.path.join(UID_FID_SVD_PATH, 'user_token2id.json'), user_token2id)
save_json(os.path.join(UID_FID_SVD_PATH, 'feed_token2id.json'), feed_token2id)

# uid-aid
author_token2id = {int(token): i for i, token in enumerate(feed_df.authorid.unique())}
author_id2token = {i: int(token) for i, token in enumerate(feed_df.authorid.unique())}

uid_aid_matrix = build_inter_matrix(total_df, 'userid', 'authorid', user_token2id, author_token2id)
uid_aid_sparse = sparse.csr_matrix(uid_aid_matrix)
u_svd_vector, a_svd_vector = svd(uid_aid_sparse, dim=16)

np.save(os.path.join(UID_AID_SVD_PATH, 'uid_svd_16.npy'), u_svd_vector)
np.save(os.path.join(UID_AID_SVD_PATH, 'aid_svd_16.npy'), a_svd_vector)
save_json(os.path.join(UID_AID_SVD_PATH, 'user_token2id.json'), user_token2id)
save_json(os.path.join(UID_AID_SVD_PATH, 'author_token2id.json'), author_token2id)

# uid-bgm_song_id
bgm_song_token2id = {int(token): i for i, token in enumerate(feed_df.bgm_song_id.unique())}
bgm_song_id2token = {i: int(token) for i, token in enumerate(feed_df.bgm_song_id.unique())}

uid_bgm_song_id_matrix = build_inter_matrix(total_df, 'userid', 'bgm_song_id', user_token2id, bgm_song_token2id)
uid_bgm_song_id_sparse = sparse.csr_matrix(uid_bgm_song_id_matrix)
u_svd_vector, bgm_song_id_svd_vector = svd(uid_bgm_song_id_sparse, dim=16)

np.save(os.path.join(UID_BGM_SONG_ID_SVD_PATH, 'uid_svd_16.npy'), u_svd_vector)
np.save(os.path.join(UID_BGM_SONG_ID_SVD_PATH, 'bgm_song_id_svd_16.npy'), bgm_song_id_svd_vector)
save_json(os.path.join(UID_BGM_SONG_ID_SVD_PATH, 'user_token2id.json'), user_token2id)
save_json(os.path.join(UID_BGM_SONG_ID_SVD_PATH, 'bgm_song_token2id.json'), bgm_song_token2id)

# uid-bgm_singer_id
bgm_singer_token2id = {int(token): i for i, token in enumerate(feed_df.bgm_singer_id.unique())}
bgm_singer_id2token = {i: int(token) for i, token in enumerate(feed_df.bgm_singer_id.unique())}

uid_bgm_singer_id_matrix = build_inter_matrix(total_df, 'userid', 'bgm_singer_id', user_token2id, bgm_singer_token2id)
uid_bgm_singer_id_sparse = sparse.csr_matrix(uid_bgm_singer_id_matrix)
u_svd_vector, bgm_singer_id_svd_vector = svd(uid_bgm_singer_id_sparse, dim=16)

np.save(os.path.join(UID_BGM_SINGER_ID_SVD_PATH, 'uid_svd_16.npy'), u_svd_vector)
np.save(os.path.join(UID_BGM_SINGER_ID_SVD_PATH, 'bgm_singer_id_svd_16.npy'), bgm_singer_id_svd_vector)
save_json(os.path.join(UID_BGM_SINGER_ID_SVD_PATH, 'user_token2id.json'), user_token2id)
save_json(os.path.join(UID_BGM_SINGER_ID_SVD_PATH, 'bgm_singer_token2id.json'), bgm_singer_token2id)

# tag list
build_tag_and_kw_vocab(feed_df.tag_list.to_list(), 'tag_list')
tag_token2id = json.load(open(os.path.join(UID_TAG_LIST_SVD_PATH, 'tag_list_token2id.json'), 'r', encoding='utf8'))
uid_tag_list_matrix = build_kw_tag_inter_matrix(total_df, 'userid', 'tag_list', user_token2id, tag_token2id)
uid_tag_list_sparse = sparse.csr_matrix(uid_tag_list_matrix)
u_svd_vector, tag_list_svd_vector = svd(uid_tag_list_sparse, dim=16)

np.save(os.path.join(UID_TAG_LIST_SVD_PATH, 'uid_svd_16.npy'), u_svd_vector)
np.save(os.path.join(UID_TAG_LIST_SVD_PATH, 'tag_list_svd_16.npy'), tag_list_svd_vector)

# kw_list
build_tag_and_kw_vocab(total_df.keyword_list.to_list(), 'keyword_list')
keyword_token2id = json.load(open(os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'keyword_list_token2id.json'),
                                  'r', encoding='utf8'))
uid_keyword_list_matrix = build_kw_tag_inter_matrix(total_df, 'userid', 'keyword_list', user_token2id, keyword_token2id)
uid_keyword_list_sparse = sparse.csr_matrix(uid_keyword_list_matrix)
u_svd_vector, keyword_list_svd_vector = svd(uid_keyword_list_sparse, dim=16)

np.save(os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'uid_svd_16.npy'), u_svd_vector)
np.save(os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'keyword_list_svd_16.npy'), keyword_list_svd_vector)

del total_df
gc.collect()


# deepwalk
def build_graph(df, graph, f1, f2):
    for item in df[[f1, f2]].values:
        if isinstance(item[1], str):
            if not item[1]: continue
            for token in item[1].split(';'):
                graph['item_' + token].add('user_' + str(item[0]))
                graph['user_' + str(item[0])].add('item_' + token)
        else:
            graph['item_' + str(item[1])].add('user_' + str(item[0]))
            graph['user_' + str(item[0])].add('item_' + str(item[1]))


def deepwalk(train_df, test_df, f1, f2, flag, emb_size):
    print("deepwalk:", f1, f2)
    print('Building graph ...')
    graph = defaultdict(set)
    build_graph(train_df, graph, f1, f2)
    build_graph(test_df, graph, f1, f2)

    print("Creating corpus ...")
    path_length = 10
    sentences = []
    length = []
    keys = graph.keys()
    for key in tqdm(keys, total=len(keys), desc='Walk'):
        sentence = [key]
        while len(sentence) != path_length:
            key = random.sample(graph[sentence[-1]], 1)[0]
            if len(sentence) >= 2 and key == sentence[-2]:
                break
            else:
                sentence.append(key)

        sentences.append(sentence)
        length.append(len(sentence))
        if len(sentences) % 100000 == 0:
            print(len(sentences))

    print(f'Mean sentences length: {np.mean(length)}')
    print(f'Total {len(sentences)} sentences ...')
    print('Training word2vec ...')
    random.shuffle(sentences)
    model = Word2Vec(sentences, vector_size=emb_size, window=4, min_count=1, sg=1, workers=10, epochs=20)
    print('Outputing ...')

    values = np.unique(np.concatenate([np.unique(train_df[f1]), np.unique(test_df[f1])], axis=0))

    w2v = {}
    for v in values:
        if 'user_' + str(v) not in model.wv:
            vec = np.zeros(emb_size)
        else:
            vec = model.wv['user_' + str(v)]
        w2v[v] = vec
    pickle.dump(
        w2v,
        open(
            os.path.join(DEEPWALK_PATH, f1 + '_' + f2 + '_' + f1 + '_' + flag + '_deepwalk_' + str(emb_size) + '.pkl'),
            'wb'
        )
    )

    if 'list' in f2:
        values = [items.split(';') for items in train_df[f2].values] + [items.split(';') for items in
                                                                        test_df[f2].values]
        values = set([token for v in values for token in v])
    else:
        values = np.unique(np.concatenate([np.unique(train_df[f2]), np.unique(test_df[f2])], axis=0))

    w2v = {}
    for v in values:
        if 'item_' + str(v) not in model.wv:
            vec = np.zeros(emb_size)
        else:
            vec = model.wv['item_' + str(v)]
        w2v[v] = vec
    pickle.dump(
        w2v,
        open(
            os.path.join(DEEPWALK_PATH, f1 + '_' + f2 + '_' + f2 + '_' + flag + '_deepwalk_' + str(emb_size) + '.pkl'),
            'wb'
        )
    )


merge_test_df = pd.concat([testa_merge_df, testb_merge_df], axis=0, ignore_index=True)

deepwalk(merge_df, merge_test_df, 'userid', 'feedid', 'dev', 64)
deepwalk(merge_df, merge_test_df, 'userid', 'authorid', 'dev', 64)
deepwalk(merge_df, merge_test_df, 'userid', 'bgm_song_id', 'dev', 64)
deepwalk(merge_df, merge_test_df, 'userid', 'bgm_singer_id', 'dev', 64)
deepwalk(merge_df, merge_test_df, 'userid', 'keyword_list', 'dev', 64)
deepwalk(merge_df, merge_test_df, 'userid', 'tag_list', 'dev', 64)


# W2V
class LossLogger(CallbackAny2Vec):
    """Output loss at each epoch"""

    def __init__(self):
        self.epoch = 1
        self.losses = [0]

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {self.losses[-1] - self.losses[-2]}')
        self.epoch += 1


def w2v(data, max_len, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    print(f'Processing data, total {len(data)} sentences ...')
    data = data.str.split().apply(lambda x: x[:max_len]).tolist()
    print('Starting training w2v ...')
    wv_model = Word2Vec(data,
                        vector_size=200,
                        sg=1,
                        workers=6,
                        window=5,
                        epochs=10,
                        min_count=3,
                        seed=2021,
                        compute_loss=True,
                        callbacks=[LossLogger()])
    wv_model.wv.save_word2vec_format(os.path.join(save_dir, 'w2v.bin'))

    token2id = {word: index for index, word in enumerate(wv_model.wv.index_to_key)}
    id2token = {index: word for index, word in enumerate(wv_model.wv.index_to_key)}
    save_json(os.path.join(save_dir, 'token2id.json'), token2id)
    save_json(os.path.join(save_dir, 'id2token.json'), id2token)

    embedding_matrix = wv_model.wv.vectors
    np.save(os.path.join(save_dir, 'embedding.npy'), embedding_matrix)

    print(f'finish {save_dir} ...')


description = feed_df.description.dropna().drop_duplicates()
ocr = feed_df.ocr.dropna().drop_duplicates()
asr = feed_df.asr.dropna().drop_duplicates()

description_char = feed_df.description_char.dropna().drop_duplicates()
ocr_char = feed_df.ocr_char.dropna().drop_duplicates()
asr_char = feed_df.asr_char.dropna().drop_duplicates()

desc_ocr_asr = (description.append(ocr)).append(asr)
desc_ocr_asr_char = (description_char.append(ocr_char)).append(asr_char)

w2v(desc_ocr_asr, 560, W2V_PATH)
w2v(desc_ocr_asr_char, 656, W2V_CHAR_PATH)


# tfidf
def extract_kws(type_, feed_df):
    corpus = feed_df[[f'ocr{type_}', f'asr{type_}', f'description{type_}']].fillna(' ').values.tolist()
    corpus = [sent for c in corpus for sent in c]
    vectorizer = TfidfVectorizer(max_df=0.75, min_df=7)
    x = vectorizer.fit_transform(corpus)
    id2token = {id_: token for token, id_ in vectorizer.vocabulary_.items()}
    kws = defaultdict(list)

    for i, row in tqdm(enumerate(x)):
        if corpus[i].strip():
            weight = row.toarray()[0]
            weight_sum = weight.sum()
            if weight_sum != 0:
                weight = weight / weight_sum
                index = np.argsort(weight).tolist()[::-1]
                weight_sorted = np.sort(weight).tolist()[::-1]
                cumsum_weight_sorted = np.cumsum(weight_sorted)
                mask = (cumsum_weight_sorted > 0.6).tolist()
                idx = mask.index(True)
                kw = [id2token[index[j]] for j in range(idx)]
            else:
                kw = []
        else:
            kw = []
        if i % 3 == 0:
            kws['ocr'].append(' '.join(kw))
        elif i % 3 == 1:
            kws['asr'].append(' '.join(kw))
        elif i % 3 == 2:
            kws['description'].append(' '.join(kw))
    pickle.dump(kws, open(os.path.join(TFIDF_KWS_PATH, f'kws{type_}.pkl'), 'wb'))


extract_kws('', feed_df)
extract_kws('_char', feed_df)
