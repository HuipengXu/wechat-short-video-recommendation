import os

DATA_ROOT = './data'
COMPETITION_DATA_PATH = os.path.join(DATA_ROOT, 'wedata/wechat_algo_data1')
TRAIN_TEST_DATA_PATH = os.path.join(DATA_ROOT, 'processed_data')

# svd
UID_FID_SVD_PATH = os.path.join(DATA_ROOT, 'uid_fid_svd')
UID_AID_SVD_PATH = os.path.join(DATA_ROOT, 'uid_aid_svd')
UID_BGM_SONG_ID_SVD_PATH = os.path.join(DATA_ROOT, 'uid_bgm_song_id_svd')
UID_BGM_SINGER_ID_SVD_PATH = os.path.join(DATA_ROOT, 'uid_bgm_singer_id_svd')
UID_TAG_LIST_SVD_PATH = os.path.join(DATA_ROOT, 'uid_tag_list_svd')
UID_KEYWORD_LIST_SVD_PATH = os.path.join(DATA_ROOT, 'uid_keyword_list_svd')

# deepwalk
DEEPWALK_PATH = os.path.join(DATA_ROOT, 'deepwalk')

# w2v
W2V_PATH = os.path.join(DATA_ROOT, 'desc_ocr_asr_200d')
W2V_CHAR_PATH = os.path.join(DATA_ROOT, 'desc_ocr_asr_char_200d')

MODEL_PATH = os.path.join(DATA_ROOT, 'models')
SUBMISSION_PATH = os.path.join(DATA_ROOT, 'submission')

# tfidf-kws
TFIDF_KWS_PATH = os.path.join(DATA_ROOT, 'tfidf_kws')


def main():
    locs = locals().copy()
    for name, path in locs.items():
        if 'PATH' in name:
            os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    main()
