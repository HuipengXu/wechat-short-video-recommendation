## 1. 环境配置
* pandas==1.0.5
* tqdm==4.47.0
* matplotlib==3.2.2
* scipy==1.5.0
* numpy==1.20.1
* datatable==0.11.1
* gensim==4.0.1
* lightgbm==3.2.1
* pickle5==0.0.11
* scikit_learn==0.24.2

## 2. 代码结构

.
├── README.md
├── data
│   ├── deepwalk
│   ├── desc_ocr_asr_200d
│   ├── desc_ocr_asr_char_200d
│   ├── models
│   ├── processed_data
│   ├── submission
│   ├── tfidf_kws
│   ├── uid_aid_svd
│   ├── uid_bgm_singer_id_svd
│   ├── uid_bgm_song_id_svd
│   ├── uid_fid_svd
│   ├── uid_keyword_list_svd
│   ├── uid_tag_list_svd
│   └── wedata
│       └── wechat_algo_data1
├── inference.sh
├── init.sh
├── requirements.txt
├── src
│   ├── \_\_init\_\_.py
│   ├── common_path.py
│   ├── lgb
│   │   ├── \_\_init\_\_.py
│   │   ├── lgb_infer.py
│   │   ├── lgb_prepare.py
│   │   └── lgb_train.py
│   └── prepare.py
└── train.sh

## 3. 运行流程

1. 安装环境：bash init.sh
2. 数据准备及模型训练：bash train.sh
3. 预测并生成结果文件：bash inference.sh

## 4. 模型及特征

