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

### 1. LightGBM

* 参数 1：
  * learning_rate: 0.01
  * n_estimators: 5000
  * num_leaves: 49
  * subsample: 0.65
  * colsample_bytree: 0.65
  * random_state: 2024
  * eval_metric: auc
  * early_stoppping_rounds: 100
* 参数 2：
  * learning_rate: 0.02
  * n_estimators: 6000
  * num_leaves: 49
  * subsample: 0.65
  * colsample_bytree: 0.65
  * random_state: 3000
  * eval_metric: auc
  * early_stoppping_rounds: 100
* 特征
  * 原始特征 (7 个)
    * userid
    * feedid
    * device
    * authorid
    * videoplayseconds
    * play
    * stay
  * 视频相关信息转换特征 (2 个)
    * is_finished: 播放时长大于视频时长表示完成播放
    * play_times: 播放时长除以视频时长
  * 统计特征 (52 个)
    * userid、feedid、authorid、userid 与 authorid：(64 个)
      * 历史 5 天出现的次数
      * 历史 5 天的播放完成率
      * 历史 5 天的播放次数、播放时长、停留时长的最大值和均值
      * 历史 5 天 4 种动作的正样本数量和发生率
    * userid、feedid、authorid 的全局出现次数 (3 个)
    * userid 和 feedid 、userid 和 authorid 各自在对方 group 中 id 的个数 (4 个)
    * userid 和 authorid 共现的次数以及共现比例 (3 个)
    * userid 播放时长的均值
    * authorid 播放时长的均值
    * authorid 对应的 feedid 数量
  * SVD 特征 (192 个)
    * userid 与 feedid 的奇异值分解 (64 维)
    * userid 与 authorid 的奇异值分解 (16 维)
    * userid 与 keyword  的奇异值分解 (16 维)
    * userid 与 tag  的奇异值分解 (16 维)
  * DeepWalk 特征 (256)
    * userid 与 feedid  (64 维)
    * userid 与 authorid  (64 维)
  * feed embedding 的 PCA 降维 (64 维)
  * 词级别 description、ocr、asr 经过 tfidf 权重筛选后的关键字词向量之和 (200 维 PCA 降维到 16 维)

