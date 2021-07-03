## 1. 环境配置
* numba==0.53.1
* scipy==1.4.1
* numpy==1.18.5
* lightgbm==3.2.1
* comet_ml==3.10.0
* transformers==4.3.3
* tqdm==4.60.0
* pickle5==0.0.11
* gensim==4.0.1
* torch==1.8.0
* pandas==1.1.5
* datatable==1.0.0
* scikit_learn==0.24.2


## 2. 代码结构

```shell
.
├── README.md
├── data
│   ├── deepwalk, 随机游走得到的个原始 id 特征的 embedding
│   ├── desc_ocr_asr_200d, 使用原始特征 description、ocr、asr 预训练的词向量
│   ├── desc_ocr_asr_char_200d, 使用原始特征字级别的 description、ocr、asr 预训练的词向量
│   ├── models, 训练好的模型文件
│   ├── processed_data, 经过预处理的训练测试文件
│   ├── submission, 模型推理得到的预测结果，包括单模型结果和融合结果
│   ├── tfidf_kws, 使用 tfidf 得到的 description、ocr、asr 的关键字
│   ├── uid_aid_svd, userid 和 feedid 的奇异值分解向量
│   ├── uid_bgm_singer_id_svd, userid 和 bgm_singer_id 的奇异值分解向量
│   ├── uid_bgm_song_id_svd, userid 和 bgm_song_id 的奇异值分解向量
│   ├── uid_fid_svd, userid 和 feedid 的奇异值分解向量
│   ├── uid_keyword_list_svd, userid 和 keyword_list 的奇异值分解向量
│   ├── uid_tag_list_svd, userid 和 tag_list 的奇异值分解向量
│   └── wedata
│       └── wechat_algo_data1, 官方比赛数据
├── inference.sh, 推理启动脚本
├── init.sh, 环境初始化脚本
├── requirements.txt, python 库版本
├── src
│   ├── __init__.py
│   ├── common_path.py, 生成数据目录
│   ├── ensemble.py, 融合结果代码
│   ├── lgb
│   │   ├── __init__.py
│   │   ├── lgb_infer.py, lgb 推理代码
│   │   ├── lgb_prepare.py, lgb 数据准备代码
│   │   └── lgb_train.py, lgb 训练文件
│   ├── nn
│   │   ├── __init__.py
│   │   ├── activation.py, 激活函数代码
│   │   ├── args.py, 超参代码
│   │   ├── multideepfm4wx.py, nn 模型代码
│   │   ├── test.py, nn 推理代码
│   │   ├── train.py, nn 训练代码
│   │   └── utils.py, nn 辅助函数代码
│   └── prepare.py, 全局特征代码
└── train.sh, nn 和 lgb 训练启动脚本
```

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
* 特征 (610 个)
  * 原始特征 (5个)
    * userid
    * feedid
    * device
    * authorid
    * videoplayseconds
  * 统计特征 (77 个)
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


### 2. NN

* 参数 1：
  * learning_rate: 0.01
  * batch_size: 512
  * l2_reg_embedding 0.1
  * l2 0.0001
  * epochs 3
  * seed 12
* 参数 2：
  * learning_rate: 0.01
  * batch_size: 512
  * l2_reg_embedding 0.1
  * l2 0.0001
  * epochs 3
  * seed 13
* 参数 3：
  * learning_rate: 0.01
  * batch_size: 512
  * l2_reg_embedding 0.1
  * l2 0.0001
  * epochs 3
  * seed 14
* 参数 4：
  * learning_rate: 0.01
  * batch_size: 512
  * l2_reg_embedding 0.1
  * l2 0.0001
  * epochs 3
  * seed 15
* 参数 5：
  * learning_rate: 0.01
  * batch_size: 512
  * l2_reg_embedding 0.1
  * l2 0.0001
  * epochs 3
  * seed 16
  
* 特征 (768 个)
  * 原始特征 (5个)
    * userid
    * feedid
    * device
    * authorid
    * videoplayseconds
    * keyword_list
    * tag_list
  * SVD 特征 (128 个)
    * userid 与 feedid 的奇异值分解 (64 维)
  * DeepWalk 特征 (128个)
    * userid 与 feedid  (64 维)
  * feed embedding (512 维)

  
## 5. 模型结果

| stage |  model   | weight_uauc | read_comment |   like   | click_avatar | forward  |
| :---: | :------: | :---------: | :----------: | :------: | :----------: | :------: |
| 离线  |   lgb1   |  0.672614   |   0.648805   | 0.645578 |   0.731963   | 0.730256 |
| 离线  |   lgb2   |  0.670604   |   0.647515   | 0.646236 |   0.728114   | 0.721045 |
| 离线  |   nn1    |  0.685342   |   0.663729   | 0.661672 |   0.735029   | 0.743429 |
| 离线  |   nn2    |  0.684505   |   0.662413   | 0.661523 |   0.736941   | 0.736950 |
| 离线  |   nn3    |  0.682537   |   0.658422   | 0.660000 |   0.737261   | 0.737161 |
| 离线  |   nn4    |  0.685410   |   0.659312   | 0.660831 |   0.744362   | 0.745632 |
| 离线  |   nn5    |  0.685820   |   0.660341   | 0.659931 |   0.745821   | 0.745400 |
| 在线  | ensemble |  0.682792   |   0.658657   | 0.655265 |   0.752349   | 0.722799 |

