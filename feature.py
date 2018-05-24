import pandas as pd
import numpy as np
import gc
import re
import sys
import time
import jieba
import os.path
import os
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec
df_molecule=pd.read_csv('df_molecule.csv')
df_protein_test_data=pd.read_csv('df_protein_test.csv')

df_affinity_test_toBePredicted=pd.read_csv('df_affinity_test_toBePredicted.csv')

df_protein_train_data=pd.read_csv('df_protein_train.csv')

df_affinity_train=pd.read_csv('df_affinity_train.csv')
#用于保存训练集标签
train_label=df_affinity_train['Ki']
#train_label=float(train_label)

#提取蛋白质序列基本数量特征
n = 128
texts = [[word for word in re.findall(r'.{3}',document)]
               for document in list(df_protein_test_data['Sequence'])]
model = Word2Vec(texts,size=n,window=4,min_count=1,negative=3,
                 sg=1,sample=0.001,hs=1,workers=4)
vectors = pd.DataFrame([model[word] for word in (model.wv.vocab)])

vectors['Word'] = list(model.wv.vocab)

vectors.columns= ["vec_{0}".format(i) for i in range(0,n)]+["Word"]


texts2 = [[word for word in re.findall(r'.{3}',document)]
               for document in list(df_protein_train_data['Sequence'])]
model2 = Word2Vec(texts,size=n,window=4,min_count=1,negative=3,
                 sg=1,sample=0.001,hs=1,workers=4)
vectors2 = pd.DataFrame([model[word] for word in (model.wv.vocab)])

vectors2['Word'] = list(model.wv.vocab)

vectors2.columns= ["vec_{0}".format(i) for i in range(0,n)]+["Word"]

train = lgb.Dataset(train_feat, label=label_x)

test  = lgb.Dataset(testt_feat, label=label_y,reference=train)



params = {
    'boosting_type': 'gbdt',
    'objective': 'regression_l2',
    'metric': 'l2',
    #'objective': 'multiclass',
    #'metric': 'multi_error',
    #'num_class':5,
    'min_child_weight': 3,
    'num_leaves': 2 ** 5,
    'lambda_l2': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'learning_rate': 0.05,
    'tree_method': 'exact',
    'seed': 2017,
    'nthread': 12,
    'silent': True
    }
num_round = 3000
gbm = lgb.train(params,
                  train,
                  num_round,
                  verbose_eval=50,
                  valid_sets=[train,test]
                  )
preds_sub = gbm.predict(testt_feat)

