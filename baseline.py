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
del df_affinity_train['Ki']
#提取蛋白质序列基本数量特征
def fun_D_N_rate(x):
    m=x.count('D')+x.count('N')
    a=float(m/len(x))
    return a
def fun_ALL_rate(x):
    m = x.count('ALL')
    a = float(m / len(x))
    return a
def fun_water_rate(x):
    m = x.count('V')+x.count('I')+x.count('L')+x.count('F')+x.count('W')+x.count('Y')+x.count('M')
    a = float(m / len(x))
    return a
def fun_f_w_y_h_rate(x):
    m = x.count('H')+ x.count('F') + x.count('W') + x.count('Y')
    a = float(m / len(x))
    return a
def fun_v_i_l_m_rate(x):
    m = x.count('V')+ x.count('I') + x.count('L') + x.count('M')
    a = float(m / len(x))
    return a
def fun_lovewater_rate(x):
    m = x.count('S')+x.count('T')+x.count('H')+x.count('N')+x.count('Q')+x.count('E')+x.count('D')+x.count('K')+x.count('R')
    a = float(m / len(x))
    return a
def fun_krh_rate(x):
    m = x.count('K')+ x.count('R') + x.count('H')
    a = float(m / len(x))
    return a
def fun_de_rate(x):
    m = x.count('D')+ x.count('E')
    a = float(m / len(x))
    return a
def fun_PGAS_rate(x):
    m = x.count('P')+ x.count('G') + x.count('A') + x.count('S')
    a = float(m / len(x))
    return a
def fun_zhengfulizi_rate(x):
    m2 = x.count('K') + x.count('R') + x.count('H')
    m1 = x.count('D') + x.count('E')
    try:
        a = float(m2 / m1)
        return a
    except:
        return 0
def water_likeorhate_rate(x):
    m1 = x.count('S') + x.count('T') + x.count('H') + x.count('N') + x.count('Q') + x.count('E') + x.count(
        'D') + x.count('K') + x.count('R')
    m2 = x.count('V') + x.count('I') + x.count('L') + x.count('F') + x.count('W') + x.count('Y') + x.count('M')
    try:
        a=float(m1/m2)
        return a
    except:
        return 0
def protein_sequence_num_feat(df):
    df['A_num']=df.Sequence.map(lambda x:x.count('A'))
    df['R_num']=df.Sequence.map(lambda x:x.count('R'))
    df['N_num'] = df.Sequence.map(lambda x: x.count('N'))
    df['D_num'] = df.Sequence.map(lambda x: x.count('D'))
    df['C_num'] = df.Sequence.map(lambda x: x.count('C'))
    df['E_num'] = df.Sequence.map(lambda x: x.count('E'))
    df['Q_num'] = df.Sequence.map(lambda x: x.count('Q'))
    df['G_num'] = df.Sequence.map(lambda x: x.count('G'))
    df['H_num'] = df.Sequence.map(lambda x: x.count('H'))
    df['I_num'] = df.Sequence.map(lambda x: x.count('I'))
    df['L_num'] = df.Sequence.map(lambda x: x.count('L'))
    df['K_num'] = df.Sequence.map(lambda x: x.count('K'))
    df['M_num'] = df.Sequence.map(lambda x: x.count('M'))
    df['F_num'] = df.Sequence.map(lambda x: x.count('F'))
    df['P_num'] = df.Sequence.map(lambda x: x.count('P'))
    df['S_num'] = df.Sequence.map(lambda x: x.count('S'))
    df['T_num'] = df.Sequence.map(lambda x: x.count('T'))
    df['W_num'] = df.Sequence.map(lambda x: x.count('W'))
    df['Y_num'] = df.Sequence.map(lambda x: x.count('Y'))
    df['V_num'] = df.Sequence.map(lambda x: x.count('V'))
    df['len_se']=df.Sequence.map(lambda x: len(x))
    #df['D_N_rate']=df.Sequence.map(lambda x: fun_D_N_rate(x))
    #df['all_rate'] = df.Sequence.map(lambda x: fun_ALL_rate(x))
    #df['water_rate']=df.Sequence.map(lambda x: fun_water_rate(x))
    #df['f_w_y_h_rate']=df.Sequence.map(lambda x: fun_f_w_y_h_rate(x))
    #df['v_i_l_m_rate']=df.Sequence.map(lambda x: fun_v_i_l_m_rate(x))
    #df['love_water']=df.Sequence.map(lambda x: fun_lovewater_rate(x))
    #df['k_r_h_rate']=df.Sequence.map(lambda x: fun_krh_rate(x))
    #df['d_e_rate']=df.Sequence.map(lambda x: fun_de_rate(x))
    #df['pgas_rate']=df.Sequence.map(lambda x: fun_PGAS_rate(x))
    #df['lizi_rate']=df.Sequence.map(lambda x: fun_zhengfulizi_rate(x))
    #df['water_likeorhate_rate']=df.Sequence.map(lambda x: water_likeorhate_rate(x))
    return df
#构建训练和测试样本的蛋白质特征
df_protein_train_feat=protein_sequence_num_feat(df_protein_train_data)
#del df_protein_train_feat['Sequence']

df_protein_test_feat=protein_sequence_num_feat(df_protein_test_data)
#del df_protein_test_feat['Sequence']

i=0
while(i<len(df_molecule['Fingerprint'][1])):
    df_molecule['Fingerprint'+'_'+str(i)]=df_molecule.Fingerprint.map(lambda x:int(x[i]))
    i=i+3

df_protein_train_feat=pd.merge(df_affinity_train,df_protein_train_feat,on='Protein_ID',how='left')
del df_molecule['Fingerprint']

df_protein_train_feat=pd.merge(df_protein_train_feat,df_molecule,on='Molecule_ID',how='left')
df_protein_train_feat=df_protein_train_feat.fillna(df_protein_train_feat.mean())

df_protein_test_feat=pd.merge(df_affinity_test_toBePredicted,df_protein_test_feat,on='Protein_ID',how='left')
df_protein_test_feat=pd.merge(df_protein_test_feat,df_molecule,on='Molecule_ID',how='left')
df_protein_test_feat=df_protein_test_feat.fillna(df_protein_test_feat.mean())

#test_vector
n = 128
texts = [[word for word in re.findall(r'.{3}',document)]
               for document in list(df_protein_test_data['Sequence'])]
model = Word2Vec(texts,size=n,window=4,min_count=1,negative=3,
                 sg=1,sample=0.001,hs=1,workers=4)
vectors = pd.DataFrame([model[word] for word in (model.wv.vocab)])

vectors['Word'] = list(model.wv.vocab)

vectors.columns= ["vec_{0}".format(i) for i in range(0,n)]+["Word"]
#train_vector
texts2 = [[word for word in re.findall(r'.{3}',document)]
               for document in list(df_protein_train_data['Sequence'])]
model2 = Word2Vec(texts,size=n,window=4,min_count=1,negative=3,
                 sg=1,sample=0.001,hs=1,workers=4)
vectors2 = pd.DataFrame([model[word] for word in (model.wv.vocab)])
vectors2['Word'] = list(model.wv.vocab)
vectors2.columns= ["vec_{0}".format(i) for i in range(0,n)]+["Word"]
vectors['Protein_ID']=df_protein_test_data['Protein_ID']
vectors2['Protein_ID']=df_protein_train_data['Protein_ID']
del vectors2['Word']
del vectors['Word']
print('vector_get_finish')
df_protein_train_feat=pd.merge(df_protein_train_feat,vectors2,on='Protein_ID',how='left')
df_protein_test_feat=pd.merge(df_protein_test_feat,vectors,on='Protein_ID',how='left')
del df_protein_test_feat['Protein_ID']
del df_protein_test_feat['Molecule_ID']
del df_protein_train_feat['Protein_ID']
del df_protein_train_feat['Molecule_ID']
del df_protein_train_feat['Sequence']
del df_protein_test_feat['Sequence']
print('feature_finish')
#----------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(df_protein_train_feat,train_label,test_size=0.2,random_state=100)

import xgboost as xgb

print ('start running ....')
dtrain = xgb.DMatrix(x_train,label=y_train)
dval = xgb.DMatrix(x_val,label=y_val)
param = {'learning_rate' : 0.1,
        'n_estimators': 1000,
        'max_depth': 4,
        'min_child_weight': 6,
        'gamma': 0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eta': 0.05,
        'silent': 1,
         }

num_round =150
plst = list(param.items())
plst += [('eval_metric', 'rmse')]
evallist = [(dval, 'eval'), (dtrain, 'train')]
bst=xgb.train(plst,dtrain,num_round,evallist,early_stopping_rounds=10)
dtest = xgb.DMatrix(df_protein_test_feat)
Pred = bst.predict(dtest)

df_affinity_test_toBePredicted['Ki']=Pred
df_affinity_test_toBePredicted.to_csv('baseline_4_12.csv',index=False)