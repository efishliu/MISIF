import pandas as pd
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from gensim.models import word2vec
# 引入日志配置
#import logging
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from sklearn.model_selection import train_test_split
import argparse
import os


#=============数据处理相关函数===============
#加载基础数据，user-item,user-user
def load_data(dataset):
    #返回值：df_ratings,df_trust
    print("loading data %s......" % dataset)
    ratings_path = "../data/" + dataset + "/user_item.dat"
    trust_path = "../data/" + dataset + "/user_user.dat"
    df_ratings = pd.read_table(ratings_path,header=None)
    df_trust = pd.read_table(trust_path,header=None)
    df_ratings.columns=["userid","itemid","rating"]
    df_trust.columns=["userid","friendid","trust"]
    print("loading finished.")
    return df_ratings,df_trust



#构建物品图，用于随机游走产生embedding训练序列
def build_item_graph(item_graph,user_behavior,negative_user_behavior):
    print("building item graph:")
    for key,value in tqdm(user_behavior.items()):
        for i in range(len(value)):
            if item_graph.get(value[i]) is None:
                item_graph[value[i]] = []
            item_graph[value[i]].append(value[i])
    if negative_user_behavior is not None:        
        for key,value in tqdm(negative_user_behavior.items()):
            for i in range(len(value)):
                if item_graph.get(value[i]) is None:
                    item_graph[value[i]] = []
                item_graph[value[i]].append(value[i])
    print("item graph build finished.")

#随机游走下一节点
def next_node(graph,node1):
    neighbour = graph.get(node1)
    if neighbour is None:
        return node1
    else:
        node2 = neighbour[np.random.randint(0,len(neighbour))]
        return node2

#随机游走产生embedding训练samples
def random_walk(graph,random_walk_number,random_walk_length):
    print("random walk to get embedding sample:")
    index = 0
    sample_list = []
    for key,value in tqdm(graph.items()):
        for i in range(random_walk_number):#每个节点进行20次随机游走
            sample = []
            neighbour_node = key
            sample.append(neighbour_node)
            for j in range(random_walk_length):#步长为20
                neighbour_node = next_node(graph,neighbour_node)
                sample.append(neighbour_node)
            sample_list.append(sample)
            index = index + 1
    print("finished %d samples random walk." % index)
    return sample_list

#word2vec embedding
def w2v_embedding(sample_item,item_embedding,df_ratings,embedding_size=64,emb_model_path=""):
    print("item embedding training:")
    model = word2vec.Word2Vec(sample_item,sg=1,size=embedding_size,min_count=1)
    model.save(emb_model_path)
    print("embedding model saved in %s" % emb_model_path)
    return model

#获取物品embedding
def get_item_embedding(df_ratings,item_embedding,model):
    print("getting item embedding:")
    for row in tqdm(df_ratings.itertuples()):
        item_id = str(getattr(row, 'itemid'))
        if item_embedding.get(item_id) is None:
            item_embedding[item_id] = model[item_id]  

#获取用户行为
def get_user_behavior(df_ratings,user_behavior,negative_user_behavior):
    print("getting user behavior:")
    for row in tqdm(df_ratings.itertuples()):
        userid = str(getattr(row, 'userid'))
        itemid = str(getattr(row, 'itemid'))
        rating = getattr(row, 'rating')
        if negative_user_behavior is not None:
            if int(rating) < 3:
                if negative_user_behavior.get(userid) is None:
                    negative_user_behavior[userid] = []
                negative_user_behavior[userid].append(itemid)
            else:
                if user_behavior.get(userid) is None:
                    user_behavior[userid] = []
                user_behavior[userid].append(itemid)
        else:
            if user_behavior.get(userid) is None:
                    user_behavior[userid] = []
            user_behavior[userid].append(itemid)

#将用户行为转换为embedding表示
def user_behavior_to_user_embedding(user_behavior,negative_user_behavior,user_embedding,item_embedding):
    print("user behavior convert to user embedding:")
    for user_id,user_sequence in tqdm(user_behavior.items()):
        user_embedding[user_id] = []
        for item_id in user_sequence:
            item_emb = item_embedding[item_id]
            user_embedding[user_id].append(item_emb)
    #加入小于评分3的物品
    if negative_user_behavior is not None: 
        for user_id,user_sequence in tqdm(negative_user_behavior.items()):
            if user_embedding.get(user_id) is None:
                user_embedding[user_id] = []
            for item_id in user_sequence:
                item_emb = item_embedding[item_id]
                user_embedding[user_id].append(item_emb)

#用户对物品的评分(字典形式)
def user_item_dict(df_ratings,user_item_rating):
    for row in df_ratings.itertuples():
        userid = str(getattr(row, 'userid'))
        itemid = str(getattr(row, 'itemid'))
        rating = getattr(row, 'rating')
        if user_item_rating.get(userid) is None:
            user_item_rating[userid] = {}
        user_item_rating[userid].update({itemid:rating})

#字典按值降序排序
def dict_ranking(S,Descending_S):
    for k,v in S.items():
        d = sorted(v.items(),key=lambda x:x[1],reverse=True)
        Descending_S[k] = {}
        for m,n in d:
            Descending_S[k].update({m:n})

# load feature 
def load_feature(dataset,feature,f_header):
    print("loading feature %s/%s......" % (dataset,feature))
    feature_path = "../data/" + dataset + "/" + feature
    df_feature = pd.read_table(feature_path,header=None)
    df_feature.columns = f_header
    print("loading %s/%s finished." % (dataset,feature))
    return df_feature

#获取特征的onehot表示
def get_feature_onehot(df_feature):
    feature_onehot = {}
    col_name = df_feature.columns.values.tolist()
    col1_name = col_name[0]
    col2_name = col_name[1]
    df_feature_onehot = pd.get_dummies(df_feature[col2_name])
    df_feature_onehot_dim = np.shape(df_feature_onehot.loc[0].values)[0]
    for index,row in df_feature.iterrows():
        id = str(row[col1_name])
        if df_feature.get(id) is None:
            feature_onehot[id] = df_feature_onehot.loc[index].values
        else:
            feature_onehot[id] += df_feature_onehot.loc[index].values
    return feature_onehot,df_feature_onehot_dim

#获取所有属性特征表示
def get_feature(dataset):
    if dataset == 'yelp':
        #特征总维数
        feature_dim = 0

        item_feature_onehot  = []
        user_feature_onehot = []
        #加入物品种类属性
        df_item_category = load_feature('yelp','item_category.dat',['item_id',"category"])
        item_category_onehot,item_category_onehot_dim = get_feature_onehot(df_item_category)
        item_feature_onehot.append(item_category_onehot)
        feature_dim += item_category_onehot_dim
        
        #加入物品城市属性
        df_item_city = load_feature('yelp','item_city.dat',['item_id',"city"])
        item_city_onehot,item_city_onehot_dim = get_feature_onehot(df_item_city)
        item_feature_onehot.append(item_city_onehot)
        feature_dim += item_city_onehot_dim
        
        #加入用户complish属性
        df_user_compliment = load_feature('yelp','user_compliment.dat',['user_id',"compliment"])
        user_compliment_onehot,user_compliment_onehot_dim = get_feature_onehot(df_user_compliment)
        user_feature_onehot.append(user_compliment_onehot)
        feature_dim += user_compliment_onehot_dim
        
        return item_feature_onehot,user_feature_onehot,feature_dim

#===========深度学习模型的数据处理===========
#划分数据集：训练集和测试集
def split_dataset(df_ratings,test_size=0.2):
    df_train_rating,df_test_rating = train_test_split(df_ratings,test_size=0.2,random_state=42)
    return df_train_rating,df_test_rating

#训练集或测试集的数据转换
def get_train_set(df,user_multi_interest,item_embedding,item_feature_onehot=None,user_feature_onehot=None,max_len=6):
    #print("getting deep model train set:")
    X_u = []
    X_i = []
    X_f = []
    y = []
    for row in tqdm(df.itertuples()):
        userid = str(getattr(row, 'userid'))
        itemid = str(getattr(row, 'itemid'))
        rating = getattr(row, 'rating')
        ue = user_multi_interest[userid]
        #用户多兴趣padding
        ue = keras.preprocessing.sequence.pad_sequences([ue],maxlen=max_len,value=0,padding='post',dtype='float32')
        ue = tf.convert_to_tensor(ue[0],dtype=tf.float32)
        ie = tf.convert_to_tensor(item_embedding[itemid],dtype=tf.float32)

        fe = np.array([])
        #加入item其他特征向量
        item_feature_num = len(item_feature_onehot)
        for i in range(item_feature_num):
            feature_i_shape = np.shape(list(item_feature_onehot[i].values())[0])
            try:
                fe_i = item_feature_onehot[i][itemid]
            except:
                fe_i = np.zeros(feature_i_shape)
            fe = np.append(fe,fe_i)
        #加入user其他特征向量
        user_feature_num = len(user_feature_onehot)
        for i in range(user_feature_num):
            feature_u_shape = np.shape(list(user_feature_onehot[i].values())[0])
            try:
                fe_u = user_feature_onehot[i][userid]
            except:
                fe_u = np.zeros(feature_u_shape)
            fe = np.append(fe,fe_u)
        fe = tf.convert_to_tensor(fe,dtype=tf.float32)
        X_u.append(ue)
        X_i.append(ie)
        X_f.append(fe)
        y.append(rating)
    '''
    print("train data list to np.array......")
    X_u = np.array(X_u)
    X_i = np.array(X_i)
    X_f = np.array(X_f)
    y = np.array(y)
    '''
    #print("train data list to tensor......")
    X_u = tf.convert_to_tensor(X_u)
    X_i = tf.convert_to_tensor(X_i)
    X_f = tf.convert_to_tensor(X_f)
    y = tf.convert_to_tensor(y)
    y = tf.cast(y,dtype=tf.float32)
    #print('finished getting train set.')
    
    return X_u,X_i,X_f,y



def get_dict_set(test_dict,user_multi_interest,item_embedding,item_feature_onehot=None,user_feature_onehot=None,max_len=6):
    X_u = []
    X_i = []
    X_f = []
    for userid,s in tqdm(test_dict.items()):
        for itemid,r in s.items():
            ue = user_multi_interest[userid]
            #用户多兴趣padding
            ue = keras.preprocessing.sequence.pad_sequences([ue],maxlen=max_len,value=0,padding='post',dtype='float32')
            ue = tf.convert_to_tensor(ue[0],dtype=tf.float32)
            ie = tf.convert_to_tensor(item_embedding[itemid],dtype=tf.float32)

            fe = np.array([])
            #加入item其他特征向量
            item_feature_num = len(item_feature_onehot)
            for i in range(item_feature_num):
                feature_i_shape = np.shape(list(item_feature_onehot[i].values())[0])
                try:
                    fe_i = item_feature_onehot[i][itemid]
                except:
                    fe_i = np.zeros(feature_i_shape)
                fe = np.append(fe,fe_i)
            #加入user其他特征向量
            user_feature_num = len(user_feature_onehot)
            for i in range(user_feature_num):
                feature_u_shape = np.shape(list(user_feature_onehot[i].values())[0])
                try:
                    fe_u = user_feature_onehot[i][userid]
                except:
                    fe_u = np.zeros(feature_u_shape)
                fe = np.append(fe,fe_u)
            fe = tf.convert_to_tensor(fe,dtype=tf.float32)
            X_u.append(ue)
            X_i.append(ie)
            X_f.append(fe)
    X_u = tf.convert_to_tensor(X_u)
    X_i = tf.convert_to_tensor(X_i)
    X_f = tf.convert_to_tensor(X_f)
    return X_u,X_i,X_f
'''
def get_train_user_embedding(user_multi_interest,item_embedding,userid,itemid,max_len=4):
    ue = user_multi_interest[userid]
    #用户多兴趣padding
    ue = keras.preprocessing.sequence.pad_sequences([ue],maxlen=max_len,value=0,padding='post',dtype='float32')
    ue = tf.convert_to_tensor(ue[0],dtype=tf.float32)
    ie = tf.convert_to_tensor(item_embedding[itemid],dtype=tf.float32)
    fe = np.array([])
    #加入item其他特征向量
    item_feature_num = len(item_feature_onehot)
    for i in range(item_feature_num):
        feature_i_shape = np.shape(list(item_feature_onehot[i].values())[0])
        try:
            fe_i = item_feature_onehot[i][itemid]
        except:
            fe_i = np.zeros(feature_i_shape)
        fe = np.append(fe,fe_i)
    #加入user其他特征向量
    user_feature_num = len(user_feature_onehot)
    for i in range(user_feature_num):
        feature_u_shape = np.shape(list(user_feature_onehot[i].values())[0])
        try:
            fe_u = user_feature_onehot[i][userid]
        except:
            fe_u = np.zeros(feature_u_shape)
        fe = np.append(fe,fe_u)
    fe = tf.convert_to_tensor(fe,dtype=tf.float32)
    return ue,ie,fe

'''
