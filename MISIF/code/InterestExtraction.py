import numpy as np
from scipy.special import softmax
from tqdm import tqdm
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import os
from DataProcessing import dict_ranking
#=================用户多兴趣提取相关函数=======================
#kmeans方法提取用户多兴趣
def my_kmeans(user_embedding,KI):
    kmeans = KMeans(n_clusters=KI,max_iter=50).fit(user_embedding)
    index = 0
    u = {}
    for i in kmeans.labels_:
        if u.get(i) is None:
            u[i] = []
        u[i].append(user_embedding[index])
        index += 1
    for c,e in u.items():
        me = np.mean(e,axis=0)
        u[c] = me
    ue = np.array(list(u.values()))
    return ue

#================动态路由方法提取用户多兴趣====================
def squash(v):
    v2 = np.linalg.norm(v)
    v22 = v2 * v2
    u = v2/(1+v22) * v
    return u

def dynamic_routing(user_embedding,KI,r,DR_trans_flag=False):
    user_embedding_num = len(user_embedding)
    d = len(user_embedding[0])
    b = np.zeros((user_embedding_num,KI))#初始化bij=0      
    if DR_trans_flag == 1:
        w = np.random.normal(size=((user_embedding_num,KI,d,d))) #初始化wij为正态分布数组,每个胶囊i与胶囊j的转换矩阵不同，保证初始兴趣不同
    elif DR_trans_flag == 2:
        w = np.random.normal(size=((KI,d,d))) #初始化wj为正态分布数组,每个胶囊j的转换矩阵不同
    else:
        w = np.random.normal(size=(d,d)) #初始化wij为正态分布,共享转换矩阵


    user_multi_embedding = np.zeros((KI,d))
    for l in range(r):
        c = softmax(b,axis=1)
        for j in range(KI):
            t = np.zeros((d,))
            for i in range(user_embedding_num):
                if DR_trans_flag == 1:
                    t = t + c[i][j] * np.dot(w[i][j],user_embedding[i])
                elif DR_trans_flag == 2:
                    t = t + c[i][j] * np.dot(w[j],user_embedding[i])
                else:
                    t = t + c[i][j] * np.dot(w,user_embedding[i])#使用共享转换矩阵
            user_multi_embedding[j] = t
            user_multi_embedding[j] = squash(user_multi_embedding[j])
        for i in range(user_embedding_num):
            for j in range(KI):
                if DR_trans_flag == 1:
                    b[i][j] = b[i][j] + np.dot(np.dot(user_multi_embedding[j].T,w[i][j]),user_embedding[i])
                elif DR_trans_flag == 2:
                    b[i][j] = b[i][j] + np.dot(np.dot(user_multi_embedding[j].T,w[j]),user_embedding[i])
                else:
                    b[i][j] = b[i][j] + np.dot(np.dot(user_multi_embedding[j].T,w),user_embedding[i])#使用共享转换矩阵
    return user_multi_embedding
'''
def dynamic_routing(user_embedding,KI,r):
    user_embedding_num = len(user_embedding)
    b = np.zeros((user_embedding_num,KI))#初始化bij=0
    w = np.zeros((user_embedding_num,KI))
    user_multi_embedding = []
    d = len(user_embedding[0])
    for i in range(KI):
        e = np.random.normal(size=(d,))
        user_multi_embedding.append(e)
    #print("初始化多兴趣向量:",user_multi_embedding[0])
    for l in range(r):#迭代次数
        #print("第%d次迭代......" % (l+1))
        for i in range(user_embedding_num):
            for j in range(KI):
                b[i][j] = np.dot(user_embedding[i],user_multi_embedding[j].T)
        w = softmax(b)
        for j in range(KI):
            user_multi_embedding[j] = list(np.zeros((d,)))
            for i in range(user_embedding_num):
                user_multi_embedding[j] = user_multi_embedding[j] + b[i][j] * user_embedding[i]
            user_multi_embedding[j] = squash(user_multi_embedding[j])
    #print("多兴趣向量:",user_multi_embedding[0])
    return np.array(user_multi_embedding)
'''
# additive self-Attention方法
#获取训练集
def get_SA_train_set(user_item_rating,item_embedding,max_len,emb_size):
    print("getting self-attention train set:")
    X_train_u = []
    X_train_i = []
    y_train = []
    for k,v in tqdm(user_item_rating.items()):
        index = 0
        ue = []
        for i,r in v.items():
            ue.append(item_embedding[i])
            index += 1
            if index == max_len:
                break
        if index < max_len:
            ue = keras.preprocessing.sequence.pad_sequences([ue],maxlen=max_len,value=0,padding='post',dtype='float32')
        ue = np.reshape(ue,(max_len,emb_size))
        
        index = 0
        for i,r in v.items():
            X_train_u.append(tf.convert_to_tensor(ue,dtype=tf.float32))
            X_train_i.append(tf.convert_to_tensor(item_embedding[i],dtype=tf.float32))
            y_train.append(r)
            if index == max_len:
                break
            
    X_train_u = tf.convert_to_tensor(X_train_u)
    X_train_i = tf.convert_to_tensor(X_train_i)
    y_train = np.array(y_train)
    print('finished getting self-attention train set.')
    return X_train_u,X_train_i,y_train

class AdditiveAttentionLayer(keras.layers.Layer):
    def __init__(self,da,K,d,**kwargs):
        super().__init__(**kwargs)
        self.da = da
        self.K = K
        self.d = d
        
    def build(self,input_shape):
        self.W1 = self.add_weight(name='W1',shape=((self.d,self.da)),
                                  initializer=keras.initializers.get('glorot_uniform'),
                                 trainable=True)
        self.W2 = self.add_weight(name='W2',shape=((self.K,self.da)),
                                  initializer=keras.initializers.get('glorot_uniform'),
                                 trainable=True)
        super().build(input_shape)
        
    def call(self,x):
        t = K.dot(x,self.W1)
        t = K.tanh(t)
        a = K.dot(t,tf.transpose(self.W2))
        a = K.softmax(a,axis=2)
        a = K.permute_dimensions(a,[0,2,1])
        Vu = K.batch_dot(a,x,axes=[2,1])
        Vu = K.reshape(Vu,(-1,self.K * self.d))

        return Vu



def additive_attention_model_train(user_item_rating,item_embedding,KI,sa_model_path,sa_padding_num,emb_size):
    #将用户-物品-评分按评分降序排列
    dec_uir = {}
    dict_ranking(user_item_rating,dec_uir)
    SA_X_u,SA_X_i,SA_y = get_SA_train_set(user_item_rating,item_embedding,sa_padding_num,emb_size)

    input_u = keras.layers.Input(shape=((sa_padding_num,emb_size)),name="user_embedding_input")
    input_i = keras.layers.Input(shape=((emb_size,)),name="item_embedding_input")
    additive_attention = AdditiveAttentionLayer(da=64,K=KI,d=emb_size,name='additive_attention_layer')(input_u)
    concat = keras.layers.concatenate([additive_attention,input_i])
    output = keras.layers.Dense(1,name="output")(concat)
    additive_attention_model = keras.Model(inputs=[input_u,input_i],outputs=output)

    additive_attention_model.summary()
    additive_attention_model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=[keras.metrics.MeanAbsoluteError()])
    print("_____________additive attention model start training_____________")
    additive_attention_history = additive_attention_model.fit([SA_X_u,SA_X_i],SA_y,batch_size=128,epochs=50)

    additive_attention_model.save(sa_model_path)

    W1,W2 = additive_attention_model.get_layer('additive_attention_layer').get_weights()

    return W1,W2

def additive_attention(e,W1,W2):
    W1 = np.array(W1)
    W2 = np.array(W2)
    e = np.array(e)
    A = np.matmul(np.tanh(np.matmul(e,W1)),W2.T)
    A = softmax(A,axis=1)
    Vu = np.matmul(A.T,e)
    return Vu
# multi head dot attention方法
class DotAttentionLayer(keras.layers.Layer):
    def __init__(self,num_head,pad_num,output_dim,kernel_initializer='glorot_uniform',**kwargs):
        self.num_head = num_head
        self.output_dim = output_dim
        self.pad_num = pad_num
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        super(DotAttentionLayer,self).__init__(**kwargs)
        
    def build(self,input_shape):
        self.WQ=self.add_weight(name='WQ',
           shape=(self.num_head,input_shape[2],self.output_dim),#input_shape[2]=emb_size
           initializer=self.kernel_initializer,
           trainable=True)
        
        self.WK=self.add_weight(name='WK',
           shape=(self.num_head,input_shape[2],self.output_dim),
           initializer=self.kernel_initializer,
           trainable=True)
        
        self.WV=self.add_weight(name='WV',
           shape=(self.num_head,input_shape[2],self.output_dim),
           initializer=self.kernel_initializer,
           trainable=True)
        
        self.built = True

        
    def call(self,x):
        q=K.dot(x,self.WQ[0])

        k=K.dot(x,self.WK[0])
        v=K.dot(x,self.WV[0])
        e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))
        e=e/(self.output_dim**0.5)
        e=K.softmax(e)
        outputs=K.batch_dot(e,v)
        outputs = K.reshape(outputs,(-1,self.pad_num * self.output_dim))
        
        for i in range(1,self.num_head):
            q=K.dot(x,self.WQ[i])
            k=K.dot(x,self.WK[i])
            v=K.dot(x,self.WV[i])
            e=K.batch_dot(q,K.permute_dimensions(k,[0,2,1]))
            e=e/(self.output_dim**0.5)
            e=K.softmax(e)
            o=K.batch_dot(e,v)
            o = K.reshape(o,(-1,self.pad_num * self.output_dim))
            outputs=K.concatenate([outputs,o])
        return outputs
        
        
    def compute_output_shape(self,input_shape):
        return (input_shape[0],self.num_head * input_shape[1] * self.output_dim)

def dot_attention_model_train(user_item_rating,item_embedding,KI,sa_model_path,sa_padding_num,emb_size):
    #将用户-物品-评分按评分降序排列
    dec_uir = {}
    dict_ranking(user_item_rating,dec_uir)
    SA_X_u,SA_X_i,SA_y = get_SA_train_set(user_item_rating,item_embedding,sa_padding_num,emb_size)

    input_u = keras.layers.Input(shape=((sa_padding_num,emb_size)),name="user_embedding_input")#
    input_i = keras.layers.Input(shape=((emb_size,)),name="item_embedding_input")#64,
    dot_attention = DotAttentionLayer(KI,sa_padding_num,emb_size,name="dot_attention_layer")(input_u)#
    concat = keras.layers.concatenate([dot_attention,input_i])
    output = keras.layers.Dense(1,name="output")(concat)
    dot_attention_model = keras.Model(inputs=[input_u,input_i],outputs=output)

    dot_attention_model.summary()
    dot_attention_model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=[keras.metrics.MeanAbsoluteError()])
    print("_____________multi head dot attention model start training_____________")
    dot_attention__history = dot_attention_model.fit([SA_X_u,SA_X_i],SA_y,batch_size=128,epochs=50)

    dot_attention_model.save(sa_model_path)

    WQ,WK,WV= dot_attention_model.get_layer('dot_attention_layer').get_weights()

    return WQ,WK,WV

def dot_attention(e,WQ,WK,WV):
    e = np.array(e)
    WQ = np.array(WQ)
    WK = np.array(WK)
    WV = np.array(WV)
    num_head,emb_size,output_dim = np.shape(WQ)
    e_len = len(e)
    Vu = []
    for i in range(num_head):
        Q = np.matmul(e,WQ[i]) #n*out_dim
        K = np.matmul(e,WK[i]) #n*out_dim
        V = np.matmul(e,WV[i]) #n*out_dim
        t = softmax(np.matmul(Q,K.T)/(output_dim**0.5),axis=1)#n*n
        o = np.matmul(t,V) #n*out_dim
        o = np.max(o,axis=0)
        Vu.append(o)
    Vu = np.array(Vu)
    return Vu



#用户多兴趣提取
def user_multi_embedding_extract(user_embedding,user_multi_embedding,KI,method="DR",flag=False,
                                item_embedding=None,user_item_rating=None,dataset=None,
                                sa_padding_num=None,emb_size=None):
    
    # addtive-attention model train
    if method == 'SA_AA':
        sa_model_path = '../data/' + dataset + "/SA_AA_" + str(KI) + "_" + str(emb_size) + ".model"
        if os.path.exists(sa_model_path):
            print("W1,W2 train finished,loading additive attention model......")
            additive_attention_model = keras.models.load_model(sa_model_path)
            W1,W2 = additive_attention_model.get_layer('additive_attention_layer').get_weights()
        else:
            print("train additive attention model:")
            W1,W2 = additive_attention_model_train(user_item_rating,item_embedding,KI,sa_model_path,sa_padding_num,emb_size)
    # multi head dot attention
    if method == 'SA_DA':
        sa_model_path = '../data/' + dataset + "/SA_DA_" + str(KI) + "_" + str(emb_size) + ".model"
        if os.path.exists(sa_model_path):
            print("WQ,WK,WV train finished,loading multi head dot attention model......")
            dot_attention_model = keras.models.load_model(sa_model_path)
            WQ,WK,WV = dot_attention_model.get_layer('dot_attention_layer').get_weights()
        else:
            print("train multi head dot attention model:")
            WQ,WK,WV = dot_attention_model_train(user_item_rating,item_embedding,KI,sa_model_path,sa_padding_num,emb_size)
        
    print("extracting user multi embedding, use %s method:" % method)
    for u,e in tqdm(user_embedding.items()):
        K_ = max(1,min(KI,int(np.log(len(e)))))  
        if method == 'DR':
            u_multi_e = dynamic_routing(e,K_,10,flag)
        elif method == 'KM':
            u_multi_e = my_kmeans(e,K_)
        elif method == 'SA_AA':
            u_multi_e = additive_attention(e,W1,W2)
        elif method == 'SA_DA':
            u_multi_e = dot_attention(e,WQ,WK,WV)
        else:
            print("method not exist!")
            exit(0)
        user_multi_embedding[u] = u_multi_e

# multi-head self-attention 
