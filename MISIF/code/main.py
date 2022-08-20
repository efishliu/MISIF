import numpy as np
from gensim.models import word2vec
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import argparse
import os
from sklearn.model_selection import KFold
import time
import pynvml

import DataProcessing as DP
import InterestExtraction as IE
import SocialNetwork as SN
import InterestSpread as IS
import Evaluation as Eva

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, default='yelp', help='dataset:douban_movie|douban_book|yelp 数据集.default:yelp')
parser.add_argument('-rating_del_flag', type=str, default='False', help='Whether to handle items with a score less than 3:True|False 是否处理评分小于3的物品.default:False')
parser.add_argument('-random_walk_number', type=int, default=20, help='a node random walk number:10|20|30 节点随机游走次数.default:20')
parser.add_argument('-random_walk_length', type=int, default=20, help='a random walk length:10|20|30 随机游走步长.default:20')
parser.add_argument('-emb_size', type=int, default=64, help='embedding size:8|16|32|args.emb_size|128.default:64')
parser.add_argument('-alpha', type=float, default=0.5, help='alpha:pcc/common_friends pcc与common_friends权重.default:0.5')
parser.add_argument('-beta', type=float, default=0.8, help='beta:direct_trust/indirect_trust 直接/间接信任值权重.default:0.8')
parser.add_argument('-KI', type=int, default=4, help='最大兴趣嵌入数目:2|4|6|8.default:4')
parser.add_argument('-MI_method', type=str, default='DR', help='多兴趣提取方法:KM,DR,SA_AA,SA_DA.default:DR')
parser.add_argument('-DR_trans_flag', type=int, default=2, help='转换矩阵类型:0(共享),1(wij),2(wj).default:2')
parser.add_argument('-KF', type=int, default=-1, help='multi interest communication friends number:0|2|4|6|8|10|20|-1启发式 选取KF个朋友进行传播.default:-1')
parser.add_argument('-input_method', type=str, default='attention', help='输入层的拼接方法：attention|mean|max.default:attention')
parser.add_argument('-batch_size', type=int, default=256, help='batch_size:1|8|16|32|64|128.default:256')
parser.add_argument('-loss_function', type=str, default="mse", help='loss function: mse|..... 损失函数类型.default:mse')
parser.add_argument('-optimizer', type=str, default="Adam", help='optimizer: SGD|RMSProp|Nadam|Adam. default:Adam')
parser.add_argument('-lr', type=int, default=0.1, help='lr. default:0.01')
parser.add_argument('-decay', type=int, default=0.0005, help='decay_rate. default:0.0005')
parser.add_argument('-epoch_num', type=int, default=300, help='epoch number,default:300')
#parser.add_argument('-data_split_rate', type=int, default=0.2, help='train data/test data,default:0.2')
#parser.add_argument('-mask_flag', type=str, default='True', help='mask flag,default:True')
parser.add_argument('-kfold', type=int, default=5, help='k fold num:5|10,default:5')
parser.add_argument('-result_path', type=str, default='../result/temp.log', help='result path,default:../result/temp.log')
parser.add_argument('-info', type=str, default='', help='information')
parser.add_argument('-top_n', type=int, default=10, help='topk 5|10|20|50,default:10')
parser.add_argument('-recall_num', type=int, default=100, help='recall number 50|100|200,default:100')
args = parser.parse_args()


def main():
    print("dataset:%s, emb_size:%s, alpha:%s, beta:%s, KI:%s, MI_method:%s, DR_FLAG:%s, KF:%s, input_method:%s, optimizer:%s, lr:%s, kfold:%s, info:%s, " %
            (str(args.dataset),str(args.emb_size),str(args.alpha),str(args.beta),str(args.KI),str(args.MI_method),str(args.DR_trans_flag),str(args.KF),
            str(args.input_method),str(args.optimizer),str(args.lr),str(args.kfold),str(args.info)))
    #=========数据预处理部分===========
    #加载数据
    #dataset = "yelp"
    df_ratings,df_trust = DP.load_data(args.dataset)

    #生成用户序列
    user_behavior = {}
    if args.rating_del_flag == 'True':
        negative_user_behavior = {}
    else:
        negative_user_behavior = None
    
    DP.get_user_behavior(df_ratings,user_behavior,negative_user_behavior)
    emb_model_path = "../data/" + args.dataset + "/emb_" + str(args.rating_del_flag) + "_" + str(args.emb_size) + ".model"
    if os.path.exists(emb_model_path):
        item_embedding = {}
        emb_model = word2vec.Word2Vec.load(emb_model_path)
        DP.get_item_embedding(df_ratings,item_embedding,emb_model)
    else:
        #构建item图
        item_graph = {}
        DP.build_item_graph(item_graph,user_behavior,negative_user_behavior)
        #随机游走生成节点样本
        sample_item = DP.random_walk(item_graph,args.random_walk_number,args.random_walk_length)

        #embedding训练,生成item_embedding
        item_embedding = {}
        emb_model = DP.w2v_embedding(sample_item,item_embedding,df_ratings,args.emb_size,emb_model_path)
        DP.get_item_embedding(df_ratings,item_embedding,emb_model)

    #用户序列转换为用户向量
    user_embedding = {}
    DP.user_behavior_to_user_embedding(user_behavior,negative_user_behavior,user_embedding,item_embedding)

    #用户对物品的评分(字典形式)
    user_item_rating = {}
    DP.user_item_dict(df_ratings,user_item_rating)

    #==========社交网络中信任值计算=============
    #生成用户一阶朋友
    user_user = {}
    SN.get_user_friends(df_trust,user_user)
    #计算直接信任值
    DT = {}
    SN.user_direct_trust(DT,user_user,user_item_rating,args.alpha)

    #生成二阶朋友
    user_indirect_friends = {}
    SN.get_user_indirect_friends(user_user,user_indirect_friends)
    #计算间接信任值
    IT = {}
    SN.user_indirect_trust(user_user,user_indirect_friends,IT,DT)

    #计算信任值
    S = {}
    SN.user_trust(DT,IT,S,args.beta)

    #================用户多兴趣提取================
    user_multi_embedding = {}
    if args.MI_method == 'KM':
        IE.user_multi_embedding_extract(user_embedding,user_multi_embedding,args.KI,"KM")
    elif args.MI_method == 'DR':
        IE.user_multi_embedding_extract(user_embedding,user_multi_embedding,args.KI,'DR',args.DR_trans_flag)
    elif args.MI_method == 'SA_AA':
        IE.user_multi_embedding_extract(user_embedding,user_multi_embedding,args.KI,'SA_AA',item_embedding=item_embedding,user_item_rating=user_item_rating,dataset=args.dataset,sa_padding_num=args.KI,emb_size=args.emb_size)
    elif args.MI_method == 'SA_DA':
        IE.user_multi_embedding_extract(user_embedding,user_multi_embedding,args.KI,'SA_DA',item_embedding=item_embedding,user_item_rating=user_item_rating,dataset=args.dataset,sa_padding_num=args.KI,emb_size=args.emb_size)
    else:
        print("multi intertests method is not exist!")
        exit(0)

    #================用户多兴趣传播================
    #信任值排序
    Descending_S = {}
    DP.dict_ranking(S,Descending_S)

    #用户多兴趣传播
    user_multi_interest = user_multi_embedding.copy()
    IS.user_interest_communication(S,user_multi_embedding,user_multi_interest,Descending_S,args.KF,user_embedding)

    #================深度学习模型==================
    print("starting train deep model......")
    print("tf.version:tf%s  keras.version:keras%s" % (tf.__version__,keras.__version__))



    #=====================加入其他属性信息，包括用户属性和物品属性======================
    item_feature_onehot,user_feature_onehot,feature_dim = DP.get_feature(args.dataset)

    #平均池化层
    class MeanLayer(keras.layers.Layer):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            
        def build(self,input_shape):
            #self.kernel = self.add_weight(name="kernel",shape=(input_shape[-1],),trainable=False)
            super().build(input_shape)
            
        def call(self,x):
            u = K.mean(x,axis=1)
            return u

    #最大池化层
    class MaxLayer(keras.layers.Layer):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            
        def build(self,input_shape):
            #self.kernel = self.add_weight(name="kernel",shape=(input_shape[-1],),trainable=False)
            super().build(input_shape)
            
        def call(self,x):
            u = K.max(x,axis=1)
            return u

    #user-item注意力层
    class AttentionLayer(keras.layers.Layer):
        def __init__(self,**kwargs):
            super().__init__(**kwargs)
            
        def build(self,input_shape):
            #self.kernel = self.add_weight(name="kernel",shape=(input_shape[-1],),trainable=False)
            super().build(input_shape)
            
        def call(self,x):
            ue,ie = x
            a = K.batch_dot(ue,ie)
            A = K.softmax(a,axis = 1)
            ua = K.batch_dot(tf.transpose(ue,[0,2,1]),A)
            return ua

    #loss函数
    def loss_graphrec(y_true, y_pred):
        return (1 / (2 * args.batch_size)) * K.mean(K.square(y_pred - y_true), axis=-1)



    class RecallCallbacks(keras.callbacks.Callback):
        def __init__(self,model,df_test,user_multi_interest,item_embedding,item_feature_onehot,user_feature_onehot,max_len,k,recall_num):
            self.model = model
            self.df_test= df_test
            self.user_multi_interest = user_multi_interest
            self.item_embedding = item_embedding
            self.item_feature_onehot = item_feature_onehot
            self.user_feature_onehot= user_feature_onehot
            self.max_len= max_len
            self.k = k
            self.recall_num = recall_num
            super().__init__()
        def on_train_end(self, logs: dict):
            recall10 = Eva.get_recall_k(self.model,self.df_test,self.user_multi_interest,self.item_embedding,self.item_feature_onehot,self.user_feature_onehot,self.max_len,self.k,self.recall_num)
            print("\nmodel recall",str(self.k),": ",str(recall10),"\n")
            return


    #k折交叉检验
    if args.kfold > 0:
        kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
        index = 1
        test_result = []
        valication_result = []
        for train_index,test_index in kfold.split(df_ratings):
            print("_____________start %dth training_____________" % index)
            if index == 1:
                #检查GPU占用情况
                GPU_free_id = -1 #不使用GPU
                pynvml.nvmlInit()#初始化
                deviceCount = pynvml.nvmlDeviceGetCount()
                for i in range(deviceCount):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    mem_used = int(meminfo.used/1024/1024)
                    if mem_used < 128:
                        GPU_free_id = i
                        break
                os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_free_id)
                print("__________use GPU:%s__________" % str(GPU_free_id))

            #划分数据集
            df_train_rating,df_test_rating = df_ratings.iloc[train_index],df_ratings.iloc[test_index]
            X_train_u,X_train_i,X_train_f,y_train = DP.get_train_set(df_train_rating,user_multi_interest,item_embedding,item_feature_onehot,user_feature_onehot,args.KI)
            X_test_u,X_test_i,X_test_f,y_test = DP.get_train_set(df_test_rating,user_multi_interest,item_embedding,item_feature_onehot,user_feature_onehot,args.KI)
            


            #构建深度学习模型
            input_u = keras.layers.Input(shape=((args.KI,args.emb_size)),name="user_embedding_input")
            input_i = keras.layers.Input(shape=((args.emb_size,)),name="item_embedding_input")
            input_f = keras.layers.Input(shape=((feature_dim,)),name="feature_embedding_input")
            feature_embedding_layer = keras.layers.Dense(40,activation="relu")(input_f)
            masklayer = keras.layers.Masking(mask_value=0.,)(input_u)
            flattenlayer = keras.layers.Flatten()(masklayer)
            if args.input_method == 'attention':
                att = AttentionLayer()([masklayer,input_i])
                concat = keras.layers.concatenate([att,input_i])
            elif args.input_method == 'mean':
                meanlayer = MeanLayer()(masklayer)
                concat = keras.layers.concatenate([meanlayer,input_i])
            elif args.input_method == 'max':
                maxlayer = MaxLayer()(masklayer)
                concat = keras.layers.concatenate([maxlayer,input_i])
            else:#直接拼接
                concat = keras.layers.concatenate([flattenlayer,input_i])

            #concat2 = concat = keras.layers.concatenate([concat,feature_embedding_layer])
            concat2 = concat = keras.layers.concatenate([flattenlayer,concat,feature_embedding_layer])
            dropout = keras.layers.Dropout(0.5)(concat2)

            hidden1 = keras.layers.Dense(2*args.KI*args.emb_size)(dropout)
            prelu1 = keras.layers.PReLU(alpha_initializer='zeros')(hidden1)
            dropout1 = keras.layers.Dropout(0.5)(prelu1)

            hidden2 = keras.layers.Dense(args.KI*args.emb_size)(dropout1)
            prelu2 = keras.layers.PReLU(alpha_initializer='zeros')(hidden2)
            dropout2 = keras.layers.Dropout(0.5)(prelu2)

            hidden3 = keras.layers.Dense(args.emb_size)(dropout2)
            prelu3 = keras.layers.PReLU(alpha_initializer='zeros')(hidden3)
            dropout3 = keras.layers.Dropout(0.5)(prelu3)

            hidden4 = keras.layers.Dense(args.emb_size // 2)(dropout3)
            prelu4 = keras.layers.PReLU(alpha_initializer='zeros')(hidden4)
            dropout4 = keras.layers.Dropout(0.5)(prelu4)

            output = keras.layers.Dense(1,name="output")(dropout4)
            '''
            #hidden1 = keras.layers.Dense(200,)(concat2)
            hidden1 = keras.layers.Dense(4 * args.emb_size * args.KI,)(concat2)
            #leakyrelu1 = keras.layers.LeakyReLU(alpha=0.3)(hidden1)
            prelu1 = keras.layers.PReLU(alpha_initializer='zeros')(hidden1)
            
            hidden2 = keras.layers.Dense(2 * args.emb_size * args.KI,)(prelu1)
            #leakyrelu2 = keras.layers.LeakyReLU(alpha=0.3)(hidden2)
            prelu2 = keras.layers.PReLU(alpha_initializer='zeros')(hidden2)
            hidden3 = keras.layers.Dense(args.emb_size * args.KI,)(prelu2)
            prelu3 = keras.layers.PReLU(alpha_initializer='zeros')(hidden3)
            output = keras.layers.Dense(1,name="output")(prelu3)
            '''
            deep_model = keras.Model(inputs=[input_u,input_i,input_f],outputs=[output])
            print("deep model is building......")
            if args.optimizer == 'Adam':
                deep_model_optimizer = keras.optimizers.Adam()
            elif args.optimizer == 'RMSProp':
                deep_model_optimizer = keras.optimizers.RMSprop()
            elif args.optimizer == 'Nadam':
                deep_model_optimizer = keras.optimizers.Nadam()
            elif args.optimizer == 'SGD':
                deep_model_optimizer = keras.optimizers.SGD(lr=args.lr,decay=args.decay)
            else:
                deep_model_optimizer = keras.optimizers.Adam()

            #回调函数
            early_stopping_cb = keras.callbacks.EarlyStopping(patience=80,restore_best_weights=True)
            #recall10_cb = RecallCallbacks(deep_model,df_test_rating,X_test_u,X_test_i,X_test_f,10)

            #recall10_cb = RecallCallbacks(deep_model,df_test_rating,user_multi_interest,item_embedding,item_feature_onehot,user_feature_onehot,args.KI,10,args.recall_num)

            deep_model.compile(optimizer=deep_model_optimizer, loss=loss_graphrec, metrics=[keras.metrics.RootMeanSquaredError(),keras.metrics.MeanAbsoluteError()])
            #model training:
            history = deep_model.fit([X_train_u,X_train_i,X_train_f],y_train,batch_size=args.batch_size,epochs=args.epoch_num,validation_split=0.1,validation_freq=1,callbacks=[early_stopping_cb])

            print("_____________start %dth testing_____________" % index)
            test_score = deep_model.evaluate([X_test_u,X_test_i,X_test_f], y_test,)
            test_score = [test_score[1],test_score[2]]  #0loss,1rmse,2mae
            valication_score = [min(history.history["val_root_mean_squared_error"]),min(history.history["val_mean_absolute_error"])]
            print("test set result: %s" % test_score)
            print("valication set result: %s" % valication_score)
            test_result.append(test_score)
            valication_result.append(valication_score)
            index += 1
        
        test_avg_rmse = np.mean(test_result,axis=0)[0]
        test_avg_mae = np.mean(test_result,axis=0)[1]
        val_avg_rmse = np.mean(valication_result,axis=0)[0]
        val_avg_mae = np.mean(valication_result,axis=0)[1]
        test_result = np.array(test_result)
        valication_result = np.array(valication_result)
        test_best_rmse = np.min(test_result[:,0])
        test_best_mae = np.min(test_result[:,1])
        val_best_rmse = np.min(valication_result[:,0])
        val_best_mae = np.min(valication_result[:,1])
        
        print("\ntest dataset avg rmse: %s ,avg mae: %s" % (str(test_avg_rmse),str(test_avg_mae)))
        print("\nvalication dataset avg rmse: %s ,avg mae: %s" % (str(val_avg_rmse),str(val_avg_mae)))
        print("\ntest dataset best rmse: %s ,mae: %s" % (test_best_rmse,test_best_mae))
        print("\nvalication dataset best rmse: %s ,mae: %s\n" % (val_best_rmse,val_best_mae))

        with open(args.result_path,'a+') as f:
            f.write("\n")
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write("\ndataset:%s, emb_size:%s, alpha:%s, beta:%s, KI:%s, MI_method:%s, DR_FLAG:%s, KF:%s, input_method:%s, optimizer:%s, lr:%s, kfold:%s, info:%s, " %
            (str(args.dataset),str(args.emb_size),str(args.alpha),str(args.beta),str(args.KI),str(args.MI_method),str(args.DR_trans_flag),str(args.KF),
            str(args.input_method),str(args.optimizer),str(args.lr),str(args.kfold),str(args.info)))

            f.write("\ntest set result: %s" % test_result)
            f.write("\nvalication set result: %s" % valication_result)
            f.write("\ntest dataset avg rmse: %s ,avg mae: %s" % (str(test_avg_rmse),str(test_avg_mae)))
            f.write("\nvalication dataset avg rmse: %s ,avg mae: %s" % (str(val_avg_rmse),str(val_avg_mae)))
            f.write("\ntest dataset best rmse: %s ,mae: %s" % (test_best_rmse,test_best_mae))
            f.write("\nvalication dataset best rmse: %s ,mae: %s\n" % (val_best_rmse,val_best_mae))
        
    


    '''
    hidden1 = keras.layers.Dense(200,activation="relu")(concat2)
    hidden2 = keras.layers.Dense(200,activation="relu")(hidden1)
    #hidden3 = keras.layers.Dense(50,activation="relu",kernel_regularizer=keras.regularizers.l2(0.01))(hidden2)
    '''



    '''
    #回调函数
    cp_path = "./result/" + args.dataset + "/keras_model.h5"
    checkpoint1 = keras.callbacks.ModelCheckpoint(cp_path,save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
    run_logdir = "./result/" + args.dataset + "/tensorboard"+ .log"
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    '''
    '''
    #early_stopping_cb = keras.callbacks.EarlyStopping(patience=20,restore_best_weights=True)
    #K折交叉检验
    if args.kfold_num != -1:
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=80,)
        #K折交叉检验：args.kfold_num=10
        kfold = KFold(n_splits=args.kfold_num, shuffle=True, random_state=42)

    else:
        #划分数据集
        df_train_rating,df_test_rating = DP.split_dataset(df_ratings,args.data_split_rate)
        #获取训练集
        X_train_u,X_train_i,X_train_f,y_train = DP.get_train_set(df_train_rating,user_multi_interest,item_embedding,item_feature_onehot,user_feature_onehot,args.KI)
        #获取测试集
        X_test_u,X_test_i,X_test_f,y_test = DP.get_train_set(df_test_rating,user_multi_interest,item_embedding,item_feature_onehot,user_feature_onehot,args.KI)

        
        #early_stopping_cb = keras.callbacks.EarlyStopping(patience=100)
        print("_____________start model training_____________")
        history = deep_model.fit([X_train_u,X_train_i,X_train_f],y_train,batch_size=args.batch_size,epochs=args.epoch_num,validation_split=0.1,validation_freq=1,callbacks=[early_stopping_cb])
        print("_____________start model testing______________")
        score = deep_model.evaluate([X_test_u,X_test_i,X_test_f], y_test,)
        print("test set result: %s" % score)
        with open(args.result_path,'a') as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            f.write("\ndataset:%s, emb_size:%s, alpha:%s, beta:%s, KI:%s, MI_method:%s, KF:%s, input_method:%s, mask_flag=%s, CUDA=%s, optimizer=%s, info=%s, DR_FLAG=%s" %
                    (str(args.dataset),str(args.emb_size),str(args.alpha),str(args.beta),str(args.KI),str(args.MI_method),str(args.KF),
                    str(args.input_method),str(args.mask_flag),str(GPU_free_id),str(args.optimizer),str(args.info),str(args.DR_trans_flag)))
            f.write("\nmodel train history:")
            f.write(str(history.history))
            f.write("\ntest set result: %s\n" % score)
    '''




if __name__ == "__main__":
    main()

