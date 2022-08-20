import DataProcessing as DP
import numpy as np
import pandas as pd
from tqdm import tqdm
def get_recall_k(model,df_test,user_multi_interest,item_embedding,item_feature_onehot=None,user_feature_onehot=None,max_len=6,k=10,recall_num=100):
    print("\n---------start recall k calulate-------")
    test_user = list(set(df_test["userid"]))
    test_item = list(set(df_test["itemid"]))
    test_user = [str(x) for x in test_user]
    test_item = [str(x) for x in test_item]
    print("test user number:%s,item number:%s" % (str(len(test_user)),str(len(test_item))))

    #user_multi_recall_dict = multi_recall(user_multi_interest,item_embedding,test_user,test_item,m=50)
    df_user_multi_recall,user_multi_recall_dict = multi_recall(user_multi_interest,item_embedding,test_user,test_item,recall_num)
    print("\n---------start get dict set-------")
    
    #X_u,X_i,X_f = DP.get_dict_set(user_multi_recall_dict,user_multi_interest,item_embedding,item_feature_onehot,user_feature_onehot,max_len)
    X_u,X_i,X_f,y_ = DP.get_train_set(df_user_multi_recall,user_multi_interest,item_embedding,item_feature_onehot,user_feature_onehot,max_len,)
    print("\nfinished load data")
    predict_rating = model.predict([X_u,X_i,X_f])
    print("\nfinished model predict")
    #更新模型预测评分
    index = 0
    for userid,s in user_multi_recall_dict.items():
        for itemid,r in s.items():
            user_multi_recall_dict[userid].update({itemid:predict_rating[index][0]})

    #模型评分排序
    dec_user_multi_recall_dict = {}
    DP.dict_ranking(user_multi_recall_dict,dec_user_multi_recall_dict)

    #每个用户topk
    u_topk = {}
    for u,s in dec_user_multi_recall_dict.items():
        topk = list(s.keys())[:k]
        u_topk[u] = topk
    
    u_true = {}
    #每个用户真实点击的物品
    for row in df_test.itertuples():
        userid = str(getattr(row, 'userid'))
        itemid = str(getattr(row, 'itemid'))
        if u_true.get(userid) is None:
            u_true[userid] = []
        u_true[userid].append(itemid)
    #recall
    recall = []
    for userid in test_user:
        u_true_num = len(u_true[userid])
        intersection = list(set(u_topk[userid]) & set(u_true[userid]))
        u_pre_num = len(intersection)
        recall.append([u_pre_num,u_true_num,float(u_pre_num)/u_true_num])
    
    user_len = len(test_user)
    result1 = np.sum(recall,axis=0)
    recall_k10 = result1[0] / result1[1]
    return recall_k10



def get_top_k(user_item_rating,k):
    topk_rating = {}
    for u,s in user_item_rating.items():
        user_seq = list(s.keys())
        if len(user_seq) < k:
            topk_rating[u] = user_seq
        else:
            topk_rating[u] = user_seq[:k]

    return topk_rating

def multi_recall(user_multi_interest,item_embedding,test_user,test_item,m=100):
    user_multi_recall = {}
    print("-------start calute user interest and item smi------\n")
    for userid in tqdm(test_user):
        user_recall = {}
        #对兴趣进行m个物品的召回,各兴趣进行召回然后综合排序
        multi_len = len(user_multi_interest[userid])
        for mi in range(multi_len):
            multi_interest_i = user_multi_interest[userid][mi]
            for itemid in test_item:
                if user_recall.get(itemid) is None:
                    user_recall[itemid] = np.dot(multi_interest_i,item_embedding[itemid])
                else:
                    user_recall[itemid] = max(user_recall[itemid],np.dot(multi_interest_i,item_embedding[itemid]))
        user_multi_recall[userid] = {}
        user_multi_recall[userid].update(user_recall)

    dec_user_multi_recall = {}
    DP.dict_ranking(user_multi_recall,dec_user_multi_recall)
        
    
    user_multi_recall_dict = {}
    print("---------start select 100 item for each user-------\n")
    for userid,item in tqdm(dec_user_multi_recall.items()):
        index = 0
        for itemid,rating in item.items():
            if user_multi_recall_dict.get(userid) is None:
                user_multi_recall_dict[userid] = {}
            user_multi_recall_dict[userid].update({itemid:rating})
            index += 1
            if index == m:
                break
    
    userid_list = []
    itemid_list = []
    rating_list = []
    
    for userid,item in tqdm(user_multi_recall_dict.items()):
        for itemid,rating in item.items():
            userid_list.append(userid)
            itemid_list.append(itemid)
            rating_list.append(rating)
    data = {'userid':userid_list,"itemid":itemid_list,"rating":rating_list}
    df_user_multi_recall = pd.DataFrame.from_dict(data)

    return df_user_multi_recall,user_multi_recall_dict






        
        
        

        



