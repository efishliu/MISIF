import pandas as pd
import numpy as np
from scipy.special import softmax
from tqdm import tqdm


#=================社交网络相关函数===================
#生成一阶朋友
def get_user_friends(df_trust,user_user):
    for row in df_trust.itertuples():
        user_id = str(getattr(row,"userid"))
        friend_id = str(getattr(row,"friendid"))
        if user_user.get(user_id) is None:
            user_user[user_id] = [friend_id,]
        else:
            user_user[user_id].append(friend_id)

#皮尔逊相关系数计算
def pcc(user_i,user_j,user_item_rating):
    user_i_item_rating = user_item_rating[user_i]
    user_j_item_rating = user_item_rating[user_j]
    Ri_avg= np.mean(list(user_i_item_rating.values()))
    Rj_avg = np.mean(list(user_j_item_rating.values()))
    Ric = []
    Rjc = []
    for i in user_i_item_rating.keys():
        if user_j_item_rating.get(i) is not None:
            Ric.append(user_i_item_rating[i])
            Rjc.append(user_j_item_rating[i])
    v1 = 0
    v2 = 0
    v3 = 0
    for i in range(len(Ric)):
        v1 = v1 + (Ric[i]-Ri_avg)* (Rjc[i]-Rj_avg)
        v2 = v2 + (Ric[i]-Ri_avg) ** 2
        v3 = v3 + (Rjc[i]-Rj_avg) ** 2
    if v1 == 0 or v2 == 0 or v3 == 0:
        #print("用户%s与朋友%s的共同兴趣值为%d"%(user_i,user_j,0))
        value = 0
    else:
        value = v1 / ((np.sqrt(v2)) * (np.sqrt(v3)))
    #print("用户%s与朋友%s的共同兴趣值为%s"%(user_i,user_j,value))
    return value

#共同好友值计算
def common_friends(user_user,user_i,user_j):
    user_i_friend = user_user[user_i]
    if user_user.get(user_j) is None:
        user_j_friend = []
    else:
        user_j_friend = user_user[user_j]
    common_friend = (set(user_i_friend) & set(user_j_friend))
    value = len(common_friend) / len(user_i_friend)
    if value == 0:
        value = 1 / len(user_i_friend) #给予一个默认值
    #print("用户%s与朋友%s的共同好友值为%s"%(user_i,user_j,value))
    return value

#计算直接信任值
def user_direct_trust(DT,user_user,user_item_rating,alpha):
    print("calculating user direct trust value:")
    for u,f in tqdm(user_user.items()):
        for v in f:
            if DT.get(u) is None:
                DT[u] = {}
            direct_trust = alpha*pcc(u,v,user_item_rating) + (1-alpha)*common_friends(user_user,u,v)
            DT[u].update({v:direct_trust})

#生成二阶朋友
def get_user_indirect_friends(user_user,user_indirect_friends):
    for u,v in user_user.items():
        user_indirect_friends[u] = set()
        for f in v:
            if user_user.get(f) is not None:
                for in_f in user_user[f]:
                    user_indirect_friends[u].add(in_f)

#计算间接信任值
def user_indirect_trust(user_user,user_indirect_friends,IT,DT):
    print("calculating user indirect trust value:")
    for u,v in tqdm(user_indirect_friends.items()):#初始化
        if v:
            IT[u] = {}
        for f in v:
            IT[u].update({f:0})
    for u,v in DT.items():
        for f in v.keys():
            if  DT.get(f) is not None:
                for inf in DT[f].keys():
                    IT[u][inf] = IT[u][inf] + DT[u][f] * DT[f][inf]
#计算信任值
def user_trust(DT,IT,S,beta):
    print("calculating user trust value:")
    for u,v in tqdm(DT.items()):
        S[u] = {}
        for f in v:
            S[u].update({f:beta*DT[u][f]})
    for u,v in tqdm(IT.items()):
        if S.get(u) is None:
            S[u] = {}
            for f in v:
                S[u].update({f:(1-beta)*IT[u][f]})
        else:
            for f in v:
                if S[u].get(f) is None:
                    S[u].update({f:(1-beta)*IT[u][f]})
                else:
                    S[u].update({f:beta*DT[u][f] + (1-beta)*IT[u][f]})