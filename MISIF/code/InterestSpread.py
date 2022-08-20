import numpy as np
from scipy.special import softmax
from tqdm import tqdm

#==================用户多兴趣传播相关函数===================
#获取信任值前k个朋友
def get_topk_friends(DS,u,k,user_embedding):
    #启发式选取朋友个数
    k_f = []
    if k == -1:
        k_ = int(1/max(0.5,np.log(len(user_embedding[u]))) * 10) #当点击的物品数目为1时，选取的朋友最多为20
    else:
        k_ = k
    i = 0
    for m,n in DS[u].items():
        k_f.append(m)
        i += 1
        if i >= k_:
            break
    return k_f

#用户多兴趣传播
def user_interest_communication(S,user_multi_embedding,user_multi_interest,Descending_S,KF,user_embedding):
    print("user multi interest communication:")
    #user_multi_interest = user_multi_embedding.copy()
    for u,v in tqdm(S.items()):
        for f in get_topk_friends(Descending_S,u,KF,user_embedding):
            A_uf = np.matmul(user_multi_embedding[u],user_multi_embedding[f].T)
            A_uf = softmax(A_uf,axis=1)
            if np.isnan(A_uf).any():#检查数据
                print('A_uf is nan! index = (%s,%s)' % (u,f))
            user_multi_interest[u] = user_multi_interest[u] + S[u][f] * np.matmul(A_uf,user_multi_embedding[f])
        if np.isnan(user_multi_interest[u]).any():#检查数据
            print('has nan! index = %s,' % u)
    print('user multi interest communication finished.')
