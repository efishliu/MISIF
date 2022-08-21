# MISIF
项目介绍： A Multi-Interest and Social Interest-Field Framework for Social Recommendation 论文源码
### 主要工作
* 我们提出了一个基于多兴趣感知和社交兴趣场的社交推荐框架(MISIF)，该框架能融合利用用户社交关系，用户历史行为和物品属性信息进行社交推荐。
* MISIF采用了三种无监督学习方法提取用户多兴趣以获得更强的表达能力。据我们目前所知，MISIF是第一个使用胶囊网络进行社交推荐的模型，我们设计了一个两层的胶囊网络并使用动态路由方法提取用户多兴趣。
* 我们在社交网络中构建用户社交兴趣场，根据用户的不同兴趣选择不同的朋友进行兴趣扩散融合，减轻了社交关系中的兴趣噪声。
* 我们在三个公开数据集上证明了所提出的MISIF框架的有效性。

### MISIF项目模块
* [main](./MISIF/code/main.py)：MISIF主函数  
* [DataProcessing](./MISIF/code/DataProcessing.py)：数据预处理模块  
* [SocialNetwork](./MISIF/code/SocialNetwork.py)：社交网络处理模块  
* [InterestExtraction](./MISIF/code/InterestExtraction.py)：用户多兴趣提取模块
* [InterestSpread](./MISIF/code/InterestSpread.py): 用户多兴趣传播模块  
* [Evaluation](./MISIF/code/Evaluation.py)：模型评估模块

### MISIF模型运行与评估
`sh ./MISIF/code/scripts/auto_train.sh`  
### 对比实验
* [MF](./BASELINE/mf.py)
* [PMF](./BASELINE/pmf.py)
* [CUNE](./BASELINE/social_cune.py)
* [SoMF](./BASELINE/social_mf.py)
* [SoRec](./BASELINE/social_rec.py)
* [GraphRec](./GraphRec)
