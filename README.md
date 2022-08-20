# MISIF
项目介绍： A Multi-Interest and Social Interest-Field Framework for Social Recommendation 论文源码

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
