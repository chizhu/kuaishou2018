# kuaishou2018
高校大数据-快手活跃用户预测 初赛第7 复赛35 [大赛地址](https://www.kesci.com/home/competition/5ab8c36a8643e33f5138cba4)

--------
### 数据构造
按照周星星们的划分方式：<br>
1-16 data1  标签17-23 ->8-23 data2 标签 24-30 <br>
data1+2 -> 15-30 data3 标签 31-37<br>
### 特征工程
从提供的四个表register,launch,action,video四个表里提取一些统计特征:max,min,mean,sum,std,skew,kurt，还有差分特征，统计启动/播放/拍摄的时间间隔等。不得不说，这个比赛不像往常比赛，很多特征都会是“有毒”的，所以，整个比赛下来我总共构造了三个版本的特征。初赛一直用的是版本一，初赛是使用F1作为评价指标，显然个数和阈值就显得很重要，这里面有各种trick。到了复赛，指标换成auc，更加直接。数据量大了，我的特征不work了，开始重新构造一些新特征，然而上分缓慢，最后才发现，特征多了可能有毒，通过删除特征，我差不多提升了6个万分点。

### 模型
初期一直用的lgb单模型，速度快，真不是一般的快，和xgb,cat对比起来很明显。
### 模型融合 
我主要使用 xgb,lgb,cat和nn的融合。大佬们都说nn和lgb的融合效果好。可能我的姿势不对，哎 上分艰难。加上单人作战，特征单一，模型之间差异性太小，融合效果很不理想，凉凉。
###  最后
kescid的平台真的不错，从网站ui到线上k-lab环境。感谢群里各种大佬的分享，让我学到很多。之后，还要多多学习，尤其是nn的构造。

