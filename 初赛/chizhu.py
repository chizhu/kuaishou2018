import pandas as pd
import os
import datetime
import lightgbm as lgb
from sklearn.metrics import auc, log_loss, roc_auc_score, f1_score, recall_score, precision_score
import warnings
warnings.filterwarnings("ignore")
#######配置参数区#############
###设置A榜数据路径,请工作人员修改
pathA='/Users/chizhu/data/competition_data/快手活跃用户.tar/'
###设置B榜数据路径,请工作人员修改
pathB = '/Users/chizhu/data/competition_data/chusai_b_train.tar/'
###########config end#######


t1 = datetime.datetime.now()
###加载数据
print("加载A数据...")
action = pd.read_csv(pathA+"user_activity_log.txt", sep="\t", header=None)
action.columns = ['user_id', 'day', 'page',
                  'video_id', 'author_id', "action_type"]

register = pd.read_csv(pathA+"user_register_log.txt", sep="\t", header=None)
register.columns = ['user_id', 'register_day', 'register_type', "device_type"]

launch = pd.read_csv(pathA+"app_launch_log.txt", sep="\t", header=None)
launch.columns = ['user_id', 'day']

video = pd.read_csv(pathA+"video_create_log.txt", sep="\t", header=None)
video.columns = ['user_id', 'day']
action = action.sort_values(by=['user_id', "day"])
launch = launch.sort_values(by=['user_id', "day"])
video = video.sort_values(by=['user_id', "day"])
register = register.sort_values(by=['user_id', "register_day"])

###特征提取
 
def get_inpred_count(temp, flag):
    
    count = 0
    for i in [flag+'_lastdaytopred+day_dis_mean', flag+'_lastdaytopred+day_dis_max', flag+"_lastdaytopred+day_dis_min"]:
        if temp[i].values[0] > 0 and temp[i].values[0] <= 7:
            count += 1
    return count


def get_launch_feat(df, pred_firstday):
    temp = pd.DataFrame(index=range(1))
    temp['user_id'] = df['user_id'].unique()[0]
    temp['launch_count'] = len(df)
    temp['launch_day_mean'] = df['dur_day'].mean()
    temp['launch_day_min'] = df['dur_day'].min()
    temp['launch_day_max'] = df['dur_day'].max()
    last_7days_len = len(df[df['dur_day'] <= 7])
    temp['last_7day_launch_rate'] = last_7days_len/len(df)
    last_5days_len = len(df[df['dur_day'] <= 5])
    temp['last_5day_launch_rate'] = last_5days_len/len(df)
    last_3days_len = len(df[df['dur_day'] <= 3])
    temp['last_3day_launch_rate'] = last_3days_len/len(df)

    temp['launch_day_std'] = df['dur_day'].std()
    temp['launch_day_skew'] = df['dur_day'].skew()
    temp['last_launch_day_before_pred'] = pred_firstday-df['day'].values[-1]
    df_diff = df.diff(1)
    temp['launch_day_dis_max'] = df_diff['day'].max()
    temp['launch_day_dis_min'] = df_diff['day'].min()
    temp['launch_day_dis_mean'] = df_diff['day'].mean()
    temp['launch_day_dis_std'] = df_diff['day'].std()
    temp['launch_day_dis_skew'] = df_diff['day'].skew()

    temp['launch_lastdaytopred+day_dis_mean'] = 0 - \
        temp['last_launch_day_before_pred'].values[0] + \
        temp['launch_day_dis_mean'].values[0]

    temp['launch_lastdaytopred+day_dis_max'] = 0 - \
        temp['last_launch_day_before_pred'].values[0] + \
        temp['launch_day_dis_max'].values[0]
    temp['launch_lastdaytopred+day_dis_min'] = 0 - \
        temp['last_launch_day_before_pred'].values[0] + \
        temp['launch_day_dis_min'].values[0]

    temp["launch_inpredday_count"] = get_inpred_count(temp, "launch")

    for i in range(pred_firstday-7, pred_firstday):
        if i in df['day'].values:
            temp['last_'+str(pred_firstday-i)+"_launch_num"] = 1
        else:
            temp['last_'+str(pred_firstday-i)+"_launch_num"] = 0

    for i in [2, 3, 5, 7]:
        temp["launch_" +
             str(i)+"_continuous_day_count"] = get_continuous_day_count(i, df)
    return temp


def get_continuous_day_count(cont_num, df):
    day_set = df['day'].unique()

    day_count = 0
    for i in day_set:
        flag = 0
        for j in range(1, cont_num):
            if i+j in day_set:
                flag += 1

        if flag == cont_num-1:
            day_count += 1

    return day_count


def get_video_feat(df, pred_firstday):
    temp = pd.DataFrame(index=range(1))
    temp['user_id'] = df['user_id'].unique()[0]
    temp['video_count'] = len(df)

    temp['day_video_mean'] = df.groupby('day', as_index=False).count()[
        'user_id'].mean()
    temp['day_video_std'] = df.groupby('day', as_index=False).count()[
        'user_id'].std()
    temp['day_video_max'] = df.groupby('day', as_index=False).count()[
        'user_id'].max()
    temp['day_video_min'] = df.groupby('day', as_index=False).count()[
        'user_id'].min()
    temp['day_video_skew'] = df.groupby('day', as_index=False).count()[
        'user_id'].skew()
    temp['day_video_last'] = df.groupby('day', as_index=False).count()[
        'user_id'].values[-1]
    b = dict(zip(df.groupby('day', as_index=False).count()[
             'day'].values, df.groupby('day', as_index=False).count()['user_id'].values))
    for i in range(pred_firstday-7, pred_firstday):
        if i in b.keys():
            temp['last_'+str(pred_firstday-i)+"_video_num"] = b[i]
        else:
            temp['last_'+str(pred_firstday-i)+"_video_num"] = 0
    temp['video_day_count'] = df.groupby('day', as_index=False).count()[
        'user_id'].count()
    df_temp = df.groupby('day', as_index=False).count()
    df_temp['dur_day'] = df_temp['day'].apply(lambda x: pred_firstday-x, 1)

    temp['video_day_mean'] = df_temp['dur_day'].mean()
    temp['video_day_min'] = df_temp['dur_day'].min()
    temp['video_day_max'] = df_temp['dur_day'].max()
    last_7days_len = len(df_temp[df_temp['dur_day'] <= 7])
    temp['last_7day_video_rate'] = last_7days_len/len(df_temp)

    last_5days_len = len(df_temp[df_temp['dur_day'] <= 5])
    temp['last_5day_video_rate'] = last_5days_len/len(df)
    last_3days_len = len(df_temp[df_temp['dur_day'] <= 3])
    temp['last_3day_video_rate'] = last_3days_len/len(df)

    temp['video_day_std'] = df_temp['dur_day'].std()
    temp['video_day_skew'] = df_temp['dur_day'].skew()
    temp['last_video_day_before_pred'] = pred_firstday-df_temp['day'].values[-1]
    df_diff = df_temp.diff(1)
    temp['video_day_dis_max'] = df_diff['day'].max()
    temp['video_day_dis_min'] = df_diff['day'].min()
    temp['video_day_dis_mean'] = df_diff['day'].mean()
    temp['video_day_dis_std'] = df_diff['day'].std()
    temp['video_day_dis_skew'] = df_diff['day'].skew()

    for i in [2, 3, 5, 7]:
        temp["video_" +
             str(i)+"_continuous_day_count"] = get_continuous_day_count(i, df_temp)
    return temp


def get_action_feat(df, pred_firstday):
    temp = pd.DataFrame(index=range(1))
    temp['user_id'] = df['user_id'].unique()[0]
    temp['action_count'] = len(df)

    temp['day_action_mean'] = df.groupby('day', as_index=False).count()[
        'user_id'].mean()
    temp['day_action_std'] = df.groupby(
        'day', as_index=False).count()['user_id'].std()
    temp['day_action_max'] = df.groupby(
        'day', as_index=False).count()['user_id'].max()
    temp['day_action_min'] = df.groupby(
        'day', as_index=False).count()['user_id'].min()
    temp['day_action_skew'] = df.groupby('day', as_index=False).count()[
        'user_id'].skew()
    temp['day_action_last'] = df.groupby('day', as_index=False).count()[
        'user_id'].values[-1]
    b = dict(zip(df.groupby('day', as_index=False).count()[
             'day'].values, df.groupby('day', as_index=False).count()['user_id'].values))
    for i in range(pred_firstday-7, pred_firstday):
        if i in b.keys():
            temp['last_'+str(pred_firstday-i)+"_action_num"] = b[i]
        else:
            temp['last_'+str(pred_firstday-i)+"_action_num"] = 0
    temp['action_day_count'] = df.groupby('day', as_index=False).count()[
        'user_id'].count()
    df_temp = df.groupby('day', as_index=False).count()
    df_temp['dur_day'] = df_temp['day'].apply(lambda x: pred_firstday-x, 1)

    temp['action_day_mean'] = df_temp['dur_day'].mean()
    temp['action_day_min'] = df_temp['dur_day'].min()
    temp['action_day_max'] = df_temp['dur_day'].max()
    last_7days_len = len(df_temp[df_temp['dur_day'] <= 7])
    temp['last_7day_action_rate'] = last_7days_len/len(df_temp)

    last_5days_len = len(df_temp[df_temp['dur_day'] <= 5])
    temp['last_5day_action_rate'] = last_5days_len/len(df)
    last_3days_len = len(df_temp[df_temp['dur_day'] <= 3])
    temp['last_3day_action_rate'] = last_3days_len/len(df)

    temp['action_day_std'] = df_temp['dur_day'].std()
    temp['action_day_skew'] = df_temp['dur_day'].skew()
    temp['last_action_day_before_pred'] = pred_firstday - \
        df_temp['day'].values[-1]
    df_diff = df_temp.diff(1)
    temp['action_day_dis_max'] = df_diff['day'].max()
    temp['action_day_dis_min'] = df_diff['day'].min()
    temp['action_day_dis_mean'] = df_diff['day'].mean()
    temp['action_day_dis_std'] = df_diff['day'].std()
    temp['action_day_dis_skew'] = df_diff['day'].skew()

    temp['action_lastdaytopred+day_dis_mean'] = 0 - \
        temp['last_action_day_before_pred'].values[0] + \
        temp['action_day_dis_mean'].values[0]

    temp['action_lastdaytopred+day_dis_max'] = 0 - \
        temp['last_action_day_before_pred'].values[0] + \
        temp['action_day_dis_max'].values[0]
    temp['action_lastdaytopred+day_dis_min'] = 0 - \
        temp['last_action_day_before_pred'].values[0] + \
        temp['action_day_dis_min'].values[0]

    temp["action_inpredday_count"] = get_inpred_count(temp, "action")

    for i in [2, 3, 5, 7]:
        temp["action_" +
             str(i)+"_continuous_day_count"] = get_continuous_day_count(i, df_temp)

    ###page

    temp['page_type_count'] = len(df.page.unique())
    for i in range(5):
        temp['page_'+str(i)+"_count"] = len(df[df.page == i])/len(df)
    ###action_type
    temp['action_type_count'] = len(df.action_type.unique())
    for i in range(6):
        temp['action_type_' +
             str(i)+"_count"] = len(df[df.action_type == i])/len(df)

    df['id_same'] = df["user_id"]-df['author_id']
    a = len(df[df['id_same'] == 0])
    temp['author_and_id_the_same'] = a/len(df)

    return temp


def get_action_author_feat(action):
    author_temp = action.groupby(
        ['author_id', "user_id"], as_index=False).count()
    author = author_temp.groupby("author_id", as_index=False).count()[
        ['author_id', 'user_id']]
    author.rename(columns={'user_id': 'author_count',
                           'author_id': 'user_id', }, inplace=True)
    action_temp = action.groupby(
        "user_id", as_index=False).count()[['user_id']]
    action_author = pd.merge(action_temp, author, on="user_id", how="left")
    return action_author


###标签的构建
def get_label(data, launch, pred_begin=None, pred_end=None):
    launch_data = launch[(launch.day >= pred_begin) & (launch.day <= pred_end)]
    target = launch_data.groupby(
        "user_id", as_index=False).count()[['user_id']]
    target['label'] = 1
    train = pd.merge(data, target, on="user_id", how="left")
    train['label'] = train['label'].fillna(0)
    return train


###数据集的构建
def get_data(action, video, register, launch, register_lastday, train_begin, pred_firstday):
    register_data = register[register.register_day <= register_lastday]
    launch_data = launch[(launch.day >= train_begin) &
                         (launch.day <= register_lastday)]
    video_data = video[(video.day >= train_begin) &
                       (video.day <= register_lastday)]
    action_data = action[(action.day >= train_begin) &
                         (action.day <= register_lastday)]

    register_data['register_to_predday'] = register_data.apply(
        lambda x: pred_firstday-x['register_day'], 1)

    launch_data['dur_day'] = launch_data['day'].apply(
        lambda x: pred_firstday-x, 1)
    launch_feat = launch_data.groupby("user_id").apply(
        lambda x: get_launch_feat(x, pred_firstday))
    launch_feat.index = range(len(launch_feat))

    video_feat = video_data.groupby("user_id").apply(
        lambda x: get_video_feat(x, pred_firstday))
    video_feat.index = range(len(video_feat))

    action_feat = action_data.groupby("user_id").apply(
        lambda x: get_action_feat(x, pred_firstday))
    action_feat.index = range(len(action_feat))

    action_other = get_action_author_feat(action_data)

    ####merge

    data = pd.merge(register_data, launch_feat, on="user_id", how="left")
    data = pd.merge(data, video_feat, on="user_id", how="left")
    data = pd.merge(data, action_feat, on="user_id", how="left")
    data = pd.merge(data, action_other, on="user_id", how="left")

    return data


print("开始执行数据划分与产生特征,每跑一次特征差不多需要20-30分钟，请耐心等候...")
#####A_data
print("构建A数据...")
####data1
print("A_data1...")
start = datetime.datetime.now()
data1 = get_data(action, video, register, launch, 16, 1, 17)
data1 = get_label(data1, launch, 17, 23)
print("use time:", datetime.datetime.now()-start, " s")
####data2
print("A_data2...")
start = datetime.datetime.now()
data2 = get_data(action, video, register, launch, 23, 8, 24)
data2 = get_label(data2, launch, 24, 30)
print("use time:", datetime.datetime.now()-start, " s")

###data3
print("A_data3...")
start = datetime.datetime.now()
data3 = get_data(action, video, register, launch, 30, 15, 31)
print("use time:", datetime.datetime.now()-start, " s")

####数据合并
data12 = pd.concat([data1, data2])


###加载数据
print("加载B数据...")
action = pd.read_csv(pathB+"user_activity_log.txt", sep="\t", header=None)
action.columns = ['user_id', 'day', 'page',
                  'video_id', 'author_id', "action_type"]

register = pd.read_csv(pathB+"user_register_log.txt", sep="\t", header=None)
register.columns = ['user_id', 'register_day', 'register_type', "device_type"]

launch = pd.read_csv(pathB+"app_launch_log.txt", sep="\t", header=None)
launch.columns = ['user_id', 'day']

video = pd.read_csv(pathB+"video_create_log.txt", sep="\t", header=None)
video.columns = ['user_id', 'day']
action = action.sort_values(by=['user_id', "day"])
launch = launch.sort_values(by=['user_id', "day"])
video = video.sort_values(by=['user_id', "day"])
register = register.sort_values(by=['user_id', "register_day"])


print("构建B数据...")
####data1
print("B_data1...")
start = datetime.datetime.now()
bdata1 = get_data(action, video, register, launch, 16, 1, 17)
bdata1 = get_label(bdata1, launch, 17, 23)
print("use time:", datetime.datetime.now()-start, " s")
####data2
print("B_data2...")
start = datetime.datetime.now()
bdata2 = get_data(action, video, register, launch, 23, 8, 24)
bdata2 = get_label(bdata2, launch, 24, 30)
print("use time:", datetime.datetime.now()-start, " s")

###data3
print("B_data3...")
start = datetime.datetime.now()
bdata3 = get_data(action, video, register, launch, 30, 15, 31)
print("use time:", datetime.datetime.now()-start, " s")

###数据合并
bdata12 = pd.concat([bdata1, bdata2])
data_all = pd.concat([data12, bdata12])###a+b

#####模型训练
print("开始模型训练")
params = {
    'boosting_type': 'gbdt',
    'metric': {'auc', },
    'learning_rate': 0.01,
    'verbose': 0,
    'num_leaves': 32,
    'objective': 'binary',
    'feature_fraction': 0.4,
    'bagging_fraction': 0.7,  
    'seed': 1024,
    'nthread': 12,
}

###model1
print("model1...")
train = bdata12
test = bdata3

features = ['device_type', 'action_count', 'launch_day_min',
            'launch_lastdaytopred+day_dis_mean', 'page_3_count',
            'day_action_last', 'action_day_min', 'last_launch_day_before_pred',
            'page_1_count', 'register_type', 'day_action_max',
            'day_action_mean', 'action_type_1_count', 'last_5day_action_rate',
            'launch_day_mean', 'launch_day_dis_mean', 'last_1_action_num',
            'launch_day_skew', 'launch_day_dis_std', 'last_3day_action_rate',
            'launch_lastdaytopred+day_dis_max', 'register_day',
            'day_action_std', 'action_type_0_count',
            'last_action_day_before_pred', 'day_action_min',
            'action_lastdaytopred+day_dis_mean', 'action_day_skew',
            'launch_lastdaytopred+day_dis_min', 'page_0_count',
            'launch_day_std', 'day_action_skew', 'last_3_action_num',
            'launch_2_continuous_day_count', 'launch_count', 'page_2_count',
            'launch_day_dis_skew', 'last_4_action_num', 'action_day_dis_std',
            'action_type_2_count', 'action_day_mean', 'register_to_predday',
            'action_lastdaytopred+day_dis_max', 'action_day_std',
            'action_lastdaytopred+day_dis_min', 'last_1_launch_num',
            'launch_3_continuous_day_count', 'launch_day_dis_min',
            'last_5day_launch_rate', 'last_3day_launch_rate',
            'last_2_launch_num', 'launch_day_dis_max', 'page_4_count',
            'last_6_action_num', 'action_day_dis_mean',
            'last_7day_action_rate', 'video_day_mean',
            'launch_5_continuous_day_count', 'action_day_max',
            'action_day_dis_skew', 'video_day_min', 'page_type_count',
            'video_count', 'author_and_id_the_same', 'last_2_action_num',
            'action_type_3_count', 'last_7_action_num', 'last_5day_video_rate',
            'action_type_count', 'action_2_continuous_day_count',
            'last_5_action_num', 'launch_day_max', 'last_4_launch_num',
            'action_day_dis_max', 'action_day_count', 'action_day_dis_min',
            'day_video_mean', 'day_video_std', 'video_day_max',
            'launch_inpredday_count', 'last_video_day_before_pred',
            'action_3_continuous_day_count', 'action_5_continuous_day_count',
            'last_7day_launch_rate', 'last_5_launch_num',
            'last_3day_video_rate']
label = 'label'

dtrain = lgb.Dataset(train[features], label=train[label])
dtest=lgb.Dataset(train[features], label=train[label])

model = lgb.train(params, dtrain, num_boost_round=750,
                  valid_sets=[dtrain, dtest],
                  verbose_eval=200,
                  )
pred = model.predict(test[features], num_iteration=model.best_iteration)
####model1_result
pred1=pred

###model2
print("model2...")
train = data_all
test = bdata3
features = ['device_type', 'action_count', 'register_type', 'day_action_last',
            'last_1_action_num', 'page_1_count', 'page_3_count',
            'launch_day_skew', 'launch_lastdaytopred+day_dis_mean',
            'launch_day_min', 'day_action_max', 'day_action_mean',
            'action_type_1_count', 'launch_day_mean', 'last_5day_action_rate',
            'page_0_count', 'day_action_min', 'day_action_std',
            'launch_day_dis_mean', 'action_day_min', 'action_day_skew',
            'last_launch_day_before_pred', 'action_type_0_count',
            'launch_day_dis_std', 'page_2_count', 'last_3_action_num',
            'action_type_2_count', 'last_3day_action_rate', 'day_action_skew',
            'last_2_action_num', 'last_4_action_num', 'launch_day_dis_skew',
            'action_lastdaytopred+day_dis_mean',
            'launch_lastdaytopred+day_dis_max', 'register_day',
            'launch_day_std', 'action_day_std', 'action_day_dis_std',
            'action_day_mean', 'launch_count', 'launch_2_continuous_day_count',
            'page_4_count', 'last_7_action_num', 'last_action_day_before_pred',
            'launch_lastdaytopred+day_dis_min', 'last_6_action_num',
            'action_type_3_count', 'launch_3_continuous_day_count',
            'author_and_id_the_same', 'register_to_predday',
            'action_day_dis_skew', 'video_day_mean', 'last_3day_launch_rate',
            'action_lastdaytopred+day_dis_min', 'action_day_dis_mean',
            'last_5_action_num', 'launch_day_dis_max',
            'action_lastdaytopred+day_dis_max', 'last_5day_launch_rate',
            'action_day_max', 'page_type_count', 'last_7day_action_rate',
            'launch_5_continuous_day_count', 'video_day_min',
            'last_5day_video_rate', 'last_1_launch_num', 'day_video_mean',
            'action_3_continuous_day_count', 'video_count',
            'launch_day_dis_min', 'last_2_launch_num', 'day_video_std',
            'action_2_continuous_day_count', 'last_video_day_before_pred',
            'action_day_dis_max', 'last_7day_launch_rate', 'video_day_max',
            'last_3day_video_rate', 'day_video_max', 'action_day_count',
            'action_type_count', 'action_5_continuous_day_count',
            'launch_day_max', 'video_day_std', 'last_5_launch_num',
            'day_video_last', 'last_7_video_num', 'last_4_video_num',
            'video_day_skew', 'day_video_min', 'video_day_dis_mean',
            'action_day_dis_min', 'last_4_launch_num',
            'action_inpredday_count', 'day_video_skew', 'last_5_video_num',
            'last_1_video_num', 'launch_inpredday_count', 'last_3_video_num',
            'video_2_continuous_day_count', 'video_day_dis_std',
            'launch_7_continuous_day_count', 'last_6_launch_num',
            'last_3_launch_num', 'action_type_5_count', 'author_count',
            'last_7_launch_num']

label = 'label'

dtrain = lgb.Dataset(train[features], label=train[label])
dtest = lgb.Dataset(train[features], label=train[label])

model = lgb.train(params, dtrain, num_boost_round=750,
                  valid_sets=[dtrain, dtest],
                  verbose_eval=200,
                  )
pred = model.predict(test[features], num_iteration=model.best_iteration)
####model2_result
pred2 = pred

###model3
print("model3...")
train = bdata12
test = bdata3
features = ['device_type', 'action_count', 'launch_day_min',
            'launch_lastdaytopred+day_dis_mean', 'page_3_count',
            'day_action_last', 'action_day_min', 'last_launch_day_before_pred',
            'page_1_count', 'register_type', 'day_action_max',
            'day_action_mean', 'action_type_1_count', 'last_5day_action_rate',
            'launch_day_mean', 'launch_day_dis_mean', 'last_1_action_num',
            'launch_day_skew', 'launch_day_dis_std', 'last_3day_action_rate',
            'launch_lastdaytopred+day_dis_max', 'register_day',
            'day_action_std', 'action_type_0_count',
            'last_action_day_before_pred', 'day_action_min',
            'action_lastdaytopred+day_dis_mean', 'action_day_skew',
            'launch_lastdaytopred+day_dis_min', 'page_0_count',
            'launch_day_std', 'day_action_skew', 'last_3_action_num',
            'launch_2_continuous_day_count', 'launch_count', 'page_2_count',
            'launch_day_dis_skew', 'last_4_action_num', 'action_day_dis_std',
            'action_type_2_count', 'action_day_mean', 'register_to_predday',
            'action_lastdaytopred+day_dis_max', 'action_day_std',
            'action_lastdaytopred+day_dis_min', 'last_1_launch_num',
            'launch_3_continuous_day_count', 'launch_day_dis_min',
            'last_5day_launch_rate', 'last_3day_launch_rate',
            'last_2_launch_num', 'launch_day_dis_max', 'page_4_count',
            'last_6_action_num', 'action_day_dis_mean',
            'last_7day_action_rate', 'video_day_mean',
            'launch_5_continuous_day_count', 'action_day_max',
            'action_day_dis_skew', 'video_day_min', 'page_type_count',
            'video_count', 'author_and_id_the_same', 'last_2_action_num',
            'action_type_3_count', 'last_7_action_num', 'last_5day_video_rate',
            'action_type_count', 'action_2_continuous_day_count',
            'last_5_action_num', 'launch_day_max', 'last_4_launch_num',
            'action_day_dis_max', 'action_day_count', 'action_day_dis_min',
            'day_video_mean', 'day_video_std', 'video_day_max',
            'launch_inpredday_count', 'last_video_day_before_pred',
            'action_3_continuous_day_count', 'action_5_continuous_day_count',
            'last_7day_launch_rate', 'last_5_launch_num',
            'last_3day_video_rate', 'last_7_launch_num', 'last_3_launch_num',
            'day_video_last', 'last_5_video_num', 'video_day_std',
            'video_day_dis_mean', 'last_7day_video_rate', 'video_day_dis_min',
            'action_inpredday_count', 'last_6_launch_num',
            'launch_7_continuous_day_count', 'day_video_skew',
            'last_4_video_num', 'last_3_video_num', 'last_7_video_num',
            'last_1_video_num', 'video_day_dis_max',
            'video_2_continuous_day_count']

label = 'label'

dtrain = lgb.Dataset(train[features], label=train[label])
dtest = lgb.Dataset(train[features], label=train[label])

model = lgb.train(params, dtrain, num_boost_round=750,
                  valid_sets=[dtrain, dtest],
                  verbose_eval=200,
                  )
pred = model.predict(test[features], num_iteration=model.best_iteration)
####model3_result
pred3 = pred

####加权融合
print("加权融合...")
test['prob'] = pred1*0.4+pred2*0.3+pred3*0.3
###取阈值大于等于0.384
submit = test[test['prob'] >= 0.384][['user_id']]
####再加上28，29，30号新注册的用户(按概率值排序，取前37个，为了不跟上面的submit重复，这里的新用户概率值都应该是小于0.384的)
a = test[test.register_day >= 28]
a = a[['user_id', 'register_day', 'prob']]
a = a.sort_values(by=['prob'], ascending=False)
b = a[a.prob < 0.384]

rw = b.head(37)[['user_id']]
result = pd.concat([submit, rw])
####产生提交结果
print("产生结果...")
if not os.path.exists("submit"):
    os.mkdir("submit")
result.to_csv("submit/submit.csv", index=False, header=None)
print("输出完成...")
print(" total use time:", datetime.datetime.now()-t1, " s")


