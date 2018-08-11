import datetime
import warnings
warnings.filterwarnings('ignore')
def get_inpred_count(temp,flag):
    count=0
    for i in [flag+'_lastdaytopred+day_dis_mean',flag+'_lastdaytopred+day_dis_max',flag+"_lastdaytopred+day_dis_min"]:
        if temp[i].values[0]>0 and temp[i].values[0]<=7:
            count+=1
    return count
def get_launch_feat(df,pred_firstday):
    temp=pd.DataFrame(index=range(1))
    temp['user_id']=df['user_id'].unique()[0]
    temp['launch_count']=len(df)
    temp['launch_day_mean']=df['dur_day'].mean()
    temp['launch_day_min']=df['dur_day'].min()
    temp['launch_day_max']=df['dur_day'].max()
    temp['launch_day_max-min']=temp['launch_day_max']-temp['launch_day_min']
    last_7days_len=len(df[df['dur_day']<=7])
    temp['last_7day_launch_rate']=last_7days_len/len(df)
    last_5days_len=len(df[df['dur_day']<=5])
    temp['last_5day_launch_rate']=last_5days_len/len(df)
    last_3days_len=len(df[df['dur_day']<=3])
    temp['last_3day_launch_rate']=last_3days_len/len(df)
    last_2days_len=len(df[df['dur_day']<=2])
    temp['last_2day_launch_rate']=last_2days_len/len(df)
    last_1days_len=len(df[df['dur_day']<=1])
    temp['last_1day_launch_rate']=last_1days_len/len(df)
    temp['last_3day_launch_sum']=temp['last_1day_launch_rate']+temp['last_2day_launch_rate']+temp['last_3day_launch_rate']
    temp['last_2day_launch_sum']=temp['last_1day_launch_rate']+temp['last_2day_launch_rate']
    temp['launch_day_std']=df['dur_day'].std()
    temp['launch_day_skew']=df['dur_day'].skew()
    temp['last_launch_day_before_pred']=pred_firstday-df['day'].values[-1]
    df_diff=df.diff(1)
    temp['launch_day_dis_max']=df_diff['day'].max()
    temp['launch_day_dis_min']=df_diff['day'].min()
    temp['launch_day_dis_mean']=df_diff['day'].mean()
    temp['launch_day_dis_std']=df_diff['day'].std()
    temp['launch_day_dis_skew']=df_diff['day'].skew()
     
    temp['launch_lastdaytopred+day_dis_mean']=0-temp['last_launch_day_before_pred'].values[0]+temp['launch_day_dis_mean'].values[0]
    
    temp['launch_lastdaytopred+day_dis_max']=0-temp['last_launch_day_before_pred'].values[0]+temp['launch_day_dis_max'].values[0]
    temp['launch_lastdaytopred+day_dis_min']=0-temp['last_launch_day_before_pred'].values[0]+temp['launch_day_dis_min'].values[0]
    
    temp["launch_inpredday_count"]=get_inpred_count(temp,"launch")
    temp['launch_sum']=0
    for i in range(pred_firstday-7,pred_firstday):
        if i in df['day'].values:
            temp['last_'+str(pred_firstday-i)+"_launch_num"]=1/(pred_firstday-i)
        else :
            temp['last_'+str(pred_firstday-i)+"_launch_num"]=0
        temp['launch_sum']+=temp['last_'+str(pred_firstday-i)+"_launch_num"]
    for i in [2,3,5,7]:
        temp["launch_"+str(i)+"_continuous_day_count"]=get_continuous_day_count(i,df)
    
    return temp


def get_continuous_day_count(cont_num,df):
    day_set=df['day'].unique()

    day_count=0
    for i in day_set:
        flag=0
        for j in range(1,cont_num):
            if i+j in day_set:
                flag+=1
       
        if flag==cont_num-1:
            day_count+=1
    
    
    return day_count
            
    
    
def get_video_feat(df,pred_firstday):
    temp=pd.DataFrame(index=range(1))
    temp['user_id']=df['user_id'].unique()[0]
    temp['video_count']=len(df)
    
    temp['day_video_mean']=df.groupby('day',as_index=False).count()['user_id'].mean()
    temp['day_video_std']=df.groupby('day',as_index=False).count()['user_id'].std()
    temp['day_video_max']=df.groupby('day',as_index=False).count()['user_id'].max()
    temp['day_video_min']=df.groupby('day',as_index=False).count()['user_id'].min()
    temp['day_video_skew']=df.groupby('day',as_index=False).count()['user_id'].skew()
    temp['day_video_last']=df.groupby('day',as_index=False).count()['user_id'].values[-1]
    b=dict(zip(df.groupby('day',as_index=False).count()['day'].values,df.groupby('day',as_index=False).count()['user_id'].values))
    temp['video_sum']=0
    for i in range(pred_firstday-7,pred_firstday):
        if i in b.keys():
            temp['last_'+str(pred_firstday-i)+"_video_num"]=b[i]/(pred_firstday-i)
        else :
            temp['last_'+str(pred_firstday-i)+"_video_num"]=0
        temp['video_sum']+=temp['last_'+str(pred_firstday-i)+"_video_num"]
    temp['video_day_count']=df.groupby('day',as_index=False).count()['user_id'].count()
    df_temp=df.groupby('day',as_index=False).count()
    df_temp['dur_day']=df_temp['day'].apply(lambda x:pred_firstday-x,1)
    
    temp['video_day_mean']=df_temp['dur_day'].mean()
    temp['video_day_min']=df_temp['dur_day'].min()
    temp['video_day_max']=df_temp['dur_day'].max()
    temp['video_day_max-min']=temp['video_day_max']-temp['video_day_min']
    last_7days_len=len(df_temp[df_temp['dur_day']<=7])
    temp['last_7day_video_rate']=last_7days_len/len(df_temp)
    
    last_5days_len=len(df_temp[df_temp['dur_day']<=5])
    temp['last_5day_video_rate']=last_5days_len/len(df)
    last_3days_len=len(df_temp[df_temp['dur_day']<=3])
    temp['last_3day_video_rate']=last_3days_len/len(df)
    last_2days_len=len(df_temp[df_temp['dur_day']<=2])
    temp['last_2day_video_rate']=last_2days_len/len(df)
    last_1days_len=len(df_temp[df_temp['dur_day']<=1])
    temp['last_1day_video_rate']=last_1days_len/len(df)
    
    temp['video_day_std']=df_temp['dur_day'].std()
    temp['video_day_skew']=df_temp['dur_day'].skew()
    temp['last_video_day_before_pred']=pred_firstday-df_temp['day'].values[-1]
    df_diff=df_temp.diff(1)
    temp['video_day_dis_max']=df_diff['day'].max()
    temp['video_day_dis_min']=df_diff['day'].min()
    temp['video_day_dis_mean']=df_diff['day'].mean()
    temp['video_day_dis_std']=df_diff['day'].std()
    temp['video_day_dis_skew']=df_diff['day'].skew()
    
    
    for i in [2,3,5,7]:
        temp["video_"+str(i)+"_continuous_day_count"]=get_continuous_day_count(i,df_temp)
    
    return temp
  
    
    
def get_action_feat(df,pred_firstday):
    temp=pd.DataFrame(index=range(1))
    temp['user_id']=df['user_id'].values[0]
    temp['action_count']=len(df)
    
    temp['day_action_mean']=df.groupby('day',as_index=False).count()['user_id'].mean()
    temp['day_action_std']=df.groupby('day',as_index=False).count()['user_id'].std()
    temp['day_action_max']=df.groupby('day',as_index=False).count()['user_id'].max()
    temp['day_action_min']=df.groupby('day',as_index=False).count()['user_id'].min()
    temp['day_action_skew']=df.groupby('day',as_index=False).count()['user_id'].skew()
    temp['day_action_last']=df.groupby('day',as_index=False).count()['user_id'].values[-1]
    b=dict(zip(df.groupby('day',as_index=False).count()['day'].values,df.groupby('day',as_index=False).count()['user_id'].values))
    temp['action_sum']=0
    for i in range(pred_firstday-7,pred_firstday):
        if i in b.keys():
            temp['last_'+str(pred_firstday-i)+"_action_num"]=b[i]/(pred_firstday-i)
        else :
            temp['last_'+str(pred_firstday-i)+"_action_num"]=0
        temp['action_sum']+=temp['last_'+str(pred_firstday-i)+"_action_num"]
    temp['action_day_count']=df.groupby('day',as_index=False).count()['user_id'].count()
    df_temp=df.groupby('day',as_index=False).count()
    df_temp['dur_day']=df_temp['day'].apply(lambda x:pred_firstday-x,1)

    
    temp['action_day_mean']=df_temp['dur_day'].mean()
    temp['action_day_min']=df_temp['dur_day'].min()
    temp['action_day_max']=df_temp['dur_day'].max()
    temp['action_day_max-min']=temp['action_day_max']-temp['action_day_min']
    last_7days_len=len(df_temp[df_temp['dur_day']<=7])
    temp['last_7day_action_rate']=last_7days_len/len(df_temp)
    
    last_5days_len=len(df_temp[df_temp['dur_day']<=5])
    temp['last_5day_action_rate']=last_5days_len/len(df)
    last_3days_len=len(df_temp[df_temp['dur_day']<=3])
    temp['last_3day_action_rate']=last_3days_len/len(df)
#     last_2days_len=len(df_temp[df_temp['dur_day']<=2])
#     temp['last_2day_action_rate']=last_2days_len/len(df)
#     last_1days_len=len(df_temp[df_temp['dur_day']<=1])
#     temp['last_1day_action_rate']=last_1days_len/len(df)
    
    temp['action_day_std']=df_temp['dur_day'].std()
    temp['action_day_skew']=df_temp['dur_day'].skew()
    temp['last_action_day_before_pred']=pred_firstday-df_temp['day'].values[-1]
    df_diff=df_temp.diff(1)
    temp['action_day_dis_max']=df_diff['day'].max()
    temp['action_day_dis_min']=df_diff['day'].min()
    temp['action_day_dis_mean']=df_diff['day'].mean()
    temp['action_day_dis_std']=df_diff['day'].std()
    temp['action_day_dis_skew']=df_diff['day'].skew()
    
    temp['action_lastdaytopred+day_dis_mean']=0-temp['last_action_day_before_pred'].values[0]+temp['action_day_dis_mean'].values[0]
    
    temp['action_lastdaytopred+day_dis_max']=0-temp['last_action_day_before_pred'].values[0]+temp['action_day_dis_max'].values[0]
    temp['action_lastdaytopred+day_dis_min']=0-temp['last_action_day_before_pred'].values[0]+temp['action_day_dis_min'].values[0]
    
    temp["action_inpredday_count"]=get_inpred_count(temp,"action")
    
    for i in [2,3,5,7]:
        temp["action_"+str(i)+"_continuous_day_count"]=get_continuous_day_count(i,df_temp)
        
    ###page
    
    temp['page_type_count']=len(df.page.unique())
    for i in range(5):
        temp['page_'+str(i)+"_count"]=len(df[df.page==i])/len(df)
    ###action_type
    temp['action_type_count']=len(df.action_type.unique())
    for i in range(6):
        temp['action_type_'+str(i)+"_count"]=len(df[df.action_type==i])/len(df)
        
    df['id_same']=df["user_id"]-df['author_id']
    a=len(df[df['id_same']==0])
    temp['author_and_id_the_same']=a/len(df)
    
    
    return temp   



def get_action_author_feat(action,register):
    author_temp=action.groupby(['author_id',"user_id"],as_index=False).count()
    author=author_temp.groupby("author_id",as_index=False).count()[['author_id','user_id']]
    author.rename(columns={'user_id':'author_count', 'author_id':'user_id', }, inplace = True)
    id_list=register[['user_id']]
    action_other_feat=pd.merge(id_list,author,on="user_id",how="left")
    
    star=action[action.action_type==2]
    star=star[['author_id','action_type']]
    star.columns=['user_id','followers']
    star_count=star.groupby("user_id",as_index=False).count()
    action_other_feat=pd.merge(action_other_feat,star_count,on="user_id",how="left")  
       
    
    return action_other_feat


def get_label(data,launch,pred_begin=None,pred_end=None):
    print("get label...")
    starttime = datetime.datetime.now()
    launch_data=launch[(launch.day>=pred_begin)&(launch.day<=pred_end)]
    target=launch_data.groupby("user_id",as_index=False).count()[['user_id']]
    target['label']=1
    train=pd.merge(data,target,on="user_id",how="left")
    train['label']=train['label'].fillna(0)
    
    print("use time:",datetime.datetime.now()-starttime )
   
    
    return train

def get_data(action,video,register,launch,register_lastday,train_begin,pred_firstday):
    register_data=register[register.register_day<=register_lastday]
    launch_data=launch[(launch.day>=train_begin)&(launch.day<=register_lastday)]
    video_data=video[(video.day>=train_begin)&(video.day<=register_lastday)]
    action_data = action[(action.day>=train_begin)&(action.day<=register_lastday)]
    
    register_data['register_to_predday']=register_data.apply(lambda x:pred_firstday-x['register_day'],1)
    
    print("process launch...")
    starttime = datetime.datetime.now()
    launch_data['dur_day']=launch_data['day'].apply(lambda x:pred_firstday-x,1)
#     launch_feat=launch_data.groupby("user_id").apply(lambda x:get_launch_feat(x,pred_firstday))
#     launch_feat.index=range(len(launch_feat))
    launch_feat=applyParallel(get_launch_feat,launch_data,pred_firstday)
    print("use time:",datetime.datetime.now()-starttime )
    
    print("process video...")
    starttime = datetime.datetime.now()
#     video_feat=video_data.groupby("user_id").apply(lambda x:get_video_feat(x,pred_firstday))
#     video_feat.index=range(len(video_feat))
    video_feat=applyParallel(get_video_feat,video_data,pred_firstday)
    print("use time:",datetime.datetime.now()-starttime )
    
    print("process action...")
    starttime = datetime.datetime.now()
    action_feat=action_data.groupby("user_id").apply(lambda x:get_action_feat(x,pred_firstday))
    action_feat.index=range(len(action_feat))
    # action_feat=applyParallel(get_action_feat,action_data,pred_firstday)
    print("use time:",datetime.datetime.now()-starttime )
    
    print("process action other...")
    starttime = datetime.datetime.now()
    action_other=get_action_author_feat(action_data,register_data)
    print("use time:",datetime.datetime.now()-starttime )
    
    ####merge
    
    data=pd.merge(register_data,launch_feat,on="user_id",how="left")
    data=pd.merge(data,video_feat,on="user_id",how="left")
    data=pd.merge(data,action_feat,on="user_id",how="left")
    data=pd.merge(data,action_other,on="user_id",how="left")
    
    
    
    return data
    