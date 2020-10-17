# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:36:45 2019

@author: Guo
"""
from category_encoders import *
"""
enc = TargetEncoder(cols=['gender'])

training_numeric_dataset = enc.fit_transform(X_train, y_train)
testing_numeric_dataset = enc.transform(X_test)

enc0 = LeaveOneOutEncoder(cols=[2])
training = pd.DataFrame(data=np.array([[1,2,0],[2,3,0],[3,4,1],[4,5,1]]))
testing = pd.DataFrame(data=np.array([[1,2],[2,3]]))

training_numeric_dataset = enc0.fit_transform(training[training.columns[:2]], training[training.columns[2]])
testing_numeric_dataset = enc0.transform(testing)
"""
import pandas as pd
import numpy as np
import random
import time,datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Times New Roman']  # 指定默认字体,FangSong
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
from sklearn.ensemble import (RandomTreesEmbedding,RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier,AdaBoostClassifier)
from sklearn.metrics import roc_auc_score,roc_curve,auc,accuracy_score,classification_report,confusion_matrix,f1_score,precision_score,recall_score
from sklearn.metrics import precision_recall_curve,average_precision_score
from lightgbm import LGBMClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")
from numpy import percentile

plt.rcParams['savefig.dpi']=300
plt.rcParams['figure.dpi'] = 300

#################################################################################################################    
fig_path='../20180814/sql_data/figure_path/'
data_path='../20180814/sql_data/data_path/'

demp_path = '../20180814/WEB数据/show_data/'
input_path='../20180814/sql_data/'
output_file='../20180814/sql_data/output/'
time0=time.time()
print("load eGFRInfor...")
eGFR=pd.read_csv(input_path+"201811_eGFR.csv")
eGFR=eGFR.fillna(0.01)
eGFR=eGFR.replace({'\\N':0.01})
eGFR['eGFR']=eGFR['eGFR'].astype(float)
eGFR=eGFR[eGFR['eGFR']>=60].reset_index(drop=True)

eGFR['医院编号_id']=eGFR['医院编号'].astype(str)+'_'+eGFR['id'].astype(str)
eGFR=eGFR[['医院编号_id','eGFR']]
#################################################################################################################
feature = pd.read_csv('total_feat.csv',encoding='gbk')

feature=feature.values.tolist()
from itertools import chain
feature=list(chain.from_iterable(feature))

total_feat = list(set(feature))
total_feat = [c.replace('Myocardial infarctionCerebrovascular disease','Any tumor') for c in total_feat]



total_feat_noSCr = [x for x in total_feat if 'SCr' not in x]
total_feat_noSCr = [c.replace('Myocardial infarctionCerebrovascular disease','Any tumor') for c in total_feat_noSCr]

#total_feat.append('KIDIGO_AKI')
#total_feat_noSCr.append('KIDIGO_AKI')

total_feat.append('医院编号_id')
total_feat_noSCr.append('医院编号_id')
#################################################################################################################
hour = 24#48#24 #48 #72
print("hour ",hour)
#################################################################################################################
print("load basicInfor...")
b_Info_A=pd.read_csv('df_basicInfor.csv',encoding='gbk')

b_Info_A['医院编号_id']=b_Info_A['医院编号'].astype(str)+'_'+b_Info_A['id'].astype(str)
b_eGFR_A=pd.merge(b_Info_A,eGFR,on=['医院编号_id'])

b_eGFR_A=b_eGFR_A.fillna(-1)
b_eGFR_A=b_eGFR_A.replace({'\\N':-1})

drop_list=['出院科室','住院天数','院内死亡','出院状态','结果值','基线日期','基线值','max_scr']#,'AKIStage','AKImaxStage'
residue_list=[col for col in b_eGFR_A if col not in drop_list]
b_eGFR_A=b_eGFR_A[residue_list]#134752
"""
b_eGFR_A.shape

(581894, 15)

X.shape

580461

"""
#b_eGFR_A[b_eGFR_A['入院日期']>b_eGFR_A['出院日期']].shape
print("b_eGFR_A shape ",b_eGFR_A.shape)
"""
b_eGFR_A_AKI = b_eGFR_A[b_eGFR_A['KIDIGO_AKI']==1]
b_eGFR_A_noAKI = b_eGFR_A[b_eGFR_A['KIDIGO_AKI']==0].reset_index(drop=True)
b_eGFR_A_AKI = b_eGFR_A_AKI[b_eGFR_A_AKI["检验日期"]>=b_eGFR_A_AKI["入院日期"]].reset_index(drop=True)
b_eGFR_A = pd.concat([b_eGFR_A_AKI,b_eGFR_A_noAKI],ignore_index=True)
"""
print("b_eGFR_A shape ",b_eGFR_A.shape)
"""
b_eGFR_A shape  (581894, 15)
b_eGFR_A shape  (536602, 15)
["检验日期"]< ["入院日期"](1182, 13)
"""
def time_choose(time_flag):
    if time_flag==24:
       b_eGFR_A['出院日期']=pd.to_datetime(b_eGFR_A['出院日期'])#-datetime.timedelta(1)
       b_eGFR_A['检验日期']=pd.to_datetime(b_eGFR_A['检验日期'])#-datetime.timedelta(1)
    elif time_flag==48:
       b_eGFR_A['出院日期']=pd.to_datetime(b_eGFR_A['出院日期'])-datetime.timedelta(1)
       b_eGFR_A['检验日期']=pd.to_datetime(b_eGFR_A['检验日期'])-datetime.timedelta(1)
    elif time_flag==72:
       b_eGFR_A['出院日期']=pd.to_datetime(b_eGFR_A['出院日期'])-datetime.timedelta(2)
       b_eGFR_A['检验日期']=pd.to_datetime(b_eGFR_A['检验日期'])-datetime.timedelta(2)
    return b_eGFR_A['出院日期'],b_eGFR_A['检验日期']
#b_eGFR_A.to_csv('b_eGFR_A_(48SCr).csv',encoding='gbk')

b_eGFR_A['出院日期'],b_eGFR_A['检验日期'] = time_choose(hour)

b_eGFR_A['入院日期']=pd.to_datetime(b_eGFR_A['入院日期'])
b_eGFR_A['出院日期']=pd.to_datetime(b_eGFR_A['出院日期'])
b_eGFR_A['检验日期']=pd.to_datetime(b_eGFR_A['检验日期'])
#################################################################################################################
input_ke_path=input_path+'已修改/'
ke1=pd.read_excel(input_ke_path+'科室-省医分类示例.xlsx')
ke2=pd.read_excel(input_ke_path+'2五华分科-已修改.xlsx')
ke3=pd.read_excel(input_ke_path+'3崇左-已修改.xlsx')
ke4=pd.read_excel(input_ke_path+'4喀什-已修改.xlsx')
ke5=pd.read_excel(input_ke_path+'5陆丰-已修改.xlsx')
ke6=pd.read_excel(input_ke_path+'6内蒙古-已修改.xlsx')
ke7=pd.read_excel(input_ke_path+'7重庆-已修改.xlsx')
ke8=pd.read_excel(input_ke_path+'9四川-已修改.xlsx')
ke9=pd.read_excel(input_ke_path+'10浙江-已修改.xlsx')
ke10=pd.read_excel(input_ke_path+'11上海九院-已修改.xlsx')
ke11=pd.read_excel(input_ke_path+'14安徽-已修改.xlsx')
ke12=pd.read_excel(input_ke_path+'15新疆-已修改.xlsx')
ke13=pd.read_excel(input_ke_path+'16呼和浩特-已修改.xlsx')
ke14=pd.read_excel(input_ke_path+'17吉林-已修改.xlsx')
ke15=pd.read_excel(input_ke_path+'21东莞-已修改.xlsx')
ke16=pd.read_excel(input_ke_path+'23万宁-已修改.xlsx')

ke3['二级学科1']=ke3['二级学科1'].fillna('中医其他')
ke4['二级学科1']=ke4['二级学科1'].fillna('中医其他')#中西医结合
ke6['二级学科1']=ke6['二级学科1'].fillna('中医其他')#中医学,中西医结合
ke7['二级学科1']=ke7['二级学科1'].fillna('中医其他')#中西医结合
ke8['二级学科1']=ke8['二级学科1'].fillna('中医其他')#城东病区
ke9['二级学科1']=ke9['二级学科1'].fillna('中医其他')#中医学,临床医学
ke12['二级学科1']=ke12['二级学科1'].fillna('中医其他')#中医科
ke13['二级学科1']=ke13['二级学科1'].fillna('中医其他')#中医科
ke14['二级学科1']=ke14['二级学科1'].fillna('中医其他')#中医科
ke15['二级学科1']=ke15['二级学科1'].fillna('中医其他')#中医康复科,中医科
ke15['二级学科1']=ke15['二级学科1'].fillna('中医其他')#中医科,五官科

keys1,vals1=ke1["科室"],ke1["二级学科1"]
ke1_dict=dict(zip(keys1,vals1))

keys2,vals2=ke2["科室"],ke2["二级学科1"]
ke2_dict=dict(zip(keys2,vals2))

keys3,vals3=ke3["科室"],ke3["二级学科1"]
ke3_dict=dict(zip(keys3,vals3))

keys4,vals4=ke4["科室"],ke4["二级学科1"]
ke4_dict=dict(zip(keys4,vals4))

keys5,vals5=ke5["科室"],ke5["二级学科1"]
ke5_dict=dict(zip(keys5,vals5))

keys6,vals6=ke6["科室"],ke6["二级学科1"]
ke6_dict=dict(zip(keys6,vals6))

keys7,vals7=ke7["科室"],ke7["二级学科1"]
ke7_dict=dict(zip(keys7,vals7))

keys8,vals8=ke8["科室"],ke8["二级学科1"]
ke8_dict=dict(zip(keys8,vals8))

keys9,vals9=ke9["科室"],ke9["二级学科1"]
ke9_dict=dict(zip(keys9,vals9))

keys10,vals10=ke10["科室"],ke10["二级学科1"]
ke10_dict=dict(zip(keys10,vals10))

keys11,vals11=ke11["科室"],ke11["二级学科1"]
ke11_dict=dict(zip(keys11,vals11))

keys12,vals12=ke12["科室"],ke12["二级学科1"]
ke12_dict=dict(zip(keys12,vals12))

keys13,vals13=ke13["科室"],ke13["二级学科1"]
ke13_dict=dict(zip(keys13,vals13))

keys14,vals14=ke14["科室"],ke14["二级学科1"]
ke14_dict=dict(zip(keys14,vals14))

keys15,vals15=ke15["科室"],ke15["二级学科1"]
ke15_dict=dict(zip(keys15,vals15))

keys16,vals16=ke16["科室"],ke16["二级学科1"]
ke16_dict=dict(zip(keys16,vals16))

dict_1=dict(ke1_dict,**ke2_dict)
dict_2=dict(ke3_dict,**ke4_dict)
dict_3=dict(ke5_dict,**ke6_dict)
dict_4=dict(ke7_dict,**ke8_dict)
dict_5=dict(ke9_dict,**ke10_dict)
dict_6=dict(ke11_dict,**ke12_dict)
dict_7=dict(ke13_dict,**ke14_dict)
dict_8=dict(ke15_dict,**ke16_dict)

dict_1=dict(dict_1,**dict_2)
dict_2=dict(dict_3,**dict_4)
dict_3=dict(dict_5,**dict_6)
dict_4=dict(dict_7,**dict_8)

dict_1=dict(dict_1,**dict_2)
dict_2=dict(dict_3,**dict_4)
dict_all=dict(dict_1,**dict_2)

del dict_1,dict_2,dict_3,dict_4,dict_5,dict_6,dict_7,dict_8

b_eGFR_A["department"]=b_eGFR_A["入院科室"].map(dict_all)
A=[2,3,5,23]
b_eGFR_A['医院等级']=b_eGFR_A['医院编号'].apply(lambda x:'三级' if x not in A else '二级')

###################################################################################################################
print("load testInfor...")#
test_Info_A=pd.read_pickle(output_file+'df_testInfor.pkl')

test_Info_A['医院编号_id'] = test_Info_A['医院编号'].astype(str) + '_' + test_Info_A['id'].astype(str)
test_Info_A['检验日期']=pd.to_datetime(test_Info_A['检验日期'])
test_Info_A=test_Info_A.rename(columns={'检验日期':'化验日期'})
test_bas_A=pd.merge(b_eGFR_A,test_Info_A,how='left', on=['医院编号_id','id','医院编号'])#(12152293, 17)

test_bas_A_0=test_bas_A[test_bas_A['KIDIGO_AKI']==0]
test_bas_A_1=test_bas_A[test_bas_A['KIDIGO_AKI']==1]

test_bas_A_0=test_bas_A_0[(test_bas_A_0['化验日期']<test_bas_A_0['出院日期'])&(test_bas_A_0['化验日期']>=test_bas_A_0['入院日期'])]
test_bas_A_1=test_bas_A_1[(test_bas_A_1['化验日期']<test_bas_A_1['检验日期'])&(test_bas_A_1['化验日期']>=test_bas_A_1['入院日期'])]
test_bas_A_date=pd.concat([test_bas_A_0,test_bas_A_1],ignore_index=True)#(10492117, 17)(id= 479985)

test_bas_A_date['标本类型_指标']=test_bas_A_date['标本类型'].astype('str')+'_'+test_bas_A_date['指标'].astype('str')

##################################将所有'\N'的样本填充为’血清‘#################################################################################
test_bas_A_date['标本类型'] = test_bas_A_date['标本类型'].replace({'\\N':'血清'})

#\N715739


test_Info_A[test_Info_A['医院编号_id']=='3_49451']
test_bas_A_date[test_bas_A_date['医院编号_id']=='3_49451']
b_eGFR_A[b_eGFR_A['医院编号_id']=='3_49451'][['入院日期','出院日期','检验日期']]

A = test_bas_A_1[(test_bas_A_1['检验日期']<test_bas_A_1['入院日期'])]#&(b_eGFR_A['化验日期']>b_eGFR_A['入院日期'])]
A[A['医院编号_id']=='14_139470']

###################################################################################################################
C1=test_bas_A_date[test_bas_A_date['标本类型'].str.contains('血')]#.reset_index(drop=True)#(id=405000),全血|静脉血|动脉血|血常规|空腹血|血浆|血清|血液|血生化|血_
C0=test_bas_A_date[test_bas_A_date['标本类型'].str.contains('手|餐|脐|抗凝')]
test1=list(C1.标本类型)
test2=list(C0.标本类型)
ret=list(set(test1)-set(test2))#['血浆', '血清', '血液', '血', '静脉血', '空腹血清', '血生化', '动脉血', '全血', '空腹血', '血常规']
test_bas_A_date=test_bas_A_date[test_bas_A_date.标本类型.isin(ret)].reset_index(drop=True)#(11490477, 18)

test_bas_A_date=test_bas_A_date[test_bas_A_date['结果值']>=0].reset_index(drop=True)#(11490020, 18)

old_list=['二氧化碳结合率','血尿酸','尿酸(UA)','白蛋白(ALB)','红细胞比积','血小板计数']
new_list=['二氧化碳结合力','尿酸','尿酸','白蛋白','红细胞压积','血小板']

test_bas_A_date['指标']=test_bas_A_date['指标'].replace(old_list,new_list)
test_bas_A_date['指标']=test_bas_A_date['指标'].replace(['血清氯','氯(CL)','氯(CL-)'],'氯')
test_bas_A_date['指标']=test_bas_A_date['指标'].replace(['血清钠','钠(Na)','钠(Na+)'],'钠')
test_bas_A_date['指标']=test_bas_A_date['指标'].replace(['血清钾','血清钾(K+)','钾离子'],'钾')
test_bas_A_date['指标']=test_bas_A_date['指标'].replace(['血红蛋白浓度','血红蛋白含量'],'血红蛋白')

old_list=['二氧化碳结合力','尿酸','白蛋白','红细胞压积','血红蛋白','血小板','钾','钠','氯']
new_list=['CO2CP','UA','ALB','HCT','HGB','PLT','K','NA','CL']
test_bas_A_date['指标']=test_bas_A_date['指标'].replace(old_list,new_list)

#category=['钾','钠','氯','白蛋白','尿酸','红细胞压积','二氧化碳结合力','血小板计数','血红蛋白']
category=['K','NA','CL','ALB','UA','HCT','CO2CP','PLT','HGB']

test_bas_A_date=test_bas_A_date[test_bas_A_date['指标'].isin(category)].reset_index(drop=True)

test_bas_A_date['标本类型_指标']=test_bas_A_date['标本类型']+'_'+test_bas_A_date['指标']

test_d=test_bas_A_date.groupby(['医院编号_id','标本类型_指标']).size().reset_index().rename(columns={0:'times'})#标本类型_指标
test_d1=test_d.groupby(['医院编号_id'])['标本类型_指标'].count().reset_index().rename(columns={'标本类型_指标':'做的检测指标种类'})#标本类型_指标

zhibiao_col=test_bas_A_date.groupby(['指标']).size()
biaoben_col=test_bas_A_date.groupby(['标本类型']).size()
test_df_feat=pd.DataFrame()
test_df_feat['医院编号_id']=test_d1['医院编号_id']

for i in category:
    print(i)
    test_df_category=test_bas_A_date[test_bas_A_date['指标']==i]#标本类型_指标

    #test_feat=pd.DataFrame(list(test_df_category['医院编号_id'].drop_duplicates()),columns={'医院编号_id'})

    df_test_sort=test_df_category.sort_values(by=['医院编号_id','化验日期'])
    '''
    first_test_value=df_test_sort.drop_duplicates(['医院编号_id'],keep='first')['结果值'].reset_index(drop=True)
    last_test_value=df_test_sort.drop_duplicates(['医院编号_id'],keep='last')['结果值'].reset_index(drop=True)
    test_feat['first '+i]= first_test_value#i+'_first_value'
    test_feat['last '+i]= last_test_value'''#i+'_last_value'

    first_test_value = df_test_sort.drop_duplicates(['医院编号_id'],keep='first')[['医院编号_id','结果值']].reset_index(
            drop=True).rename(columns={"结果值": 'first '+i})#i + "_first_value"
    last_test_value = df_test_sort.drop_duplicates(["医院编号_id"], keep="last")[["医院编号_id", "结果值"]].reset_index(
            drop=True).rename(columns={"结果值":'last '+i})             # i + "_last_value"
    test_feat = pd.merge(first_test_value, last_test_value, on=["医院编号_id"])

    test_cnt=test_df_category.groupby(['医院编号_id'])['结果值'].agg('count').reset_index().rename(
            columns={'结果值':str(i)+' test times'})
    test_min=test_df_category.groupby(['医院编号_id'])['结果值'].agg('min').reset_index().rename(
            columns={'结果值':'minimum '+str(i)})#str(i)+'_min'
    test_max=test_df_category.groupby(['医院编号_id'])['结果值'].agg('max').reset_index().rename(
            columns={'结果值':'maximum '+str(i)})#str(i)+'_max'
    test_mean=test_df_category.groupby(['医院编号_id'])['结果值'].agg('mean').reset_index().rename(
            columns={'结果值':'mean '+str(i)})#str(i)+'_mean'
    test_std=test_df_category.groupby(['医院编号_id'])['结果值'].agg('std').reset_index().rename(
            columns={'结果值':'standard deviation '+str(i)})#str(i)+'_std'

    test_feat=pd.merge(test_feat,test_cnt,on=['医院编号_id'],how='left')
    test_feat=pd.merge(test_feat,test_min,on=['医院编号_id'],how='left')
    test_feat=pd.merge(test_feat,test_max,on=['医院编号_id'],how='left')
    test_feat=pd.merge(test_feat,test_mean,on=['医院编号_id'],how='left')
    test_feat=pd.merge(test_feat,test_std,on=['医院编号_id'],how='left')

    test_df_feat=pd.merge(test_df_feat,test_feat,on=['医院编号_id'],how='left')#(491633, 64)

test_df_des=test_df_feat.describe()
df_Info_0=pd.merge(b_eGFR_A,test_df_feat,on=['医院编号_id'],how='left')#add test info
df_Info_0=df_Info_0.fillna(-1)#缺失的检测指标填-1
df_Info_0_des=df_Info_0.describe()

###################################################################################################################
print("load scrInfor...")
scr_Info_A=pd.read_csv(input_path+'df_scrInfor.csv')#(3639878, 6)
scr_Info_A=scr_Info_A.rename(columns={'检验日期':'SCr日期','结果值':'SCr值'})

scr_Info_A=scr_Info_A.drop(['标本类型','指标'],axis=1)
scr_Info_A['医院编号_id']=scr_Info_A['医院编号'].astype(str) + '_' +scr_Info_A['id'].astype(str)
SCr_bas_A=pd.merge(b_eGFR_A,scr_Info_A,on=['医院编号_id','id','医院编号'],how='left')

SCr_bas_A['SCr日期']=pd.to_datetime(SCr_bas_A['SCr日期'])

SCr_bas_A_0=SCr_bas_A[SCr_bas_A['KIDIGO_AKI']==0]
SCr_bas_A_1=SCr_bas_A[SCr_bas_A['KIDIGO_AKI']==1]

SCr_bas_A_0=SCr_bas_A_0[(SCr_bas_A_0['SCr日期']>=SCr_bas_A_0['入院日期'])&(SCr_bas_A_0['SCr日期']<SCr_bas_A_0['出院日期'])]
SCr_bas_A_1=SCr_bas_A_1[(SCr_bas_A_1['SCr日期']>=SCr_bas_A_1['入院日期'])&(SCr_bas_A_1['SCr日期']<SCr_bas_A_1['检验日期'])]
SCr_bas_A_date=pd.concat([SCr_bas_A_0,SCr_bas_A_1],ignore_index=True)

SCr_date_sort=SCr_bas_A_date.sort_values(by=['医院编号_id','SCr日期'])
first_SCr_value=SCr_date_sort.drop_duplicates(['医院编号_id'],keep='first')['SCr值']#.rename(columns={'first_SCr值':'SCr值'})#.reset_index()
last_SCr_value=SCr_date_sort.drop_duplicates(['医院编号_id'],keep='last')['SCr值']#.rename(columns={'SCr值':'last_SCr值'}).reset_index()
id_num=list(SCr_date_sort['医院编号_id'].drop_duplicates())#.reset_index()

SCr=pd.DataFrame(list(zip(id_num,first_SCr_value,last_SCr_value)))
SCr=SCr.rename(columns={0:'医院编号_id',1:'first SCr',2:'last SCr'})#'first_SCr值',2:'last_SCr值'

df_Info_1=pd.merge(df_Info_0,SCr,on=['医院编号_id'],how='left')
df_Info_1=df_Info_1.fillna(-1)
df_Info_1=df_Info_1[(df_Info_1['first SCr']<2000)]#first_SCr值
df_Info_1=df_Info_1[(df_Info_1['first SCr']!=-1.076600)&(df_Info_1['first SCr']!=-0.101400)].reset_index(drop=True)#排除scr>2000的4个，以及为负的两个不正常指标

SCr_min=SCr_bas_A_date.groupby(['医院编号_id'])['SCr值'].agg('min').reset_index().rename(
        columns={'SCr值':'minimum SCr'})#'SCr'+'_min'
SCr_max=SCr_bas_A_date.groupby(['医院编号_id'])['SCr值'].agg('max').reset_index().rename(
        columns={'SCr值':'maximum SCr'})
SCr_mean=SCr_bas_A_date.groupby(['医院编号_id'])['SCr值'].agg('mean').reset_index().rename(
        columns={'SCr值':'mean SCr'})
SCr_std=SCr_bas_A_date.groupby(['医院编号_id'])['SCr值'].agg('std').reset_index().rename(
        columns={'SCr值':'standard deviation SCr'})

"""
add SCr_cnt
"""
SCr_cnt=SCr_bas_A_date.groupby(['医院编号_id'])['SCr值'].agg('count').reset_index().rename(
        columns={'SCr值':'SCr test times'})

SCr_Info=pd.merge(SCr_min,SCr_max,on=['医院编号_id'],how='left')
SCr_Info=pd.merge(SCr_Info,SCr_mean,on=['医院编号_id'],how='left')
SCr_Info=pd.merge(SCr_Info,SCr_std,on=['医院编号_id'],how='left')#shape=(566489, 5),df_Info_1.shape=(566484, 78)
SCr_Info=pd.merge(SCr_Info,SCr_cnt,on=['医院编号_id'],how='left')#shape=(566489, 5),df_Info_1.shape=(566484, 78)

df_Info_2=pd.merge(df_Info_1,SCr_Info,on=['医院编号_id'],how='left')#(566484, 82)
###################################################################################################################
print("load medicine Infor...")
advice_Info_A=pd.read_csv(input_path+'df_adviceInfor.csv')#(6025678, 7),id=560472
advice_Info_A=advice_Info_A.drop_duplicates().reset_index(drop=True)#.shape(4488844, 7)

advice_Info_A['开嘱日期'] =pd.to_datetime(advice_Info_A['开嘱日期'] )
advice_Info_A['停嘱日期'] =pd.to_datetime(advice_Info_A['停嘱日期'] )
advice_Info_A['医院编号_id'] = advice_Info_A['医院编号'].astype(str) + '_' + advice_Info_A['id'].astype(str)#

advice_Info_A=advice_Info_A[advice_Info_A['药品分类编码']!='\\N']#(5550911, 8)
#advice_bas_A=pd.merge(b_eGFR_A,advice_Info_A, how='left', on=['医院编号_id','id','医院编号'])#(4910717, 18)
advice_bas_A=pd.merge(b_eGFR_A,advice_Info_A, on=['医院编号_id','id','医院编号'])#(3495305, 18),id=490133
#486109
advice_bas_A=advice_bas_A[advice_bas_A['停嘱日期']>=advice_bas_A['入院日期']].reset_index(drop=True)#.shape=(2727318, 19)

advice_bas_A00=advice_bas_A[advice_bas_A['停嘱日期']<advice_bas_A['出院日期']]
advice_bas_A11=advice_bas_A[advice_bas_A['停嘱日期']>=advice_bas_A['出院日期']]

advice_bas_A00['day_gap']=advice_bas_A00['停嘱日期']
advice_bas_A11['day_gap']=advice_bas_A11['出院日期']
advice_bas_A=pd.concat([advice_bas_A11,advice_bas_A00],ignore_index=True)

advice_bas_A0=advice_bas_A[advice_bas_A['开嘱日期']>=advice_bas_A['入院日期']]
advice_bas_A1=advice_bas_A[advice_bas_A['开嘱日期']<advice_bas_A['入院日期']]#需特殊处理的药物

advice_bas_A0['day_start']=advice_bas_A0['开嘱日期']
advice_bas_A1['day_start']=advice_bas_A1['入院日期']
advice_bas_A=pd.concat([advice_bas_A0,advice_bas_A1],ignore_index=True)

advice_bas_A_0=advice_bas_A[advice_bas_A['KIDIGO_AKI']==0].reset_index(drop=True)#
advice_bas_A_1=advice_bas_A[advice_bas_A['KIDIGO_AKI']==1].reset_index(drop=True)#

A=advice_bas_A_1[advice_bas_A_1['检验日期']>=advice_bas_A_1['day_start']].reset_index(drop=True)
A1=A[A['检验日期']<A['day_gap']]
A0=A[A['检验日期']>=A['day_gap']]
A1['day_gap']=A1['检验日期']
A0['day_gap']=A0['day_gap']
A=pd.concat([A0,A1],ignore_index=True)

advice_bas_A=pd.concat([advice_bas_A_0,A],ignore_index=True)#id=472711,shape=(2514731, 20)

B1=advice_bas_A.groupby(['医院编号_id','药品分类编码']).size().reset_index().rename(columns={0:'times'})
med_count=B1.groupby(['医院编号_id'])['药品分类编码'].count().reset_index().rename(columns={'药品分类编码':'The number of potential nephrotoxic drugs used'})
#advice_bas_A['day_gap'].dt.strftime('%Y-%m-%d')

advice_bas_A['day_gap']=pd.to_datetime(advice_bas_A['day_gap']).dt.date#strftime('%Y-%m-%d')
advice_bas_A['day_start']=pd.to_datetime(advice_bas_A['day_start']).dt.date#strftime('%Y-%m-%d')

advice_bas_A["使用天数"]=(advice_bas_A['day_gap']-advice_bas_A['day_start']).dt.days+1#
advice_bas_A=advice_bas_A[advice_bas_A["使用天数"]>0].reset_index(drop=True)#(2514731, 21)=>(2506651, 21)

###########################################################################################################
advice_bas_A_simple=advice_bas_A[['医院编号_id','药品分类编码',"使用天数"]]
med_sum=advice_bas_A_simple.groupby(['医院编号_id','药品分类编码'])["使用天数"].sum().reset_index()

advice_excel=pd.read_excel(input_path+'药品分类-陈源汉核对20181121-1218.xlsx')
advice_excel=advice_excel[advice_excel["重新分类"]!=0].reset_index(drop=True)
advice_dict=dict(zip(advice_excel.编号,advice_excel.重新分类))

advice_excel['编号']=advice_excel['编号'].astype(str)

med_list=list(advice_excel['编号'])

df_feature=pd.DataFrame()
df_feature_II=pd.DataFrame()
df_feature['医院编号_id']=med_count['医院编号_id']
df_feature_II['医院编号_id']=med_count['医院编号_id']

j=0#j=27,50种药物中一共有21种药物使用人次超过10000，对这21种药物进行排列分析，是否使用，使用次数
for i in med_list:
    t=med_sum[med_sum['药品分类编码']==i]
    med_feat=t.rename(columns={'使用天数':'第{}种药使用天数'.format(str(i))}).reset_index(drop=True)
    cols=[c for c in med_feat.columns if c not in ['药品分类编码']]
    print(i)
    df_feature=pd.merge(df_feature,med_feat[cols],on=['医院编号_id'],how='left')

df_feature=df_feature.fillna(0)
df_feature_Des=df_feature.describe()

for i in advice_excel['编号']:
    df_feature['第{}种药是否使用'.format(i)]=df_feature['第{}种药使用天数'.format(i)]

for col in df_feature.columns[-50:]:
    df_feature[col]=df_feature[col].apply(lambda x:1 if x!=0 else 0)

df_feature=pd.merge(df_feature,med_count,on=['医院编号_id'],how='left')

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

anti_list = get_key (advice_dict, '抗感染药')
urine_list= get_key (advice_dict, '利尿剂')
blood_list= get_key (advice_dict, '血管活性药物')
RASI_list= get_key (advice_dict, 'RASI')
I_list= get_key (advice_dict, '碘造影剂')
zhixue_list=get_key (advice_dict, '止血药')

anti_infect=[]
for i in anti_list:
    anti_infect.append('第{}种药使用天数'.format(i))
df_feature['the days of use of anti-infective drugs']=df_feature[anti_infect].sum(axis=1)#按列相加

urine=[]
for i in urine_list:
    urine.append('第{}种药使用天数'.format(i))
df_feature['the days of use of  diuretic drugs']=df_feature[urine].sum(axis=1)#按列相加

blood=[]
for i in blood_list:
    blood.append('第{}种药使用天数'.format(i))
df_feature['the days of use of vasoactive drugs']=df_feature[blood].sum(axis=1)#血管活性药物天数

RASI=[]
for i in RASI_list:
    RASI.append('第{}种药使用天数'.format(i))
df_feature['the days of use of RASI']=df_feature[RASI].sum(axis=1)#按行相加，RASI使用天数

I=[]
for i in I_list:
    I.append('第{}种药使用天数'.format(i))
df_feature['the days of use of iodinated CMs']=df_feature[I].sum(axis=1)#按行相加，碘造影剂使用天数

zhixue=[]
for i in zhixue_list:
    zhixue.append('第{}种药使用天数'.format(i))
df_feature['the days of use of hemostatic drugs']=df_feature[zhixue].sum(axis=1)#按行相加，止血药使用天数

df_feature['the days of use of chemotherapy drugs']=df_feature['第10-1种药使用天数']#化疗药
df_feature['the days of use of NSAIDs']=df_feature['第8-2种药使用天数']#NSAIDs使用天数
df_feature['the days of use of CNI']=df_feature['第9种药使用天数']#CIN免疫抑制剂使用天数
df_feature['the days of use of hypophysin']=df_feature['第20-5种药使用天数']#垂体后叶素使用天数

med_feat=[ '抗感染药','利尿剂','血管活性药物','RASI','碘造影剂','止血药',
          '化疗药','NSAIDs','CIN免疫抑制剂', '垂体后叶素']
for i,j in zip(df_feature.columns[-10:],med_feat):
    print(i,j)
    df_feature['是否使用{}'.format(j)]=df_feature[i].apply(lambda x: 1 if x!=0 else 0)

#################################################################################################################

print("load icdInfor...")
ICD_Info_A=pd.read_csv(input_path+'df_icdInfor.csv')

ICD_Info_A=ICD_Info_A[ICD_Info_A['id']!='腰骶椎间盘突出'].reset_index()
ICD_Info_A['医院编号']=ICD_Info_A['医院编号'].astype(int)#ci此处的医院编号是float型，后面merge产生问题
ICD_Info_A['医院编号_id']=ICD_Info_A['医院编号'].astype(str)+'_'+ICD_Info_A['id'].astype(str)
#ICD_Info_A=ICD_Info_A.drop(['医院编号','id'],axis=1)

ICD_Info_A['诊断']=ICD_Info_A['诊断'].astype('str')
ICD_Info_A['charlson相关编码']=ICD_Info_A['charlson相关编码'].replace({-999:0})

ICD_Info_A[ICD_Info_A.columns[:19]]=ICD_Info_A[ICD_Info_A.columns[:19]].astype(int)
ICD_Info_A[ICD_Info_A.columns[:19]]=ICD_Info_A[ICD_Info_A.columns[:19]].replace({-999:0})

disease_icd=ICD_Info_A.groupby(['医院编号_id'])['1'].count().reset_index().rename(columns={'1':'疾病诊断种类'})

def icd_features(ICD_Info_A,charlson_item):
    icd_i_max=ICD_Info_A.groupby(['医院编号_id'])[charlson_item].max().reset_index()
    return icd_i_max

ICD_Info_A.groupby(['医院编号_id'])['2'].max().reset_index()

icd_feat = pd.DataFrame(list(df_feature['医院编号_id']),columns={'医院编号_id'}) # df_ICD_Info

for i in range(1,20):
    icd_i_max=icd_features(ICD_Info_A,str(i))
    print(i)
    icd_feat=pd.merge(icd_feat,icd_i_max,on=['医院编号_id'],how='left')

icd_feat=icd_feat.drop('12',axis=1)#删除之前保存需要检验的数据

score = [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 6, 6]#去掉一个2
def calculate_charlson(icd_feat,score):
    icd_feat['charlson_total_score'] = 0
    for i in [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18]:
        icd_feat[str(i)]=icd_feat[str(i)].apply(lambda x: x * score[i - 1] if (x!=-1) else 0)# if x!=-1 else 0

        icd_feat['charlson_total_score'] = icd_feat['charlson_total_score'] + icd_feat[str(i)]
        print(i)
    return icd_feat['charlson_total_score']

icd_feat['charlson_total_score']=calculate_charlson(icd_feat, score)
icd_feat['charlson_total_score']=icd_feat['charlson_total_score'].replace(-999,-1)
icd_feat_des=icd_feat.describe()
#######################################################################################################

df_Info_2.to_csv(data_path+"{}df_Info_2.csv".format(hour),index=None,encoding='gbk')
df_Info_0.to_csv(data_path+"{}df_Info_0.csv".format(hour),index=None,encoding='gbk')
icd_feat.to_csv(data_path+"{}icd_feat.csv".format(hour),index=None,encoding='gbk')
df_feature.to_csv(data_path+"{}df_feature.csv".format(hour),index=None,encoding='gbk')
"""
df_Info_2 = pd.read_csv(data_path+"df_Info_2.csv",encoding='gbk')
df_Info_0 = pd.read_csv(data_path+"df_Info_0.csv",encoding='gbk')
icd_feat = pd.read_csv(data_path+"icd_feat.csv",encoding='gbk')
df_feature = pd.read_csv(data_path+"df_feature.csv",encoding='gbk')
"""

def scr_select(flag_scr):
    if flag_scr==1:
        Info=pd.merge(df_Info_2,df_feature, how='left', on=['医院编号_id'])
    elif flag_scr==0:
        Info=pd.merge(df_Info_0,df_feature, how='left', on=['医院编号_id'])
    drop_col=df_feature.columns[1:101]
    Info=Info.drop(drop_col,axis=1)

    Info=Info[(Info['入院科室']!=-1)&(Info['入院科室']!='2000')&(Info['入院科室']!='2018')].reset_index(drop=True)
    Info=Info[(Info['department']!=-1)].reset_index(drop=True)

    Info=Info[Info['age']!=-1].reset_index(drop=True)#
    Info=Info[(Info['性别']!=-1)&(Info['性别']!='未知')].reset_index(drop=True)
    Info['性别']=Info['性别'].apply(lambda x:x.strip()).reset_index(drop=True)

    Info['department']=Info['department'].apply(lambda x:x.strip('？')).reset_index(drop=True)
    '''dict1={'运动医学':'康复医学与理疗学','中医五官科':'中医其他','中医内科':'中医其他','中医外科':'中医其他','民族医学':'中医其他'}
    Info['department']=Info['department'].map(dict1)'''

    Info['department']=Info['department'].apply(lambda x:'康复医学与理疗学'if x=='运动医学' else x)
    Info['department']=Info['department'].apply(lambda x:'中医其他'if (x=='中医五官科')|(x=='中医内科')|(x=='中医外科')|(x=='民族医学') else x)
    #Info1=Info.replace(-1,np.nan)
    Info_2=pd.merge(Info,icd_feat,on=['医院编号_id'],how='left')#add ICD inexplainer_rf = shap.TreeExplainer(my_model)

    return Info_2



def select_feature(data):
    lbl=LabelEncoder()
    data['性别']=lbl.fit_transform(data['性别'])
    data['department'] = lbl.fit_transform(data['department'])
    #data['入院科室'] = lbl.fit_transform(data['入院科室'])#department
    data['医院等级'] = lbl.fit_transform(data['医院等级'])#department

    data["出院日期"]=pd.to_datetime(data["出院日期"]).dt.date
    data["入院日期"]=pd.to_datetime(data["入院日期"]).dt.date
    data["检验日期"]=pd.to_datetime(data["检验日期"]).dt.date
    data_0=data[data['KIDIGO_AKI']==0]
    data_1=data[data['KIDIGO_AKI']==1]

    data_0["住院天数"]=(data_0["出院日期"]-data_0["入院日期"]).dt.days+1
    data_1["住院天数"]=(data_1["检验日期"]-data_1["入院日期"]).dt.days+1
    data=pd.concat([data_0,data_1],ignore_index=True)
    data=data[(data["住院天数"]<32)&(data["住院天数"]>=0)].reset_index(drop=True)#529个id住院天数>31

    feature=[i for i in data.columns if i not in ['eGFR','AKIStage','AKImaxStage',
                                                  #"入院日期",
                                                  #"出院日期","检验日期",
#!!!!!                                                  #'department',
                                                    #"医院编号","KIDIGO_AKI"
                                                  #'1', '2', '3', '4', '5', '6', '7', '8',
                                                  #'9', '10', '11','12', '13', '14', '15',
                                                  #'16', '17', '18', '19',
                                                  #"医院编号_id",'入院科室','department',"id",
            ]]
    X=data[feature]

    X['age']=X['age'].astype(int)
    X=X.rename(columns={'性别':"gender","department":"hospital department","charlson_total_score":"Charlson total score","住院天数":"length of stay","医院等级":"hospital grade"})
    '''
    X = rank(X,['length of stay','age'],'age',rank_name='age_LOS')
    
    X = rank(X,['length of stay','age'],'length of stay',rank_name='LOS_age')
    
    X = rank(X,['length of stay','age'],'hospital department',rank_name='LOS_age_HP')
    
    X = rank(X,['length of stay','the days of use of anti-infective drugs'],'length of stay',rank_name='LOS_drug')
    
    X = rank(X,['minimum SCr','maximum SCr'],'maximum SCr',rank_name='Max_min_SCr')
    
    X = rank(X,['minimum SCr','maximum SCr'],'minimum SCr',rank_name='Min_max_SCr')'''



    return X

def train_test_select(flag_scr,top):
    Info_2 = scr_select(flag_scr)
    X = select_feature(Info_2)

    #X = X.fillna(-1)
    """
    X_des = X.describe()
    X = X.fillna(-1)
    X = X.replace(-1,np.nan)
    X['null count'] = X.isnull().sum(axis=1)
    """

    col = X.columns
    col1 = list(col)
    #col1 = [x.replace('检测次数',' test times') for x in col1]
    old_col = [ '是否使用抗感染药', '是否使用利尿剂', '是否使用血管活性药物', '是否使用RASI', '是否使用碘造影剂', '是否使用止血药', '是否使用化疗药', '是否使用NSAIDs', '是否使用CIN免疫抑制剂', '是否使用垂体后叶素']
    new_col = [ 'Antibiotic used', 'Diuretic used', 'Vasoactive used', 'RASI used', 'Iodinated contrast medium used','Hemostatic used', 'Chemotherapy used', 'NSAIDs used', 'Calcineurin inhibitor(CNI) used','Hypophysin used']
    for i in range(len(new_col)):
        col1 = [x.replace(old_col[i],new_col[i]) for x in col1]

    charlson = pd.read_excel('charlson.xlsx')
    charlson[1] = charlson[1].astype(str)

    k, v =charlson[1],charlson[2]

    charlson_dict = dict(zip(k,v))
    '''
    for k,v in charlson_dict.items():
        print(k,v)
        
        col1[-20:-11] = [x.replace(k,v) for x in col1[-20:-11]]
    
    col1[-11:-2]  = ['Diabetes','Hemiplegia','Diabetes with end-organ damage','Any tumor','Leukemia','Lymphoma','Moderate or severe liver disease','Metastatic solid tumor','AIDS']
    '''
    new_charlson = []
    for c in col1[-20:-2]:
        for k,v in charlson_dict.items():
            if c==k:
                c=v
                print(c)
                new_charlson.append(c)

    col1[-20:-2] =  new_charlson

    X.columns = col1

    if top == 0 :
        drop_list=['入院科室',"id",'KIDIGO_AKI']#"医院编号_id",'医院编号',
        col = [c for c in X.columns if c not in drop_list]

        X_,y_ = X[col],X['KIDIGO_AKI']

    elif top == 1 :
        if flag_scr == 1:

            X_,y_ = X[total_feat],X['KIDIGO_AKI']

        elif flag_scr == 0:

            X_,y_ = X[total_feat_noSCr],X['KIDIGO_AKI']

    X_train, X_test, y_train, y_truth = train_test_split(X_,y_, test_size=0.2, shuffle=True, random_state=111, stratify=y_)#[X.columns[:1]],

    return X_train, X_test, y_train, y_truth

def predict_(flag_scr,top):
    X_train, X_test, y_train, y_truth = train_test_select(flag_scr,top)

    if flag_scr == 1:
        if hour == 24:
            clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 2})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.2,class_weight={0: 1, 1: 3})
        elif hour == 72:
            clf2=lgb.LGBMClassifier(learning_rate=0.1,class_weight={0: 1, 1: 2})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.25,class_weight={0: 1, 1: 3})
        elif hour == 48:
            clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 2})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 3})
    elif flag_scr == 0:
        if hour == 24:
            clf2=lgb.LGBMClassifier(learning_rate=0.25,class_weight={0: 1, 1: 3})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 3})
        elif hour == 72:
            clf2=lgb.LGBMClassifier(learning_rate=0.1,class_weight={0: 1, 1: 3})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 3})
        elif hour == 48:
            clf2=lgb.LGBMClassifier(learning_rate=0.2,class_weight={0: 1, 1: 3})#无论top为0/1均是最优解
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.2,class_weight={0: 1, 1: 3})

    categorical_feature = ['gender','hospital department','hospital grade']#入院科室
    if top == 1:
        categorical_feature = ['hospital department','hospital grade']

    col = [c for c in X_train.columns if c not in ['SCr test times','医院编号','医院编号_id','KIDIGO_AKI','出院日期','检验日期',"入院日期"]]
    clf2.fit(X_train[col], y_train,categorical_feature=categorical_feature)

    y_pre2=clf2.predict(X_test[col])
    y_prob=clf2.predict_proba(X_test[col])[:,1]

    #print("sam predict ",clf.predict_proba(sam[col]))

    y_display = pd.DataFrame(X_test['医院编号_id'])
    y_display['predicted'] = y_prob

    print(classification_report(y_truth,y_pre2))#17_23647，17_24278
    print(confusion_matrix(y_truth,y_pre2))
    print("f1_score:",f1_score(y_truth,y_pre2))
    print("AUC:",roc_auc_score(y_truth,y_prob))
    print("precision:",precision_score(y_truth,y_pre2))
    print("recall:",recall_score(y_truth,y_pre2))
    print("specificy:",recall_score(y_truth,y_pre2,pos_label=0))
    print("accuracy:",accuracy_score(y_truth,y_pre2))

    #lgb.plot_importance(clf2,max_num_features=30,grid=False,figsize=(7.0, 5.0),title=None,xlabel=None)
    feat_importance = pd.DataFrame(X_train[col].columns.tolist(),columns=['feature'])
    feat_importance['importance']=list(clf2.feature_importances_)
    feat_importance = feat_importance.sort_values(by='importance',ascending=False).head(30).reset_index(drop=True)

    if top == 0:

        X_test_display = X_test.join(y_truth)
        X_test_display = pd.merge(y_display,X_test_display,on='医院编号_id')

        X_test_display.drop('医院编号',axis=1,inplace=True)

        X_test_display = X_test_display.fillna('NULL')
        X_test_display = X_test_display.replace(-1,'NULL')
        X_test_display = X_test_display.replace(-999,'NULL')

        X_test_display['gender'] = X_test_display['gender'].map({0:'female',1:'male'})

        X_test_display['hospital grade'] = X_test_display['hospital grade'].map({0:'Tertiary Hospital',1:'Secondary Hospital'})

        depart_dict = {2:'内科学',4:'外科学',5:'妇产科',7:'急危重症',14:'肿瘤',10:'神经病学',3:'口腔',12:'老年医学',1:'全科医学',
         13:'耳鼻喉',8:'皮肤病与性病学',11:'精神病与精神卫生学',0:'中医其他',9:'眼科',6:'康复医学与理疗学'}

        depart = pd.read_csv("depart_dict.csv",encoding='gbk')
        k,v = depart['二级学科1'],depart['Secondary Discipline 1 ']
        depart_dict2 = dict(zip(k,v))

        X_test_display['hospital department'] = X_test_display['hospital department'].map(depart_dict)
        X_test_display['hospital department'] = X_test_display['hospital department'].map(depart_dict2)
        #print(X_test_display['hospital department'].value_counts())

        drop_list = ['出院日期','检验日期',"入院日期"]
        X_test_display.drop(drop_list,axis=1,inplace=True)

        X_test_display = X_test_display.rename(columns={'医院编号_id':'Hospital Number_Patinent ID'})
        #a = X_test_display.head(1)
        col = list(X_test_display.columns)
        col[:5] = [c.title() for c in col[:5]]
        X_test_display.columns = col

        if flag_scr == 1 :
            feat_importance.to_csv(fig_path+"{}_SCr_importance.csv".format(hour),index=None)

            #X_test_display.to_csv(demp_path+"X_test_display{}.csv".format(hour),index=None)

        elif flag_scr == 0 :
            feat_importance.to_csv(fig_path+"{}_noSCr_importance.csv".format(hour),index=None)

            #X_test_display.to_csv(demp_path+"X_test_display{}_noSCr.csv".format(hour),index=None)
        return X_train, X_test, y_train, y_truth, y_prob, feat_importance,clf2,X_test_display,y_pre2#,y_display

    elif top == 1:

        return X_train, X_test, y_train, y_truth, y_prob,clf2,y_display,y_pre2

"""
Show_id = X_test_display[(X_test_display['Hospital Number_Patinent Id']=='9_382787')|(X_test_display['Hospital Number_Patinent Id']=='4_165917')]
Show_id.to_csv("Show_id.csv",index=None)

"""
"""
测试display的test数据是否有问题
col = X_test_display.columns
[c for c in col if 'ALB' in c]
col = ['first ALB',
 'last ALB',
 'ALB test times',
 'minimum ALB',
 'maximum ALB',
 'mean ALB',
 'standard deviation ALB']
A = X_test_display[X_test_display['ALB test times']==1]

A = A[col]
A_des = A.describe()
print("A.shape ",A.shape)
A = A[(A['minimum ALB']==A['maximum ALB'])&(A['first ALB']==A['last ALB'])&(A['first ALB']==A['maximum ALB'])]
print("A.shape ",A.shape)

A = X_test_display[X_test_display['first SCr']=='NULL']
A['KIDIGO_AKI'].value_counts()
0    2726
1     399

A_AKI = A[A['KIDIGO_AKI']==1]#[[]]

A_AKI = A_AKI.rename(columns={'Hospital Number_Patinent Id':'医院编号_id'})
print("A_AKI shape ",A_AKI.shape)
A_AKI = pd.merge(b_Info_A[['医院编号_id',"入院日期","检验日期","出院日期"]],A_AKI,on = '医院编号_id')
print("A_AKI shape ",A_AKI.shape)
A_AKI_des = A_AKI[['first SCr','last SCr',
 'minimum SCr',
 'maximum SCr',
 'mean SCr',
 'standard deviation SCr']].describe()

A_date = A_AKI[["入院日期","检验日期","出院日期"]]
"""
X_train, X_test, y_train, y_truth, y_prob, feat_importance,clf , X_test_display,y_pre2 = predict_(flag_scr=1,top=0)#y_display,

X_train_noScr, X_test_noScr, y_train_noScr, y_truth_noScr, y_prob_noScr, feat_importance_noScr, \
                        clf2_noScr,  X_test_display_noScr,y_pre2_noScr = predict_(flag_scr=0,top=0)#y_display_noScr,

X_train_top, X_test_top, y_train_top, y_truth_top, y_prob_top, clf2_top, y_display_top,y_pre2_top = predict_(flag_scr=1,top=1)
X_train_noScr_top, X_test_noScr_top, y_train_noScr_top, y_truth_noScr_top, y_prob_noScr_top,\
                                 clf2_noScr_top, y_display_noScr_top,y_pre2_noScr_top = predict_(flag_scr=0,top=1)

def printAuc(y_truth,y_prob,y_truth_noScr,y_prob_noScr,t,top):
    fpr,tpr,th=roc_curve(y_truth,y_prob)
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=2,alpha=0.8,color='g',label='{}h_with_SCr(AUC=%0.3f)'.format(t)%(roc_auc))
   
    fpr_noScr,tpr_noScr,th_noScr=roc_curve(y_truth_noScr,y_prob_noScr)
    roc_auc_noScr=auc(fpr_noScr,tpr_noScr)
    plt.plot(fpr_noScr,tpr_noScr,lw=2,alpha=0.8,color='r',label='{}h_without_SCr(AUC=%0.3f)'.format(t)%(roc_auc_noScr))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='b',label='Luck', alpha=0.8)
     
    plt.xlabel('1-specificity')#FPR
    plt.ylabel('Sensitivity')#TPR
    
    plt.legend(loc="lower right")
    plt.savefig(fig_path+'{}h_AUC{}.png'.format(t,top), dpi=300) #(300.0,300.0)
    plt.show()
       
def printPR(y_truth,y_prob,y_truth_noScr,y_prob_noScr,t,top):
    
    precision, recall, thresholds = precision_recall_curve(y_truth,y_prob)
    average_precision = average_precision_score(y_truth,y_prob)
    plt.plot(precision, recall,lw=1,alpha=0.5,color='r',label='{}h_with_SCr(AP=%0.3f)'.format(t)%(average_precision))
    
    precision_noScr,recall_noScr,th_noScr=precision_recall_curve(y_truth_noScr,y_prob_noScr)
    ap_no_scr = average_precision_score(y_truth_noScr,y_prob_noScr)
    plt.plot(precision_noScr, recall_noScr,lw=1,alpha=0.5,color='b',label='{}h_without_SCr(AP=%0.3f)'.format(t)%(ap_no_scr))
    
    plt.ylabel('Precision')
    plt.xlabel('Recall/Sensitivity')
    
    plt.legend(loc="lower right")
    plt.savefig(fig_path+'{}h_PR{}.png'.format(t,top), dpi=300)
    plt.show()
'''
plt.rc('font',family='Times New Roman',size=13)
printAuc(y_truth,y_prob,y_truth_noScr,y_prob_noScr,hour,0)
printPR(y_truth,y_prob,y_truth_noScr,y_prob_noScr,hour,0) 

printAuc(y_truth_top,y_prob_top,y_truth_noScr_top,y_prob_noScr_top,hour,0)
printPR(y_truth_top,y_prob_top,y_truth_noScr_top,y_prob_noScr_top,hour,0) 
'''
"""
X.shape
X_train.shape
X_un = X.nunique()
X['KIDIGO_AKI'].mean()#0.078
X.info()

X_train.groupby('gender')['KIDIGO_AKI'].transform('count')

train = X_train.join(y_train)
test = X_test
print("train shape ",train.shape)
print("test shape ",test.shape)

data = pd.concat([train,test],axis=0)
print("data shape ",data.shape)

data['department_cnt'] = data.groupby('hospital department')['KIDIGO_AKI'].transform('count')

train = data.iloc[:463809]
test = data.iloc[463809:]

col = list(train.columns)
X_train = train[c for c in col if c!='KIDIGO_AKI']
X_test = test[c for c in col if c!='KIDIGO_AKI']

con_col = ['age','first K','last K','K test times','minimum K','maximum K','mean K','standard deviation K',
 'first NA','last NA','NA test times','minimum NA','maximum NA','mean NA','standard deviation NA',
 'first CL','last CL','CL test times','minimum CL','maximum CL','mean CL','standard deviation CL',
 'first ALB','last ALB','ALB test times','minimum ALB','maximum ALB','mean ALB','standard deviation ALB',
 'first UA','last UA','UA test times','minimum UA','maximum UA','mean UA','standard deviation UA',
 'first HCT','last HCT','HCT test times','minimum HCT','maximum HCT','mean HCT','standard deviation HCT',
 'first CO2CP','last CO2CP','CO2CP test times','minimum CO2CP','maximum CO2CP','mean CO2CP','standard deviation CO2CP',
 'first PLT','last PLT','PLT test times','minimum PLT','maximum PLT','mean PLT','standard deviation PLT',
 'first HGB','last HGB','HGB test times','minimum HGB','maximum HGB','mean HGB','standard deviation HGB',
 'first SCr','last SCr','minimum SCr','maximum SCr','mean SCr','standard deviation SCr',
 'The number of potential nephrotoxic drugs used',
 'the days of use of anti-infective drugs',
 'the days of use of  diuretic drugs',
 'the days of use of vasoactive drugs',
 'the days of use of RASI',
 'the days of use of iodinated CMs',
 'the days of use of hemostatic drugs',
 'the days of use of chemotherapy drugs',
 'the days of use of NSAIDs',
 'the days of use of CNI',
 'the days of use of hypophysin',
 'Charlson total score',
 'length of stay']

cate_col = ['gender','hospital department','hospital grade','Antibiotic used',
 'Diuretic used','Vasoactive used','RASI used','Iodinated contrast medium used',
 'Hemostatic used','Chemotherapy used','NSAIDs used','Calcineurin inhibitor(CNI) used','Hypophysin used',
 'Myocardial infarction','Congestive heart failure','Peripheral vascular disease',
 'Cerebrovascular disease','Dementia','Chronic pulmonary disease','Connective tissue disease',
 'Ulcer disease','Mild liver disease','Diabetes','Hemiplegia',
 'Diabetes with end-organ damage','Any tumor','Leukemia','Lymphoma',
 'Moderate or severe liver disease','Metastatic solid tumor','AIDS',]
"""

#######################################################################################################

def clf_select(flag_scr,hour,top):
    if flag_scr == 1:
        if hour == 24:
            clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 2})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.2,class_weight={0: 1, 1: 3})
        elif hour == 72:
            clf2=lgb.LGBMClassifier(learning_rate=0.1,class_weight={0: 1, 1: 2})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.25,class_weight={0: 1, 1: 3})
        elif hour == 48:
            clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 2})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 3})
    elif flag_scr == 0:
        if hour == 24:
            clf2=lgb.LGBMClassifier(learning_rate=0.25,class_weight={0: 1, 1: 3})
            if top == 1:   
                clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 3})
        elif hour == 72:
            clf2=lgb.LGBMClassifier(learning_rate=0.1,class_weight={0: 1, 1: 3})
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 3})
        elif hour == 48:
            clf2=lgb.LGBMClassifier(learning_rate=0.2,class_weight={0: 1, 1: 3})#无论top为0/1均是最优解
            if top == 1:
                clf2=lgb.LGBMClassifier(learning_rate=0.2,class_weight={0: 1, 1: 3})   
    return clf2
"""
SHAP plot

Non-AKI, 4_165917
AKI, 9_382787

"""


def single_shap(X_train,X_test,y_train,y_prob,flag_scr,hour,clf):
    col = [c for c in X_train.columns if c not in ['SCr test times','医院编号','医院编号_id','KIDIGO_AKI','出院日期','检验日期',"入院日期"]]
    
    if flag_scr == 1:
        if hour == 24:
            clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 2})
        elif hour == 72:
            clf2=lgb.LGBMClassifier(learning_rate=0.1,class_weight={0: 1, 1: 2})
        elif hour == 48:
            clf2=lgb.LGBMClassifier(learning_rate=0.15,class_weight={0: 1, 1: 2})
    elif flag_scr == 0:
        if hour == 24:
            clf2=lgb.LGBMClassifier(learning_rate=0.25,class_weight={0: 1, 1: 3})
        elif hour == 72:
            clf2=lgb.LGBMClassifier(learning_rate=0.1,class_weight={0: 1, 1: 3})
        elif hour == 48:
            clf2=lgb.LGBMClassifier(learning_rate=0.2,class_weight={0: 1, 1: 3})#无论top为0/1均是最优解
    
    clf2.fit(X_train[col],y_train)#,categorical_feature=categorical_feature
    
    for i in ['9_382787','4_165917']:
        display_id = X_test[X_test['医院编号_id']==i][col]#该show id为 
        print("Don't preocess category features",clf2.predict_proba(display_id)[:,1])
        print("Preocess category features",clf.predict_proba(display_id)[:,1])
        print("display_id.shape",display_id.shape)
        
        start = time.time()
        explainer = shap.TreeExplainer(clf2, data=X_train[col], model_output='probability')#X_test_sam
        shap_values = explainer.shap_values(display_id)#X,X_test
        
        shap.force_plot(explainer.expected_value,shap_values,display_id,matplotlib=True,figsize=(23,3))#,link = 'logit'
        
        end = time.time()
        print((end-start)/60)
        
        print('mean:',y_prob.mean())
        print('base value:',explainer.expected_value)
    
        print(clf2.predict_proba(display_id)[:,1])
        
    display_id = X_test[(X_test['医院编号_id']=='9_382787')|(X_test['医院编号_id']=='4_165917')][col]    
    print("Don't preocess category features",clf2.predict_proba(display_id)[:,1])
    print("Preocess category features",clf.predict_proba(display_id)[:,1])
    print("display_id.shape",display_id.shape)    
    explainer = shap.TreeExplainer(clf2, data=X_train[col], model_output='probability')#X_test_sam
    shap_values = explainer.shap_values(display_id)
    
"""
data_explainer = pd.DataFrame()
data_explainer['feature'] = list(X_test[col].columns)
#data_explainer['fature value'] = display_id.values.T
#data_explainer['shap value'] = shap_values.T

data_explainer2 = pd.DataFrame(data= display_id.values.T,columns=['9_382787','4_165917'])
data_explainer3 = pd.DataFrame(data= shap_values.T,columns=['9_382787(SHAP values)','4_165917(SHAP values)'])
data_explainer = data_explainer.join(data_explainer2).join(data_explainer3)
data_explainer.set_index('feature',inplace=True)
data_explainer = data_explainer.sort_values(by=['9_382787(SHAP values)'],ascending=False )

A0 = data_explainer[['9_382787','9_382787(SHAP values)']]
A1 = data_explainer[['4_165917','4_165917(SHAP values)']]
A1 = A1.sort_values(by=['4_165917(SHAP values)'],ascending=False )
A0[A0['9_382787(SHAP values)']>0].sum()
A1[A1['4_165917(SHAP values)']>0].sum()

A0[A0['9_382787(SHAP values)']<0].sum()
A1[A1['4_165917(SHAP values)']<0].sum()

single_shap(X_train = X_train,X_test = X_test,y_train = y_train,y_prob = y_prob,flag_scr = 1,\
            hour = hour,clf = clf)
single_shap(X_train = X_train_noScr,X_test = X_test_noScr,y_train = y_train_noScr,y_prob = y_prob_noScr,flag_scr = 0,\
            hour = hour, clf = clf2_noScr)
"""
categorical_top = ['hospital department','hospital grade']

def feature_plot(model_top,X_train_top,hour,top,flag_scr,y_train_top):
       
    model_top = clf_select(flag_scr,hour,top)
    col = [c for c in X_train_top.columns if c not in ['医院编号','医院编号_id','KIDIGO_AKI','出院日期','检验日期',"入院日期"]]
    
    model_top.fit(X_train_top[col],y_train_top,categorical_feature=categorical_feature)
    
    explainer = shap.TreeExplainer(model_top)#, data = X_train_top, model_output='probability'
    
    shap_values = explainer.shap_values(X_test_top[col])#X_train_top
    #shap.summary_plot(shap_values,X_train_top[col],max_display = 53,class_names=['non-AKI','AKI'])
    shap.summary_plot(shap_values[1],X_test_top[col],max_display = 53)#X_train_top


clf2.fit(X_train[col],y_train,categorical_feature=categorical_feature)
explainer = shap.TreeExplainer(clf2)
shap_values = explainer.shap_values(X_test[col])
shap.summary_plot(shap_values[1],X_test[col],max_display = 53)


X_train_top = X_train_top.reset_index(drop=True)
y_train_top = y_train_top.reset_index(drop=True)
X_test_top = X_test_top.reset_index(drop=True)
y_truth_top = y_truth_top.reset_index(drop=True)

X_test_top = pd.merge(y_pre2[['KIDIGO_AKI','医院编号_id']],X_test_top,on='医院编号_id')

y_pre2_top = pd.DataFrame(y_pre2_top,columns={'predict label'})
'''
y_pre2_top = y_pre2_top.join(y_truth_top)
y_pre2_top = y_pre2_top.join(X_test_top['医院编号_id'])
print("y_pre2.shape ",y_pre2_top.shape)
y_pre2 = y_pre2[y_pre2["KIDIGO_AKI"]==y_pre2["predict label"]].reset_index(drop=True)
print("y_pre2.shape ",y_pre2.shape)
'''


A = pd.DataFrame(shap_values[1],columns=X_train_top[col].columns)
A = A.join(y_truth_top)
"""
A = A.join(X_test_top['医院编号_id'])
A = A.join(y_pre2_top)
"""

A_0 = A[A['length of stay']<0].reset_index(drop=True)
A_1 = A[A['length of stay']>0].reset_index(drop=True)
A_netrual = A[A['length of stay']==0].reset_index(drop=True)
print("A_0 shape ",A_0.shape)
print("A_1 shape ",A_1.shape)
print("A_netrual shape ",A_netrual.shape)

A_0["KIDIGO_AKI"].value_counts()
A_1["KIDIGO_AKI"].value_counts()

A_0 = A_0[A_0["KIDIGO_AKI"]==A_0["predict label"]].reset_index(drop=True)
A_1 = A_1[A_1["KIDIGO_AKI"]==A_1["predict label"]].reset_index(drop=True)

print("A_0 shape ",A_0.shape)
print("A_1 shape ",A_1.shape)

A_0_aki = A_0[A_0["KIDIGO_AKI"]==1]
A_0_aki = A_0_aki[A_0_aki.columns[-4:]]
A_0_aki = A_0_aki.rename(columns={'length of stay':'LOS SHAP'})
print(A_0_aki.shape)
B = pd.merge(X,A_0_aki[['LOS SHAP','医院编号_id']],on=['医院编号_id'])
B.groupby("KIDIGO_AKI").size()
print("B shape ",B.shape)
B['length of stay'].value_counts()
B['length of stay'].describe()
B['LOS SHAP'].describe()
"""
count    366.000000
mean      14.915301
std        2.983229
min       10.000000
25%       13.000000
50%       14.000000
75%       16.000000
max       26.000000
Name: length of stay, dtype: float64
count    366.000000
mean      -0.392648
std        0.304788
min       -1.623137
25%       -0.548027
50%       -0.339807
75%       -0.141908
max       -0.001246
Name: LOS SHAP, dtype: float64
"""

A_1_aki = A_1[A_1["KIDIGO_AKI"]==1]
A_1_aki = A_1_aki[A_1_aki.columns[-4:]]
A_1_aki = A_1_aki.rename(columns={'length of stay':'LOS SHAP'})
print(A_1_aki.shape)
B1 = pd.merge(X,A_1_aki[['LOS SHAP','医院编号_id']],on=['医院编号_id'])
B1.groupby("KIDIGO_AKI").size()
print("B1 shape ",B1.shape)
B1['length of stay'].value_counts()
B1['length of stay'].describe()
B1['LOS SHAP'].describe()
"""
count    7396.000000
mean        5.240535
std         2.822710
min         1.000000
25%         3.000000
50%         5.000000
75%         7.000000
max        14.000000
"""
X[X['length of stay']>=14].groupby(['KIDIGO_AKI'])
B = X[X['医院编号_id']=='10_256052']
"""
A_0
0    251794
1      7088

A_1
0    175458
1     29469
"""
"""
A_0 shape  (258882, 114)
A_1 shape  (204927, 114)
A_netrual shape  (0, 115)
"""
feature_plot(clf, X_train, hour, top = 1, flag_scr = 1)

feature_plot(clf2_top, X_train_top, hour, top = 1, flag_scr = 1,y_train_top)
feature_plot(clf2_noScr_top, X_train_noScr_top , hour = hour, top = 1, flag_scr = 0)

'''
'''
def weight_score(X_train, y_train, X_test, y_truth):
    categorical_feature = ['hospital department','hospital grade','gender']#,'hospital grade','gender'
    auc_,f1_,pre_,recall_,spe_=[],[],[],[],[]
    for w in range(1,13):
        weight={0:1,1:w}
        clf2=lgb.LGBMClassifier(learning_rate=0.25,class_weight=weight)#,n_estimators=2000,max_depth = 6
        clf2.fit(X_train, y_train ,eval_set=[(X_train, y_train)],categorical_feature=categorical_feature, verbose=50,early_stopping_rounds=500,)#verbose=100
        cv_pred = clf2.predict_proba(X_test)[:,1]
        y_pre2=clf2.predict(X_test,num_iteration=clf2.best_iteration_)
        
        print(weight)
        
        auc_.append(roc_auc_score(y_truth,cv_pred))
        f1_.append(f1_score(y_truth,y_pre2))
        pre_.append(precision_score(y_truth,y_pre2))
        recall_.append(recall_score(y_truth,y_pre2))
        spe_.append(recall_score(y_truth,y_pre2,pos_label=0))
    
    class_=pd.DataFrame(list(zip(auc_,f1_,pre_,recall_,spe_)),columns=['auc','f1','precision','recall','specificity'],index=range(1,13)) 
    max_index = f1_.index(np.max(f1_))+1
    return class_, max_index

class_0, max_index0 = weight_score(X_train, y_train, X_test, y_truth)
class_0.to_csv(fig_path+'class_SCr{}_{}.csv'.format(hour,max_index0))
class_, max_index = weight_score(X_train_noScr, y_train_noScr, X_test_noScr, y_truth_noScr)
class_.to_csv(fig_path+'class_noSCr{}_{}.csv'.format(hour,max_index))
'''
'''
"""
test指标测试无问题
"""
c1 = [c for c in X_test.columns if 'K' in c]
c2 = ['first NA','last NA','minimum NA','maximum NA']
X_test_display = deepcopy(X_test[test_df_feat.columns])
X_test_display[0] = X_test_display['maximum NA']-X_test_display['minimum NA']
X_test_display[1] = X_test_display['maximum NA']-X_test_display['first NA']
X_test_display[2] = X_test_display['maximum NA']-X_test_display['last NA']
X_test_display[3] = X_test_display['first NA']-X_test_display['minimum NA']
X_test_display[4] = X_test_display['last NA']-X_test_display['minimum NA']
display_des = X_test_display[[0,1,2,3,4]].describe()

category=['K','CL','ALB','UA','HCT','CO2CP','PLT','HGB']#'K','NA',
for i in category:
    X_test_display['0{}'.format(i)] = X_test_display['maximum {}'.format(i)]-X_test_display['minimum {}'.format(i)]
    X_test_display['1{}'.format(i)] = X_test_display['maximum {}'.format(i)]-X_test_display['first {}'.format(i)]
    X_test_display['2{}'.format(i)] = X_test_display['maximum {}'.format(i)]-X_test_display['last {}'.format(i)]
    X_test_display['3{}'.format(i)] = X_test_display['first {}'.format(i)]-X_test_display['minimum {}'.format(i)]
    X_test_display['4{}'.format(i)] = X_test_display['last {}'.format(i)]-X_test_display['minimum {}'.format(i)]    

display_des = X_test_display[X_test_display.columns[-40:]].describe()
'''