#coding=utf-8
"""
filename:       time_dif_correct_copy.py
Description:
Author:         Dengjie Xiao
IDE:            PyCharm
Change:         2019/3/18 0018  9:37    DengjieXiao     Create

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import TrainData
BASE_TIME=1546418000
bpm_name = 'TCBPM1X'
filename1= 'C:/Users/qiaoys/Downloads/{}.dat'.format(bpm_name)
ind1='timestamp'
val1= bpm_name
range1A=1500
range1B=4000
data1=pd.read_csv(filename1)
#index1=data[ind1][1:100]
#index2=data[ind2][0:10]
ind_1=data1.loc[range1A:range1B, ind1]
print(ind_1)
val_1=data1.loc[range1A:range1B, val1]
val_1val=val_1.values

co_name =  'C16Vai'
filename2='data/{}.dat'.format(co_name)
ind2='timestamp'
val2=co_name
range2A=1
range2B=8000
data2=pd.read_csv(filename2)

#index1=data[ind1][1:100]
#index2=data[ind2][0:10]
ind_2=data2.loc[range2A:range2B, ind2]
val_2=data2.loc[range2A:range2B, val2]
val_2val=val_2.values
ser1 = pd.Series(val_1val,index=ind_1- BASE_TIME)   #[index2[1],index2[2],index2[3],index2[4],index2[5],index2[6],index2[7],index2[8],index2[9],index2[10]]

ser2 = pd.Series(val_2val,index=ind_2- BASE_TIME)
ser2_ori = ser2.reindex(ind_1- BASE_TIME, method='ffill') #合并时时间对齐
c = pd.DataFrame({co_name:ser2_ori.values,bpm_name:ser1.values,'time':ind_1},columns=['time',bpm_name,co_name])
fig, ax = plt.subplots()
#ax.plot(c.iloc[0:1000,0],c.iloc[0:1000,2])#,label ='C15'
#ax.plot(c.iloc[0:1000,0],c.iloc[0:1000,1])#,label='TCBPM2X'
ax.plot(c.iloc[:,0],c.iloc[:,2])#,label ='C15'
ax.plot(c.iloc[:,0],c.iloc[:,1])#,label='TCBPM2X'
ax.legend()
ax.set_xlabel('timestamp')
ax.set_ylabel('value')
plt.show()

def loss_t_dif(time_dl):
    ser2_new = ser2.reindex(ind_1- BASE_TIME -time_dl, method='ffill')
    c = pd.DataFrame({'time':ind_1,bpm_name:ser1.values,co_name:ser2_new.values},
                     columns=['time',bpm_name,co_name])
    val_1val=c.iloc[:,1].values
    val_2val=c.iloc[:,2].values
    ser1_1 = pd.Series(val_1val,index=round(ind_1- BASE_TIME))
    ser1_2 = pd.Series(val_1val,index=round(ind_1- BASE_TIME -1))
    ser2_1 = pd.Series(val_2val,index=round(ind_1- BASE_TIME))
    ser2_2 = pd.Series(val_2val,index=round(ind_1- BASE_TIME -1))
    ser1_op = ser1_2 - ser1_1
    ser2_op = ser2_2 - ser2_1
    dif_df = pd.DataFrame({ 'time': ser1_op.index, bpm_name: ser1_op.values,co_name: ser2_op.values},
                          columns=['time',bpm_name,co_name])
    print(dif_df)
    dif_df = dif_df.dropna(axis=0,how='any')
    rangeA=1
    step=3000
    rangeB=rangeA+step
    X = dif_df.iloc[rangeA:rangeB, [2]]
    y = dif_df.iloc[rangeA:rangeB, [1]]
    y_pred,score=TrainData.MLLinearRegression(X, X, y, y)
    return(score)

a,b=[],[]
for time_dl in np.arange(0, 20, 1):
    a.append(time_dl)
    b.append(loss_t_dif(time_dl))
min_value = max(b)
index = b.index(max(b))
x_value = a[index]
print ('max_score is: {}\n best time_dl is: {}'.format(min_value,x_value))

time_dl= x_value
#非差分
ser2_new = ser2.reindex(ind_1- BASE_TIME -time_dl, method='ffill') #合并时时间对齐
c1 = pd.DataFrame({co_name:ser2_new.values,bpm_name:ser1.values,'time':ind_1},columns=['time',bpm_name,co_name])

#画差分的图
# ser2_new = ser2.reindex(ind_1 - 1546418000 - time_dl, method='ffill')
# c = pd.DataFrame({'time': ind_1, bpm_name: ser1.values, co_name: ser2_new.values},columns=['time',bpm_name,co_name])
# val_1val = c.iloc[:, 1].values
# val_2val = c.iloc[:, 2].values
# ser1_1 = pd.Series(val_1val,index=round(ind_1- 1546418000))   #[index2[1],index2[2],index2[3],index2[4],index2[5],index2[6],index2[7],index2[8],index2[9],index2[10]]
# ser1_2 = pd.Series(val_1val,index=round(ind_1- 1546418000 -1))
# ser2_1 = pd.Series(val_2val,index=round(ind_1- 1546418000))
# ser2_2 = pd.Series(val_2val,index=round(ind_1- 1546418000 -1))
# ser1_op = ser1_2 - ser1_1
# ser2_op = ser2_2 - ser2_1
# print(ser2_op)
# ser2_opnew = ser2_op.reindex(ind_1- 1546418000 -time_dl, method='ffill') #合并时时间对齐
# c = pd.DataFrame({'time':ind_1,bpm_name:ser1_op.values,co_name:ser2_new.values},columns=['time',bpm_name,co_name])
# print(c.head())


fig, ax = plt.subplots()
#ax.plot(c.iloc[0:1000,0],c.iloc[0:1000,2])#,label ='C15'
#ax.plot(c.iloc[0:1000,0],c.iloc[0:1000,1])#,label='TCBPM2X'
ax.plot(c1.iloc[:,0],c1.iloc[:,2])#,label ='C15'
ax.plot(c1.iloc[:,0],c1.iloc[:,1])#,label='TCBPM2X'
ax.legend()
ax.set_xlabel('timestamp')
ax.set_ylabel('value')
plt.show()



