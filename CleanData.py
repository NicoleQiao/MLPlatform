import pandas as pd
import sklearn.preprocessing

#normtype={"min-max","Z-score"}
def NormData(data,norm_type,ddof=0):
    if norm_type=="Z-score":
        df_norm = (data - data.mean()) / (data.std(ddof=ddof))
    elif norm_type=="min-max":
        df_norm=(data - data.min()) / (data.max() - data.min())
    else:
        return "normtype={'min-max','Z-score'}"
    return df_norm

#change one column to enum type
#pvname:col name
#cur_ranges:divide range,[1,5,7]
#right:default Ture,Indicates whether the bins include the rightmost edge or not. If right == True (the default), then the bins [1,2,3,4] indicate (1,2], (2,3], (3,4].
#enumnames:array or boolean, default None. Used as labels for the resulting bins. Must be of the same length as the resulting bins. If False, return only integer indicators of the bins.
def EnumData(data,pvname,cut_ranges, right=True,enumnames=None):
    EnumPV = pd.cut(data[pvname], cut_ranges,right=right,labels=enumnames)
    print(EnumPV)
    data[pvname]=EnumPV
    return data

#use of boolean vectors to filter the data. The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses.
#data[(data['pvname']>1000)&data['pvname']<20]

#fillna_type:{'backfill', 'bfill', 'pad', 'ffill', None}, default pad 向后补全
def fillempty(data,method='pad'):
    data.fillna(method=method)
    return data
