import pandas as pd
import numpy as np
import sklearn.preprocessing
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import spline
import DisplayData
import LoadData
from libs.detect_peaks import detect_peaks
import peakutils.peak


# normtype={"min-max","Z-score"}
def NormData(data, norm_type, ddof=0):
    if norm_type == "Z-score":
        df_norm = (data - data.mean()) / (data.std(ddof=ddof))
    elif norm_type == "min-max":
        df_norm = (data - data.min()) / (data.max() - data.min())
    else:
        return "normtype={'min-max','Z-score'}"
    return df_norm


# change one column to enum type
# pvname:col name
# cur_ranges:divide range,[1,5,7]
# right:default Ture,Indicates whether the bins include the rightmost edge or not. If right == True (the default), then the bins [1,2,3,4] indicate (1,2], (2,3], (3,4].
# enumnames:array or boolean, default None. Used as labels for the resulting bins. Must be of the same length as the resulting bins. If False, return only integer indicators of the bins.
def EnumData(data, pvname, cut_ranges, right=True, enumnames=None):
    EnumPV = pd.cut(data[pvname], cut_ranges, right=right, labels=enumnames)
    print(EnumPV)
    data[pvname] = EnumPV
    return data


# use of boolean vectors to filter the data. The operators are: | for or, & for and, and ~ for not. These must be grouped by using parentheses.
# data[(data['pvname']>1000)&data['pvname']<20]

# fillna_type:{'backfill', 'bfill', 'pad', 'ffill', None}, default pad
def fillempty(data, method='pad'):
    data.fillna(method=method)
    return data

#detect burr(peak) in data
def detect_burr(data,pv,left=None,right=None,method=0,minimum_peak_distance=100):
    titles = data.columns
    titleList=titles.values.tolist()
    if pv in titleList:
        pvn=titleList.index(pv)
        sta = DisplayData.showStatistic(data)
        print("statistic data:")
        print(sta)
        # use boxplot define threshold
        iqr = sta.loc['75%'][titles[pvn]] - sta.loc['25%'][titles[pvn]]
        if left is None:
            left = sta.loc['25%'][titles[pvn]] - 1.5 * iqr
        if right is None:
            right = sta.loc['75%'][titles[pvn]] + 1.5 * iqr
        print('min edge:', left, 'max edge:', right)
        burrdata = data[((data[titles[pvn]]) < left) | ((data[titles[pvn]]) > right)]
        LoadData.df2other(burrdata, 'csv','newfile.csv')
        y = data[titles[pvn]].values
        if method == 0:
            # find_peaks by scipy signal
            peaks, _ = signal.find_peaks(y, height=right)
            plt.plot(y,'b',lw=1)
            plt.plot(peaks, y[peaks], "+",  mec='r',mew=2, ms=8)
            plt.plot(np.zeros_like(y)+right, "--", color="gray")
            plt.title("find_peaks min_height:%7f"%right)
            plt.show()
        if method==1:
            detect_peaks(y, mph=right, mpd=minimum_peak_distance, show=True)
        if method==2:
            print('Detect peaks with minimum height and distance filters.')
            # thres=right/max(y)
            indexes = peakutils.peak.indexes(np.array(y),
                                     thres=right / max(y), min_dist=minimum_peak_distance)
            print('Peaks are: %s' % (indexes))
            plt.plot(y,'b',lw=1)
            for i in indexes:
                plt.plot(i, y[i], "+", mec='r',mew=2, ms=8)
            plt.plot(np.zeros_like(y) + right, "--", color="gray")
            plt.title("peakutils.peak thres:%f ,minimum_peak_distance:%d" % (right ,minimum_peak_distance))
            plt.show()
    else:
        print("Wrong PV name, not in ",titleList)


