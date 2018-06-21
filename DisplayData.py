import numpy
import seaborn as sns
import matplotlib.pyplot as plt

def showPlot(data):
    data.plot()
    plt.show()

def showSubPlot(data):
    data.plot(subplots=True, figsize=(8, 8))
    plt.show()

def showHist(data):
    data.hist()
    plt.show()
# For numeric data, the result's index will include count, mean, std, min, max as well as lower, 50 and upper percentiles. By default the lower percentile is 25 and the upper percentile is 75. The 50 percentile is the same as the median.
# For object data (e.g. strings or timestamps), the result's index will include count, unique, top, and freq. The top is the most common value. The freq is the most common value's frequency. Timestamps also include the first and last items.
def showStatistic(data):
    return data.describe()
#Compute pairwise correlation of columns, excluding NA/null values
#method :
#pearson : standard correlation coefficient
#kendall : Kendall Tau correlation coefficient
#spearman : Spearman rank correlation
def showCorr(data,method='pearson'):
    return data.corr(method=method)
def showCorrMap(data,method='pearson'):
    names=data.columns.values.tolist()
    length=len(names)
    correlations = data.corr(method=method)  # 计算变量之间的相关系数矩阵
    # plot correlation matrix
    fig = plt.figure()  # 调用figure创建一个绘图对象
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)  # 绘制热力图，从-1到1
    fig.colorbar(cax)  # 将matshow生成热力图设置为颜色渐变条
    ticks = numpy.arange(0, length, 1)  # 生成0-9，步长为1
    ax.set_xticks(ticks)  # 生成刻度
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)  # 生成x轴标签
    ax.set_yticklabels(names)
    plt.show()
def showSPLOM(data):
    sns.pairplot(data)
    plt.show()

def showSPLOM_G(data):
    sns.set(color_codes=True)
    g = sns.PairGrid(data)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
    plt.show()
#Normalized by N-1 by default
def showStd(data,ddof=1):
    print(data.std(ddof=ddof))
    return data.std(ddof=ddof)

def showBins(data):
    p = data.boxplot(return_type='dict')
    # for i in range(0,4):
    x = p['fliers'][0].get_xdata()
    y = p['fliers'][0].get_ydata()
    y.sort()
    for i in range(len(x)):
        if i > 0:
            plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.05 - 0.8 / (y[i] - y[i - 1]), y[i]))
        else:
            plt.annotate(y[i], xy=(x[i], y[i]), xytext=(x[i] + 0.08, y[i]))
    plt.show()

