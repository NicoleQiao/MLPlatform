from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

#split data set
def split_data(data,t_pvname):
    y = data[t_pvname]
    X = data.drop(t_pvname, axis=1)
    print("数据共有{}条，每条含有{}个特征.".format(*X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=1)
    print("训练集与测试集拆分成功，训练集有{}条，测试集有{}条。".format(X_train.shape[0], X_test.shape[0]))
    return X_train, X_test, y_train, y_test

#LinearRegression
def MLLinearRegression(X_train, X_test, y_train, y_test):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print(linreg.intercept_)
    print(linreg.coef_)
    y_pred = linreg.predict(X_test)
    for i, prediction in enumerate(y_pred):
        print('Predicted: %s, Target: %s' % (prediction, y_test[i]))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


#regression score functions: r2, absolute_error, quare_error, explained_viarance_score
# now use r2
#sklearn.metric提供了一些函数，用来计算真实值与预测值之间的预测误差：
#以_score结尾的函数，返回一个最大值，越高越好
#以_error结尾的函数，返回一个最小值，越小越好；如果使用make_scorer来创建scorer时，将greater_is_better设为False
def performance_metric_r2(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

def performance_metric_abstErr(y_true, y_predict):
    score = mean_absolute_error(y_true, y_predict)
    return score

def performance_metric_sqrErr(y_true, y_predict):
    score = mean_squared_error(y_true, y_predict)
    return score

def performance_metric_expVia(y_true, y_predict):
    score = explained_variance_score(y_true, y_predict)
    return score

def fit_model_k_fold(X, y):
    k_fold = KFold(n_splits=10)
    # Create a decision tree regressor object
    regressor = KNeighborsRegressor()
    # Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'n_neighbors': range(3, 10)}
    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric_r2)
    # Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=k_fold)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    # Return the optimal model after fitting the data
    return grid.best_estimator_


def fit_model_shuffle(X, y):
    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
    # Create a KNN regressor object
    regressor = KNeighborsRegressor()
    # Create a dictionary for the parameter 'n_neighbors' with a range from 3 to 10
    params = {'n_neighbors': range(3, 10)}
    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric_r2)
    # Create the grid search object
    grid = GridSearchCV(regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)
    # Return the optimal model after fitting the data
    return grid.best_estimator_

#KNN for non-linear regression
#data: all dataset
#t_pvname:target PV to pridict
def MLKNN(X_train, X_test, y_train, y_test):
    reg_k = fit_model_k_fold(X_train, y_train)
    reg_s=fit_model_shuffle(X_train,y_train)
    print("Parameter 'n_neighbors' is {} for the optimal model.".format(reg_k.get_params()['n_neighbors']))
    print("Parameter 'n_neighbors' is {} for the optimal model.".format(reg_s.get_params()['n_neighbors']))
    # for i, target in enumerate(reg.predict(features)):
    #     print(target)
    print('k-fold:',performance_metric_r2(y_test, reg_k.predict(X_test)))
    print('shuffle:',performance_metric_r2(y_test, reg_s.predict(X_test)))
    return 1

def MLLogisticRegression(X_train, X_test, y_train, y_test,multinominal=False,c=1.0):
    if multinominal==True:
        cls = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=c)
    else:
        cls = LogisticRegression(C=c)
    cls.fit(X_train, y_train)
    print("Coefficients:%s, intercept %s"%(cls.coef_,cls.intercept_))
    print("Residual sum of squares: %.2f"% np.mean((cls.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % cls.score(X_test, y_test))

#C:Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
#find the highest predicting score by adjusting the C parameter
# return highest score and C
def test_LogisticRegression_C(X_train, X_test, y_train, y_test,a=-2,b=4,n=100):
    Cs=np.logspace(a,b,n)
    scores=[]
    for C in Cs:
        # 选择模型
        cls = LogisticRegression(C=C)
        # 把数据交给模型训练
        cls.fit(X_train, y_train)
        scores.append(cls.score(X_test, y_test))
        print("C=%s,the predicting score of the model=%s"%(C,cls.score(X_test, y_test)))
    max_score=max(scores)
    max_index=scores.index(max_score)
    print("When C=%s, the highest score=%s"%(Cs[max_index],max_score))
    return Cs[max_index],max_score

def MLDBSCAN(X,labels_true,eps=0.5,min_samples=10):
    X = StandardScaler().fit_transform(X)
    # print(X)
    #print(labels_true)
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    #plot
    unique_labels = set(labels)
    colors = [plt.cm.get_cmap('Spectral')(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

def MLKMeans(cluster=3):
    iris=datasets.load_iris()
    X = iris.data[:, 0:2]  ##表示我们只取特征空间中的后两个维度
    estimator = KMeans(n_clusters=cluster)  # 构造聚类器
    estimator.fit(X)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    # 绘制k-means结果
    x0 = X[label_pred == 0]
    x1 = X[label_pred == 1]
    x2 = X[label_pred == 2]
    plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()

def test3d():
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.arange(-5, 5, 0.25)
    y = np.arange(-5, 5, 0.25)
    x, y = np.meshgrid(x, y)
    r = np.sqrt(x ** 2 + y ** 2)
    z = np.sin(r)
    # 高度
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # 填充rainbow颜色
    ax.contourf(x, y, z, zdir='z', offset=-2, cmap='rainbow')
    # 绘制3D图形,zdir表示从哪个坐标轴上压下去
    plt.show()