import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV


# split data set
def split_data(data, t_pvname, test_size=0.3):
    y = data[t_pvname]
    X = data.drop(t_pvname, axis=1)
    print("{} rows of data in total, every data has {} features.".format(*X.shape))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print("split succussful,{} train data,{} test.py data。".format(X_train.shape[0], X_test.shape[0]))
    return X_train, X_test, y_train, y_test


# LinearRegression
def MLLinearRegression(X_train, X_test, y_train, y_test):
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print("intercept:", linreg.intercept_)
    print("coef:", linreg.coef_)
    y_pred = linreg.predict(X_test)

    score = linreg.score(X_test, y_test)
    print("......Results of the Linear Regression......")
    print("score:", score)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    return y_pred, score
    # DisplayData.plot_predict(y_test, y_pred,title='score: %f' % score)


#RidgeRegression(
def MLRidgeRegression(X_train, X_test, y_train, y_test, alphas=[0.01, 0.1, 0.5, 1, 3, 5, 7, 10, 20, 100]):
    ridgecv = RidgeCV(alphas)
    ridgecv.fit(X_train, y_train)
    print("Best alpha for Ridge Regreesion", ridgecv.alpha_)
    ridge = Ridge(ridgecv.alpha_)
    ridge.fit(X_train, y_train)
    print('Intercept:',ridge.intercept_)
    print('Coefficients: ', ridge.coef_)
    y_pred = ridge.predict(X_test)
    score=ridge.score(X_test, y_test)
    print("Residual sum of squares: %.2f"
          % np.mean((y_pred - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % score)
    return y_pred, score


# GaussianNB
def MLGaussianNB_testmodel(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("ori data:", y_test.values)
    print("......Results of the GaussianNB......")
    # print("prediction probability:",gnb.predict_proba(X_test))
    # print("prediction:",y_pred)
    list = (y_test != y_pred)
    n = np.sum(list)
    p = n / len(y_test)
    score = gnb.score(X_test, y_test)
    print("wrong prediction precent:", p)
    print("Residual sum of squares: %.2f" % np.mean((gnb.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % score)
    return y_pred, score


def MLGaussianNB(data, target_pv, predict_data):
    X = data.drop(target_pv, axis=1)
    y = data[target_pv]
    gnb = GaussianNB()
    gnb.fit(X, y)
    y_pred = gnb.predict(predict_data)
    print("......Results of the GaussianNB......")
    print("prediction:", y_pred)
    print("prediction probability:", gnb.predict_proba(predict_data))
    return y_pred


# Decision Trees
def MLDecisionTrees_testmodel(X_train, X_test, y_train, y_test, max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print("ori data:", y_test.values)
    print("......Results of the DesionTrees......")
    # print("prediction:", y_pred)
    # print("prediction probability:", clf.predict_proba(X_test))
    list = (y_test != y_pred)
    n = np.sum(list)
    p = n / len(y_test)
    score = clf.score(X_test, y_test)
    # print("wrong prediction number:", n)
    print("wrong prediction precent:", p)
    print("Residual sum of squares: %.2f" % np.mean((clf.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % score)
    return y_pred, score


def MLDecisionTrees(data, target_pv, predict_data):
    X = data.drop(target_pv, axis=1)
    y = data[target_pv]
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(predict_data)
    print("......Results of the DecisionTrees......")
    print("prediction:", y_pred)
    print("prediction probability:", clf.predict_proba(predict_data))
    return y_pred


# PolynomialRegression
def MLPolynomialRegression(X_train, X_test, y_train, y_test, degree=2):
    quadratic_featurizer = PolynomialFeatures(degree=degree)
    X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
    X_test_quadratic = quadratic_featurizer.transform(X_test)
    regressor_quadratic = LinearRegression()
    regressor_quadratic.fit(X_train_quadratic, y_train)
    reg_y_pred = regressor_quadratic.predict(X_test_quadratic)
    score = regressor_quadratic.score(X_test_quadratic, y_test)
    print('......Results of the Polynomial Regression......')
    print('degree:', degree)
    print('score:', score)
    print("MSE:", mean_squared_error(y_test, reg_y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, reg_y_pred)))
    return reg_y_pred, score


# KNN for non-linear regression
def MLKNN_Regression(X_train, X_test, y_train, y_test, type='k-fold'):
    y_pred = []
    score: float
    if type == 'k-fold':
        reg_k = fit_model_k_fold(X_train, y_train)
        print("Parameter 'n_neighbors' is {} for the optimal model.".format(reg_k.get_params()['n_neighbors']))
        y_pred = reg_k.predict(X_test)
        score = performance_metric_r2(y_test, reg_k.predict(X_test))
        print('k-fold score:', score)
    elif type == 'shuffle':
        reg_s = fit_model_shuffle(X_train, y_train)
        print("Parameter 'n_neighbors' is {} for the optimal model.".format(reg_s.get_params()['n_neighbors']))
        y_pred = reg_s.predict(X_test)
        score = performance_metric_r2(y_test, reg_s.predict(X_test))
        print('shuffle score:', score)
    else:
        print('Type should be k-fold or shuffle.')
    return y_pred, score


# weights:distance, uniform
def MLKNN_Classification(X_train, X_test, y_train, y_test, k=5, weights='uniform'):
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    # precision = metrics.precision_score(y_test, predict)
    # recall = metrics.recall_score(y_test, predict)
    # print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
    score = metrics.accuracy_score(y_test, predict)
    print('accuracy: %.2f%%' % (100 * score))
    return predict, score


def MLLogisticRegression_testmodel(X_train, X_test, y_train, y_test, multinominal=False, c=1.0):
    if multinominal == True:
        cls = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=c)
    else:
        cls = LogisticRegression(C=c)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    print("......Results of the LogisticRegression......")
    # print("ori:", y_test.values)
    # print("prediction:", y_pred)
    list = (y_test != y_pred)
    n = np.sum(list)
    p = n / len(y_test)
    score = cls.score(X_test, y_test)
    # print("wrong prediction number:", n)
    # print("wrong prediction precent:", p)
    print("Coefficients:%s, intercept %s" % (cls.coef_, cls.intercept_))
    print("Residual sum of squares: %.2f" % np.mean((cls.predict(X_test) - y_test) ** 2))
    print('Score: %.2f' % score)
    return y_pred, score


def MLLogisticRegression(X, y, predict_data, multinominal=False, c=1.0):
    if multinominal == True:
        cls = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=c)
    else:
        cls = LogisticRegression(C=c)
    cls.fit(X, y)
    print("......Results of the LogisticRegression......")
    print("prediction:", cls.predict(predict_data))
    print("prediction probability:", cls.predict_proba(predict_data))


# C:Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
# find the highest predicting score by adjusting the C parameter
# return highest score and C
def test_LogisticRegression_C(X_train, X_test, y_train, y_test, a=-2, b=4, n=100):
    Cs = np.logspace(a, b, n)
    scores = []
    for C in Cs:
        cls = LogisticRegression(C=C)
        cls.fit(X_train, y_train)
        scores.append(cls.score(X_test, y_test))
        print("C=%s,the predicting score of the model=%s" % (C, cls.score(X_test, y_test)))
    max_score = max(scores)
    max_index = scores.index(max_score)
    print("When C=%s, the highest score=%s" % (Cs[max_index], max_score))
    return Cs[max_index], max_score


##cluster dbscan
def MLDBSCAN(data, target_pv, eps=0.5, min_samples=10):
    X = data.drop(target_pv, axis=1)
    X = StandardScaler().fit_transform(X)
    labels_true = data[target_pv]
    # print(X)
    # print(labels_true)
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print("......Results of the DBSCAN......")
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
    g = sns.FacetGrid(data, hue=target_pv)
    g.map(plt.scatter,"x1",'x2')
    # # plot
    # unique_labels = set(labels)
    # print(len(unique_labels))
    # colors = [plt.cm.get_cmap('Spectral')(each)
    #           for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:
    #         # Black used for noise.
    #         col = [0, 0, 0, 1]
    #
    #     class_member_mask = (labels == k)
    #
    #     xy = X[class_member_mask & core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=14)
    #
    #     xy = X[class_member_mask & ~core_samples_mask]
    #     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #              markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


def MLKMeans(data, feature_pv1, feature_pv2, cluster=2):
    X = data[[feature_pv1, feature_pv2]]
    estimator = KMeans(n_clusters=cluster)
    estimator.fit(X)
    label_pred = estimator.labels_
    for i in range(0, cluster):
        colors = np.random.rand(3)
        print(colors)
        xi = X[label_pred == i]
        plt.scatter(xi[feature_pv1], xi[feature_pv2], c=colors, marker='o', label='label' + str(i))
    plt.xlabel(feature_pv1)
    plt.ylabel(feature_pv2)
    plt.title('cluster number= '+str(cluster))
    plt.legend()
    plt.show()


def MLMLPClassifier(X_train, X_test, y_train, y_test, activation='relu', solver='adam', alpha=0.0001, max_iter=10000):
    model = MLPClassifier(activation=activation, solver=solver, alpha=alpha, max_iter=max_iter)  # 神经网络
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    # print(predict)
    # print(y_test.values)
    score = model.score(X_test, y_test)
    print('scikitlearn MLPClassifier score:%.5f' % score)
    return predict, score

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
