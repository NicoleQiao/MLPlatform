import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

HIDDEN_SIZE = 200                           # LSTM中隐藏节点的个数
NUM_LAYERS = 5                              # LSTM的层数
TIMESTEPS = 50                              # 循环神经网络的训练序列长度
TRAINING_STEPS = 50001                      # 训练轮数
BATCH_SIZE = 40                             # batch大小


def getdataset(lis):
    trainX, trainY, testX, testY = [], [], [], []
    for i in range(int(len(lis) * 4 / 5)):
        trainX.append([lis[i:i + TIMESTEPS]])
        trainY.append([lis[i + TIMESTEPS]])
    for i in range(len(lis) - int(len(lis) * 4 / 5) - TIMESTEPS):
        testX.append([lis[int(len(lis) * 4 / 5) + i:int(len(lis) * 4 / 5) + i + TIMESTEPS]])
        testY.append([lis[int(len(lis) * 4 / 5) + i + TIMESTEPS]])
    return np.array(trainX, dtype=np.float32),np.array(trainY, dtype=np.float32),\
           np.array(testX, dtype=np.float32),np.array(testY, dtype=np.float32)


def lstm_model(X, y, is_training):
    # 多层的LSTM
    cell = tf.nn.rnn_cell.MultiRNNCell([
        tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
        # tf.contrib.rnn.LSTMBlockCell(HIDDEN_SIZE)
        for _ in range(NUM_LAYERS)])

    '''
    lstm_cell=tf.nn.rnn_cell.LSTMCell
    cell=tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.DropoutWrapper(lstm_cell(HIDDEN_SIZE),input_keep_prob=0.8)
         for _ in range(NUM_LAYERS)])  #dropout
    '''

    # 使用TensorFlow接口将多层的LSTM结构连接成RNN网络并计算其前向传播结果。
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = outputs[:, -1, :]

    # 对LSTM网络的输出再做加一层全链接层并计算损失。注意这里默认的损失为平均
    # 平方差损失函数。
    predictions = tf.contrib.layers.fully_connected(
        output, 1, activation_fn=tf.nn.relu)  # activation_fn=tf.nn.relu,activation_fn=tf.nn.selu,activation_fn=None

    # 只在训练时计算损失函数和优化步骤。测试时直接返回预测结果。
    if not is_training:
        return predictions, None, None

    # 计算损失函数。
    loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
    global_step = tf.Variable(0, trainable=False)
    # 创建模型优化器并得到优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),
        optimizer="Adagrad", learning_rate=tf.train.exponential_decay(0.01,global_step,400,0.96,staircase=True))
    # train_op = tf.contrib.layers.optimize_loss(
    #     loss, tf.train.get_global_step(),
    #     optimizer="Adagrad", learning_rate=0.003)
    return predictions, loss, train_op


def run_eval(sess, test_X, test_y,title=''):
    # 将测试数据以数据集的方式提供给计算图。
    ds = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    ds = ds.batch(1)
    X, y = ds.make_one_shot_iterator().get_next()

    # 调用模型得到计算结果。这里不需要输入真实的y值。
    with tf.variable_scope("model", reuse=True):
        prediction, _, _ = lstm_model(X, [0.0], False)

    # 将预测结果存入一个数组。
    predictions = []
    labels = []
    for i in range(len(data) - int(len(data) * 4 / 5) - TIMESTEPS):
        p, l = sess.run([prediction, y])
        predictions.append(p)
        labels.append(l)

    # 计算均方根误差作为评价指标。
    predictions = np.array(predictions).squeeze()
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
    print("Root Mean Square Error is: %f" % rmse)

    # 绘图。
    plt.figure()
    plt.suptitle(title + "RMSE: %f" % rmse,x=0.5,y=0.95)
    plt.plot(xa, origin, color='black', label='train data', linewidth=1)
    plt.plot(xpa, predictions, label='y_predict', linewidth=1)
    plt.plot(xpa, labels, label='y_test', linewidth=1)
    plt.legend()

    plt.figure()
    plt.suptitle(title + "RMSE: %f" % rmse,x=0.5,y=0.95)
    plt.plot(predictions, label='y_predict', linewidth=1.5)
    plt.plot(labels, label='y_test', linewidth=0.8)
    plt.legend()
    plt.show()


# 导入数据
f = open('C:/Users/qiaoys/Desktop/LSTM/vadata2.csv')
df = pd.read_csv(f)
titles = df.columns
data = np.array(df[titles[2]])
trainX, trainY, testX, testY = getdataset(data)

origin, xa, xpa = [], [], []
for j in range(int(len(data) * 4 / 5)+TIMESTEPS):
    origin.append(data[j])
    xa.append(j)
for k in range(len(data)-int(len(data) * 4 / 5)-TIMESTEPS):
    xpa.append(k+int(len(data) * 4 / 5)+TIMESTEPS)

# 将训练数据以数据集的方式提供给计算图。
ds = tf.data.Dataset.from_tensor_slices((trainX, trainY))
ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)
X, y = ds.make_one_shot_iterator().get_next()

# 定义模型，得到预测结果、损失函数，和训练操作。
with tf.variable_scope("model"):
    _, loss, train_op = lstm_model(X, y, True)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 测试在训练之前的模型效果。
    print("Evaluate model before training.")
    run_eval(sess, testX, testY,title='Evaluate model before training.')

    # 训练模型。
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss])
        if i % 1000 == 0:
            print("train steps: " + str(i) + ", loss: " + str(l))

    # 使用训练好的模型对测试数据进行预测。
    print("Evaluate model after training.")
    run_eval(sess, testX, testY,title='Evaluate model after training.')
