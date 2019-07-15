# -*- coding: utf-8 -*-
"""
浅层神经网络模型实现：只有一个隐藏层
实现思路：
    1.定义网络的各层维度
    2.根据各层维度，初始化参数
    3.循环实现前向反馈神经网络
    4.根据梯度下降法实现反向传播神经网络，不断更新参数,包括：
        计算代价函数
        利用反向传播，计算参数的梯度
        利用梯度下降，更新参数
    5.训练模型
    6.预测并评估模型
"""
import numpy as np 
from matplotlib import pyplot as plt
import sklearn.linear_model

def loadDataset():
    '''
    函数说明：加载数据
    '''
    np.random.seed(1)
    m = 400 # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)
    a = 4 # maximum ray of the flower

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
        
    X = X.T
    Y = Y.T

    return X, Y

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    x.astype(np.float)
    s = 1/(1+np.exp(-x))
    return s
 

def tanh(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- tanh(x)
    """
#    x.astype(np.float128)
    x.astype(np.float)
    s = np.tanh(x)
    return s


def layer_sizes(X,Y):
    '''
    函数说明：定义网络层维度变量
       
    Arguments：
        X:特征变量数组，行数代表特征个数，列数代表样本个数
        Y:标签数组，列数表示样本个数
    Returns:
        n_x: 输入层层数，即特征数
        n_h: 隐藏层层数，这里定义为4层
        n_y: 输出层层数
    '''
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    """
    函数说明： 初始化参数
    
    Arguments:
        n_x: size of the input layer
        n_h: size of the hidden layer
        n_y: size of the output layer
    Returns:
        params: python dictionary containing your parameters:
            W1: weight matrix of shape (n_h, n_x)
            b1: bias vector of shape (n_h, 1)
            W2: weight matrix of shape (n_y, n_h)
            b2: bias vector of shape (n_y, 1)
    """
    np.random.seed(2) # we set up a seed so that your output matches ours although the initialization is random.
    
#   rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
    W1 = np.random.randn(n_h, n_x) * 0.01 
    b1 = np.zeros((n_h, 1))
    W2 = np.random.rand(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1)  )
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    #参数数组
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
 
#前向反馈神经网络
def forward_propagation(X, parameters):
    """
    函数说明：
        神经网络正向传播，利用模型参数，得到模型预测值和每一层中间数据
    
    Argument:
        X: 特征向量 (n_x, m)
        parameters：模型参数字典 (output of initialization function)
    Returns:
        A2 ：输出层结果
        cache ：包含每一层中间数据的字段 "Z1", "A1", "Z2" and "A2"，反向传播求梯度会用到
    """
    # 读取参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    # 根据参数计算各层结果
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1) #隐藏层激活函数选择 tanh
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2) #输出层激活函数用sigmoid
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache  #输出预测结果和中间值

#计算代价函数
def compute_cost(A2, Y):
    """
    函数说明：计算代价函数，代价函数为：log似然函数
    
    Arguments:
        A2 ：模型输出层结果(1, number of examples)
        Y ： 实际标签数组 (1, number of examples)
    Returns:
        cost：代价函数值
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2) ) + np.multiply( 1-Y, np.log(1-A2))
    cost = -1 / m * np.sum(logprobs)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    """
    函数说明：神经网络反向传播获取参数对应于代价函数的导数值 
    
    Arguments:
        parameters: python dictionary containing our parameters 
        cache: a dictionary containing "Z1", "A1", "Z2" and "A2".
        X: input data of shape (2, number of examples)
        Y: "true" labels vector of shape (1, number of examples)
    Returns:
        grads :  python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    ### END CODE HERE ###
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(parameters, grads, learning_rate = 0.1):
    """
    函数说明：利用梯度下降更新参数，梯度由反向传播函数得到    
    
    Arguments:
        parameters：更新前的模型参数
        grads：代价函数对应参数的求导，由 backward_propagation()得到 
    Returns:
        parameters： 更新后的参数 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    """
    函数说明：神经网络模型训练，利用前向和反向传播更新迭代参数
    
    Arguments:
        X: 特征数据数组，维度(特征数,样本数)
        Y: 标签数据数组，维度(1,样本数)
        n_h：定义隐藏层层数
        num_iterations: 迭代次数，梯度下降法参数更新次数
        print_cost: boolean型，若为True，每迭代1000次输出代价函数值
    Returns:
        parameters: 迭代后的参数W1,b1,W2,b2
    """
    
    np.random.seed(3)
    n_x, n_h, n_y = layer_sizes(X, Y)
    
    #初始化参数, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    #迭代循环更新参数
    for i in range(0, num_iterations):
        #前向神经网络,利用参数进行获取预测值 . Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        #利用预测值，计算代价函数 Inputs: "A2, Y". Outputs: "cost".
        cost = compute_cost(A2, Y)
 
        #反向传播,根据代价函数最小化，获取参数梯度. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
 
        # 梯度下降更新参数. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)
        
        #若print_cost为True，每迭代1000次输出代价       
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    """
    函数说明：利用训练得到的参数，通过正向传播得到预测值
    
    Arguments:
        parameters：训练后的参数字典{W1,b1,W2,b2}
        X:预测数据，维度为 (n_x, m)，其中n_x为样本特征数，m为样本个数
    Returns
        predictions:模型预测结果 (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)  #四舍五入，>0.5为1，否则0
    
    return predictions

def model_accuracy(real, pred):
    '''
    函数说明： 获取样本的准确率
     
    Arguments：
        real:实际样本标签
        pred:预测样本标签
    Returns:
        accuracy：预测精度
    '''
    right_num = np.dot(real, pred.T) + np.dot(1-real, (1-pred).T)  # 1预测正确的个数 + 0预测正确的个数
    accuracy = float(right_num) / real.shape[1] #准确率
    return accuracy

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
   
def lr_model(X_train, Y_train, X_test):
    clf = sklearn.linear_model.LogisticRegressionCV();
    clf.fit(X.T, Y.T.reshape(-1,));
    pred = clf.predict(X_test.T)
    return clf, pred

if __name__ == '__main__':
    X, Y = loadDataset()
    print('X维度：',X.shape)
    print('Y维度：',Y.shape)
    
    #定义神经网络维度变量：
    (n_x, n_h, n_y) = layer_sizes(X, Y)
    print(n_x, n_h, n_y)
    
    #初始化参数：
    parameters = initialize_parameters(n_x, n_h, n_y)
    print(parameters)
    #正向传播:
    A2, cache = forward_propagation(X, parameters)#获取输出值和中间变量（Z1,A1,Z2,A2）
    #反向传播：最小化代价函数，利用梯度下降法获取最优参数
    cost = compute_cost(A2, Y)                       #计算代价函数
    grads = backward_propagation(parameters, cache, X, Y) #获取梯度值dW2,db2,dW1,db1
    update_param = update_parameters(parameters, grads, learning_rate = 0.1) #利用梯度下降法更新参数
    
    ##循环正向和反向传播，不断更新参数，更新代价函数值，直到代价值变化不明显或者迭代次数达到上限
    nn_parameters = nn_model(X, Y, n_h, num_iterations = 10000, print_cost=True)
    
    #对数据进行预测
    Y_pred = predict(nn_parameters, X)
    
    # Plot the decision boundary
    model = lambda x: predict(parameters, x.T)
    plot_decision_boundary(model, X, Y.reshape(-1,))
    #对模型进行评估
    accuracy = model_accuracy(Y,Y_pred)
    print('二层神经网络 模型准确率：%.2f%% '%(accuracy*100) )
    
    # 利用LR进行预测
    lr_clf, lr_pred = lr_model(X, Y, X)
    plot_decision_boundary(lambda x: lr_clf.predict(x), X, Y.reshape(-1,))
    lr_accuracy =  model_accuracy(Y, lr_pred)
    print('lr 模型准确度：%.2f%% '%(lr_accuracy*100) )