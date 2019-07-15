# -*- coding: utf-8 -*-
"""
需求：逻辑回归预测图片中是否含有猫

实现思路：
1.加载数据
2.数据预处理：
            图像可视化
            数据格式整理，将图片数据向量化处理，长宽高分别是width、height和depth的三维图像转化为二维向量，即： (length, height, 3) 转为(length*height*3, 1)
            数据标准化处理
3.参数初始化
4.对代价函数最小化，利用梯度下降法求解最优参数
            计算参数对应于代价函数的梯度
            求解最优参数
5. 利用最优的参数，对数据进行预测
6. 计算预测结果的准确率
7. 绘制代价值和迭代次数，以及学习速率的关系图

结论：学习速率过大，会导致模型不稳定，代价函数出现振荡，训练集的代价函数很低并不意味着模型的训练结果就是好的，可能会出现过拟合现象，要选择合适 的学习速率
- Different learning rates give different costs and thus different predictions results.
- If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost). 
- A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
- In deep learning, we usually recommend that you: 
    - Choose the learning rate that better minimizes the cost function.
    - If your model overfits, use other techniques to reduce overfitting.

小知识点：
np.squeeze
np.dot
数组.reshape
         when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is to use:
        X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X

"""
import os
os.chdir(r'C:\Users\Administrator\Desktop\DL\逻辑回归\代码整理')
import numpy as np
import h5py #网页爬取
from  matplotlib import pyplot as plt
import  lr_utils


def load_dataset():
    '''
    函数说明： 加载数据
    '''
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def image2vector(dataArr):
    '''
    函数说明：图像数据转换为列向量(209, 64, 64, 3)表示209张图片，每一张图片是（64，64，3）的图像数据
    '''
    data_flatten = dataArr.reshape(dataArr.shape[0], -1).T
    print(data_flatten.shape)#行 代表特征数， 列 代表样本数
    return data_flatten

#初始化参数
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros((dim, 1))
    b = 0
    
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


#正向传播：更新代价，反向传播：更新参数
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1] #样本数
    # FORWARD PROPAGATION (FROM X TO COST)
    z = np.dot(w.T, X) + b
    A = lr_utils.basic_sigmoid(z)  # w维度(features,1) x维度(features,samples) b维度(1,1)          
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))   #计算代价函数      
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)
    ### END CODE HERE ###
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

#利用已经求得的梯度进行参数优化,并返回最后的优化参数，对应梯度值，和代价值
def optimize(w, b, X, Y, num_iterations=2000, learning_rate=0.1, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
    
    Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation (≈ 1-4 lines of code)
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        # Record the costs
        if  i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost)) 
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
    z = np.dot(w.T, X) + b
    yhat = lr_utils.basic_sigmoid(z) 
    
    yhat = [ np.around(yhat[0,i]) for i in range(yhat.shape[1]) ]  #四舍五入，yhat>0.5为1，<=0.5为0
    yhat = np.array(yhat).reshape((1,-1))
    return yhat
    
    
def accuracy(y, yhat):
    rightNum = np.dot(y,yhat.T) + np.dot(1-y,(1-y).T)
    accu = float(rightNum) / y.shape[1]
    return accu

#合并所有函数 进行模型训练
def model(X_train, y_train, X_test, y_test, num_iterations = 2000, learning_rate = 0.1, print_cost = False):
    #初始化参数
    dim = X_train.shape[0] #样本特征数目
    w, b = initialize_with_zeros(dim)
    #利用训练集 优化参数
    params, grads, costs = optimize(w, b, X_train, y_train, num_iterations, learning_rate, print_cost)
#    print('参数： ',params,'\n代价函数值：',cost)
    
    yhat_train = predict(params['w'], params['b'], X_train)
    yhat_test = predict(params['w'], params['b'], X_test)
    #模型评估：训练集准确率，测试集准确率
    train_accuracy = accuracy(y_train, yhat_train)
    test_accuracy = accuracy(y_test, yhat_test)
    print('训练集准确率：%f%%'%train_accuracy, ',  测试集准确率: %f%%'%test_accuracy)
    
    d = {"costs": costs,
         "yhat_train" : yhat_train, 
         "yhat_test": yhat_test, 
         "w" : params['w'], 
         "b" : params['b'],
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

def plot_cost_iters(costs,learning_rate, num_iterations = 1000):
    '''
    学习率固定，绘制迭代次数和代价值关系图
    '''
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations ')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
def plot_cost_learn(X_train, y_train, X_test, y_test, learning_rates, num_iterations = 1000, print_cost = False):
    '''
    迭代次数固定，绘制不同的学习率下的代价函数对应值
    '''
    for i in learning_rates:
        print ("learning rate is: " + str(i))
        modelRes = model(X_train, y_train, X_test, y_test, num_iterations, i, print_cost)
        print ('\n' + "-------------------------------------------------------" + '\n')
        #绘制图
        plt.plot(modelRes["costs"], label= str(modelRes["learning_rate"]) )
    
    plt.ylabel('cost')
    plt.xlabel('iterations')
    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    
def imageTest(w, b, my_image):
    
    from scipy import ndimage
    import scipy

#    my_image = "cat_in_iran.jpg"   # change this to the name of your image file 
    
    # We preprocess the image to fit your algorithm.
    fname = "images/" + my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T #my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1, num_px*num_px*3)).T
    my_predicted_image = predict(w, b, my_image)

    plt.imshow(image)
    print ( "预测结果为 : ",np.squeeze(my_predicted_image))
    
if __name__ == '__main__':
    #加载数据
    train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
    print(train_set_x.shape)
    print(train_set_y.shape)
    print(test_set_x.shape)
    print(test_set_y.shape)
    print(classes.shape)
    print(classes)#数据类型为 numpy.bytes_      decode('utf-8') 将字节数据解码为utf-8
                  # 1代表有猫，0代表无猫
    
    #可视化图片数据
    index = 99
    plt.imshow(train_set_x[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode('utf-8')+  "' picture.")  
    index = 102
    plt.imshow(train_set_x[index])
    print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode('utf-8')+  "' picture.")
    
    ##图像数据向量化
    train_set_x_flatten = image2vector(train_set_x)#行 代表特征数， 列 代表样本数
    test_set_x_flatten = image2vector(test_set_x)
    
    #参数初始化
    dim = train_set_x_flatten.shape[0]
    w, b = initialize_with_zeros(dim)
    print ("w = " + str(w))
    print ("b = " + str(b))
   
    #计算对应参数梯度值
    grads, cost = propagate(w, b, train_set_x_flatten, train_set_y)
    print ("dw = " + str(grads["dw"]))
    print ("db = " + str(grads["db"]))
    print ("cost = " + str(cost))
    
    #梯度下降法优化参数
    X_train = train_set_x_flatten/255    #数据标准化
    y_train = train_set_y
    X_test = test_set_x_flatten/255    #数据标准化
    y_test = test_set_y
    
    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim) #参数初始化
    params, grads, cost = optimize(w, b, X_train, y_train, num_iterations=1000, learning_rate=0.005, print_cost = False)#参数优化
    
    #模型训练
    res = model(X_train, y_train, X_test, y_test, num_iterations =5000, learning_rate = 0.0005, print_cost = True)
    #抽样查看
    index = 6
    plt.imshow(test_set_x[index])
    print ("y = " + str(test_set_y[0,index]) + ", yhat = ",str(res['yhat_train'][0,index]))
    
    #绘制在同一学习率下的不同迭代次数，对应的代价值
    plot_cost_iters(res['costs'],res['learning_rate'],res['num_iterations'])
    #固定迭代次数，学习速率和代价函数之间的关系图
    plot_cost_learn(X_train, y_train, X_test, y_test, learning_rates=[0.01, 0.005, 0.001, 0.0005, 0.0001], num_iterations = 1500, print_cost = False)
        
    #选择一张自己的图片进行预测
    w = res['w']; b = res['b']
    my_image = 'cat_in_iran.jpg'
    imageTest(w, b, my_image)
    
    my_image = 'my_image2.jpg'
    imageTest(w, b, my_image)
