# -*- coding: utf-8 -*-
"""
建立深层神经网络

"""

import numpy as np
import matplotlib as mlt
from matplotlib import pyplot as plt
import os 
os.chdir(r'C:\Users\Administrator\Desktop\DL\3.深层神经网络\代码整理')
import dnn_utils as utils


layers_dims = [12288, 20, 7, 5, 1]#4 layers nn

#深层神经网络 初始化参数,并返回参数字典{W1,b1,W2,b2,....WL,bL}
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)         # number of layers in the network,
                                #这里包含了输入层为第1层，实际需要计算参数的只有L-1层，即这里是L-1层神经网络
    for i in range(1,L):#计算L-1层神经网络参数
        parameters['W'+str(i)] = np.random.rand(layer_dims[i], layer_dims[i-1]) / np.sqrt(layer_dims[i-1]) #*0.01
        parameters['b'+str(i)] = np.zeros( (layer_dims[i],1) )
        
    return parameters


#正向传播，计算Z,A,最后计算损失函数
def linear_forward(A, W, b):
    '''
    #1.返回正向传播中线性部分运算Z,及其计算所需的中间值(W,A,b)
    '''
    #计算中间值Z和A
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b) #Z的中间缓存变量 
    return Z,linear_cache

def linear_activation_forward(A_prev, W, b, activation):
    '''
    #2..正向传播计算激活函数，并保存计算中用到中间变量，
    包括：线性部分Z的中间变量缓存(A_prev,W,b)，和激活函数计算的中间变量缓存Z 
    '''
    #线性部分计算
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    #激活函数计算
    if activation == "relu":
        A, active_cache  = utils.relu(Z)
    elif activation == "sigmoid":
        A, active_cache = utils.sigmoid(Z)
    cache = (linear_cache,active_cache)
    return A, cache

# 深度神经网络的正向传播    
def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep() 
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = [] #用于存放Zi,Ai计算时的中间缓存变量
    A = X
    L = len(parameters) // 2 # L为神经网络的层数
    
    for i in range(1, L): #前L-1层激活函数使用relu
        A_prev = A #不断更新每一层的A值
        W = parameters['W'+str(i)]
        b = parameters['b'+str(i)]
        A, cache = linear_activation_forward(A_prev, W, b, 'relu' )
        caches.append(cache) #缓存每一步的中间变量
        
    # 最后输出层L 层用sigmoid激活函数
    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, cache = linear_activation_forward(A, W, b, activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))  #最终AL为预测结果，维度(输出层，样本个数)
            
    return AL, caches

# 利用正向传播计算的A值，计算损失函数
def compute_cost(A, Y):
    '''
    3.对输出值计算损失函数
    '''
    m = Y.shape[1] #样本个数
    cost = -1/m * ( np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T) )
    return cost

#反向传播获取参数梯度
def linear_backward(dZ,cache):
    '''
    计算dw,db,dA_prev
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1] ##############?????????????????????

    dW = 1/m *np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ,axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db
       
def linear_activation_backward(dA, cache, activation):
    linear_cache,active_cache = cache
    #计算dZ
    if activation == 'relu':
        dZ = utils.relu_backward(dA, cache)
    elif activation == 'sigmoid':
        dZ = utils.sigmoid_backward(dA, cache)
    #由dZ计算dW,db,dA_prev
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, cache):
    '''
    L层深层神经网络反向传播，计算各层参数梯度
    除了L层以外，其他都是迭代循环
    '''
    grads = {}
    L = len(caches)
#    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
#    cost = -1/m(np.dot(Y,log(AL).T) + np.dot(1-Y, log(1-A).T)) 
    dAL = - (np.divide(Y,AL) - np.divide(1-Y,1-AL))
    current_layer_cache = cache[L-1]
    grads['dA'+ str(L)], grads['dW'+ str(L)], grads['db'+ str(L)] = linear_activation_backward(dAL, current_layer_cache, 'sigmoid')
    
    for i in reversed(range(L-1)):
        dA = grads['dA'+ str(i+2)]
        current_layer_cache =  cache[i]
        grads['dA'+ str(i+1)], grads['dW'+ str(i+1)], grads['db'+ str(i+1)] = linear_activation_backward(dA, current_layer_cache, 'relu')
    
    return  grads

    
    
    
#梯度下降法更新参数

#建立神经网路模型(综合上述函数)：正向传播更新损失，反向传播计算参数梯度 更新参数, 正反向传播循环迭代，直至满足指定要求


if __name__ == '__main__':   
    train_x_orig, train_y, test_x_orig, test_y, classes = utils.load_data()
    print(train_x_orig.shape)
    print(train_y.shape)
    print(test_x_orig.shape)
    print(test_y.shape)
    print(classes)
    # Reshape the training and test examples 
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    
    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))
    layer_dims = [12288, 20, 7, 5, 1]
    parameters = initialize_parameters_deep(layer_dims)
    
    #正向传播迭代一次
    AL, caches = L_model_forward(train_x_flatten, parameters)
    
    L_model_back_ward(AL, train_y, caches)
