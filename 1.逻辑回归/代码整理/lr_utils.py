# -*- coding: utf-8 -*-
"""
逻辑回归进行图像识别，工具包
1.计算sigmoid函数
2.计算sigmoid的导函数
3.图像数据转化成向量
4.代价函数计算，L1代价，L2代价
"""
import numpy as np

#sigmoid函数
def basic_sigmoid(x):
    '''
    Compute sigmoid of x.

    Arguments:
        x -- A scalar
    Return:
        s -- sigmoid(x)
    '''
    sigmoid = 1.0 / (1 + np.exp(-x))
    return sigmoid

#sigmoid的导函数
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    s = basic_sigmoid(x)
    ds = s * (1 - s)
    return ds


#图像数据向量化：长宽高分别是width、height和depth的三维图像转化为二维向量
#image2vector() that takes an input of shape (length, height, 3) and returns a vector of shape (length*height*3, 1)
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    
    return v

#L2范数归一化
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    
    # Divide x by its norm.
    x = x / x_norm

    return x

#利用Python的广播，计算softmax函数
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp / x_sum
    
    return s

#L1代价函数
def L1_loss(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(np.abs(y - yhat))
    return loss

#L2代价函数
def L2_loss(yhat, y):
#    loss = np.sum(np.power(yhat-y, 2))
    loss = np.dot(yhat-y, (yhat-y).T)
    return loss

if __name__ == '__main__':
    
    print(basic_sigmoid(3))
#    x = [1,2,3]
#    print(basic_sigmoid(x))  #list会报错，要转成数组
    x = np.array([1,2,3])
    print(basic_sigmoid(x))
    
    print(sigmoid_derivative(x)) #函数求导
    
    #图像向量化
    # This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
    image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],
       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],
       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
    print(image.shape)
    print ("image2vector(image) = " + str(image2vector(image)))
    
    
    #数据L2归一化处理
    x = np.array([[0,3,4],[1,6,4]])
    print(normalizeRows(x))
    
    ##python 广播运算，就算softmax 函数,每一行相加为1
    print(softmax(x))
    
    #计算代价函数
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print('L1 loss:',L1_loss(yhat,y))
    print('L2 loss:' ,L2_loss(yhat,y))