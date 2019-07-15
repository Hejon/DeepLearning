import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
import os
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
np.random.seed(1)

def test1():
    y_hat = tf.constant(36, name='y_hat')  # Define y_hat constant. Set to 36.
    y = tf.constant(39, name='y')  # Define y. Set to 39

    loss = tf.Variable((y - y_hat) ** 2, name='loss')  # Create a variable for the loss

    init = tf.global_variables_initializer()  # When init is run later (session.run(init)),
    # the loss variable will be initialized and ready to be computed
    with tf.Session() as session:  # Create a session and print the output
        session.run(init)  # Initializes the variables
        print(session.run(loss))  # Prints the loss

def tf_build_model():
    '''
       tensorflow 跑程序的一般步骤:
       1. Create Tensors (variables) that are not yet executed/evaluated.
       2. Write operations between those Tensors.
       3. Initialize your Tensors.
       4. Create a Session.
       5. Run the Session. This will run the operations you'd written above.
    '''

    # example
    import tensorflow as tf

    w = tf.Variable([.3], tf.float32, name='w')
    b = tf.Variable([-.3], tf.float32, name='b')
    x = tf.placeholder(tf.float32, name='x')

    linear = w * x + b                         #2. Write operations between those Tensors.

    init = tf.global_variables_initializer()   #3. Initialize your Tensors.

    session = tf.Session()                     # 4. Create a Session.

    session.run(init)# # Initializes the variables
    print(session.run(linear, {x: [1, 2, 3, 4]}))  # 5. Run the Session. This will run the operations you'd written above.


def linear_function():
    """
    Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    X = tf.constant(np.random.rand(3, 1), name='X')
    W = tf.constant(np.random.rand(4, 3), name='W')
    b = tf.constant(np.random.rand(4, 1), name='b')

    Y = tf.add(tf.matmul(W,X), b)

    # Create the session using tf.Session() and run it with sess.run(...) on the variable you want to calculate
    sess = tf.Session()
    result = sess.run(Y)

    # close the session
    sess.close()

    return result

def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')

    sigmoid = tf.sigmoid(x)

    with tf.Session() as session:
        result = session.run(sigmoid, feed_dict={x: z})
    return result

def cost(logits, labels):
    """
        Computes the cost using the sigmoid cross entropy
        
        Arguments:
        logits -- vector containing z, output of the last linear unit (before the final sigmoid activation)
        labels -- vector of labels y (1 or 0)

        Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
        in the TensorFlow documentation. So logits will feed into z, and labels into y.
        
        Returns:
        cost -- runs the session of the cost (formula (2))
    """
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    sess = tf.Session()
    cost = sess.run(cost, feed_dict={z: logits, y: labels})
    sess.close()

    return cost

#Using one-hot encoding
#y进行编码处理
def one_hot_matrix(labels, C):
    """
       Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                        corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                        will be 1.

       Arguments:
       labels -- vector containing the labels
       C -- number of classes, the depth of the one hot dimension

       Returns:
       one_hot -- one hot matrix
    """
    C = tf.constant(C, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)

    session = tf.Session()
    one_hot = session.run(one_hot_matrix)
    session.close()

    return one_hot

#变量初始化
def ones(shape):
    """
        Creates an array of ones of dimension shape

        Arguments:
        shape -- shape of the array you want to create

        Returns:
        ones -- array containing only ones
    """
    ones = tf.ones(shape)

    session = tf.Session()
    ones = session.run(ones)

    #close the session compute 'ones'
    None

    return ones

if __name__  == '__main__':
    # tf_build_model()
    #
    # a = tf.constant(2)
    # b = tf.constant(10)
    # c = tf.multiply(a, b)
    # print(c) # 定义computation graph （计算图）
    #
    # sess = tf.Session()
    # print(sess.run(c))#remember to initialize your variables, create a session and run the operations inside the session
    #
    # #placeholder :变量可以动态定义，用feed_dict给出具体值
    # x = tf.placeholder(tf.int64, name='x')
    # print(sess.run(2*x, feed_dict={x: 3}))
    # sess.close()
    #
    # #linear-function
    # print(linear_function())

    #计算损失函数
    # logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
    # cost = cost(logits, np.array([0, 0, 1, 1]))
    # print(cost)

    # #y进行one-hot编码
    # labels = np.array([1, 2, 3, 0, 2, 1])
    # one_hot = one_hot_matrix(labels, C=4)
    # print(one_hot)

    #变量初始化为1
    print('ones = ', ones([3, 2]))
