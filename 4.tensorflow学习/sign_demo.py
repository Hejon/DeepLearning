'''
利用tensorflow建立手势识别神经网络模型
'''
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
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1)

#1. 加载数据
def load_data_convert():
    '''
    加载数据，并对数据格式转换：X,Y数据格式变换，将图片数据(1080, 64, 64, 3)转成（1080，64*64*3）数组，Y进行one-hot编码

    :param X_train_flatten:
    :param Y_train_orig:
    :param X_test_flatten:
    :param Y_test_orig:
    :return:返回处理后的X，Y数据
    '''

    # 1. Loading the dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    # print(X_train_orig)
    print(type(X_train_orig))
    print('X_train: ', X_train_orig.shape)
    print('Y_train: ', Y_train_orig)

    # example of a picture
    index = 0
    plt.imshow(X_train_orig[index])
    plt.show()
    print("y = " + str(np.squeeze(Y_train_orig[:, index])))
    # Flatten the training and test images
    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T

    # Normalize image vectors 标准化处理
    X_train = X_train_flatten / 255.
    X_test = X_test_flatten / 255.
    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6)
    Y_test = convert_to_one_hot(Y_test_orig, 6)

    print("number of training examples = " + str(X_train.shape[1]))
    print("number of test examples = " + str(X_test.shape[1]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))

    return X_train, X_test, Y_train, Y_test


# 2. create_placeholders
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """
    X = tf.placeholder(tf.float32, shape=(n_x, None))
    Y = tf.placeholder(tf.float32, shape=(n_y, None))

    return X, Y

#3. 参数初始化
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    tf.set_random_seed(1)# so that your "random" numbers match ours 随机种子

    W1 = tf.get_variable('W1', [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

#4. 神经网络正向传播
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1) # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3

#5. 计算损失函数：这里因为是多分类，图像label只含一种类别，输出层激活函数用softmax
def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    return cost

#5. 神经网络反向传播
# Backward propagation & parameter updates
def backward_propagation(cost, learning_rate):
    '''
    This is where you become grateful to programming frameworks.
    All the backpropagation and the parameters update is taken care of in 1 line of code.
    It is very easy to incorporate this line in the model.

    After you compute the cost function. You will create an "`optimizer`" object. You have to call this object along with
    the cost when running the tf.session. When called, it will perform an optimization on the given cost with the chosen
    method and learning rate.

    For instance, for gradient descent the optimizer would be:
    ```python
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    ```

    To make the optimization you would do:
    ```python
       c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
    ```

    This computes the backpropagation by passing through the tensorflow graph in the reverse order. From cost to inputs.

    **Note** When coding, we often use `_` as a "throwaway" variable to store values that we won't need to use later.
     Here, `_` takes on the evaluated value of `optimizer`, which we don't need (and `c` takes the value of the `cost` variable).
    '''

#6.建立模型
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_x, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

def testModel(parameters, my_image = 'images/thumbs_up.jpg'):
    import scipy
    from PIL import Image
    from scipy import ndimage
    import numpy as np
    from matplotlib import pyplot as plt

    image = ndimage.imread(my_image, flatten=False)
    #重新设定指定像素
    image = np.array(image)
    my_image = scipy.misc.imresize(image, size=(64, 64))
    plt.imshow(my_image)
    plt.show()
    my_image = my_image.reshape((1, 64 * 64 * 3)).T
    X_test =  my_image/ 255.
    my_image_prediction = predict(X_test, parameters)
    print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))

if __name__ == '__main__':
    #1. 加载数据，并对数据格式变换
    X_train, X_test, Y_train, Y_test = load_data_convert()
    #
    # #2. X,Y creat placeholders
    # n_x = X_train.shape[0]
    # n_y = Y_train.shape[0]
    # X, Y = create_placeholders(n_x, n_y)
    # print('X = '+str(X))
    # print('Y = '+str(Y))

    #3.参数初始化
    # tf.reset_default_graph()#Add operations to the graph before calling run().
    # with tf.Session() as sess:
    #     parameters = initialize_parameters()
    #     print('W3: ', str(parameters['W3']))
    #
    # #4.正向传播神经网络
    # tf.reset_default_graph()  # Add operations to the graph before calling run().
    # with tf.Session() as sess:
    #     X, Y = create_placeholders(12288, 6)
    #     parameters = initialize_parameters()
    #     Z3 = forward_propagation(X, parameters)
    #     print("Z3 = " + str(Z3))

    # #5. cost-function计算  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    # tf.reset_default_graph()
    # with tf.Session() as sess:
    #     X, Y = create_placeholders(12288, 6)
    #     parameters = initialize_parameters()
    #     Z3 = forward_propagation(X, parameters)
    #     cost = compute_cost(Z3, Y)
    #     print("cost: "+str(cost))
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  # 梯度下降法
    #     print(optimizer)
    # #5. 反向传播神经网络
    # #6.建立模型
    parameters = model(X_train, Y_train, X_test, Y_test)

    #预测结果
    testModel(parameters, my_image='images/thumbs_up.jpg')
    testModel(parameters, my_image='images/1.jpg')