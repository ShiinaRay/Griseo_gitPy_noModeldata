# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow_core.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def imageprepare(testimg):
    #进行识别的图片
    image = Image.open(testimg)
    # plt.imshow(image)
    # plt.show()
    #把图像转换成784维的向量
    image = image.resize([28, 28])
    image = image.convert("L")
    tv = list(image.getdata())
    tv=[(255 - x) * 1.0 / 255.0 for x in tv]#需要确认一下是否与Minst数据的图片数值一致？？
    # print(tv)
    return tv


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def evaluate_one_image(testimg):
    tf.reset_default_graph()
    result = imageprepare(testimg)
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()  # 定义saver
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "./Lenet5Model/model.ckpt")  # 使用模型，参数和之前的代码保持一致
        prediction = tf.argmax(y_conv, 1)#(0.1, 0.2, 0.5, 0.03,.....)找出最大值是0.5，它的下标是2，按照Mnist数据的规定，下标就是对应实际标签。
        # for i in range(1,imagearray[])
        predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)
        # predint = prediction.eval(feed_dict={x: [result], keep_prob: 1.0}, session=sess)

    # one_hot_label = mnist.train.labels[1, :]  #识别图片的标签
    # print(mnist.train.labels)
    # print(one_hot_label)
    # print(predint)
    # label = np.argmax(one_hot_label)
    # if label == predint[0]:
    #     print('best match !')
    # else:
    #     print('error !')
    # print('图片原始标签： %d' % label)
    print('识别结果: %d' % predint[0])
    # return '识别结果: %d' % predint[0]
    return predint[0]

"""
主函数
"""
def main():
    # evaluate_one_image("./cut/8.png")#7
    # evaluate_one_image("./cut/9.png")#2
    # evaluate_one_image("./cut/10.png")#3
    evaluate_one_image("./cut/34.png")
    evaluate_one_image("./max_region/34.png")



if __name__ == '__main__':
    main()