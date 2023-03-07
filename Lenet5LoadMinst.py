# -*-coding:utf-8-*-
from tensorflow_core.examples.tutorials.mnist import input_data
import scipy.misc
import os
from matplotlib import image


def LoadMinst():
    #读取MNIST数据集 。如果不存在就事先下载
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    #把原始图像保存在MNIST_data/raw/文件夹下
    #如果没有这个文件夹，会自动创建
    save_dir = 'MNIST_data/raw/'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    #保存前20张照片
    for i in range(20):
        #请注意,mnist.train.images[i, ;]
        image_array = mnist.train.images[i, :]
        # Tensoflow 中的MNIST图片是一个784维的向量，我们重新把它还原为28X28维的图像
        image_array = image_array.reshape(28, 28)
        #保存文件的格式：
        filename = save_dir + 'mnist_train_%d.jpg' % i
        #先用scipy.misc.toimage转换为图像,再调用save直接保存
        # scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
        # image.imsave(filename, image_array, cmap='gray')
        image.imsave(filename,image_array,cmap='gray')  # cmap常用于改变绘制风格，如黑白gray，翠绿色virdidis

"""
主函数
"""
def main():
    LoadMinst()
if __name__ == '__main__':
    main()