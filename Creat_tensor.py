#create tensor
#from numpy
#list
#zeros ones
#random    正态分布初始化
#constant
#Application

import numpy as np
import tensorflow as tf
import torch

# 1 from numpy ,list  to tensor
a = np.ones([2,3])
print("a:",a)
atoTf = tf.convert_to_tensor(a)
print("atoTf:",atoTf)

#zeros  ones
b = tf.zeros([2,3])   #参数为shape
print(b)

#zeros_like
#ones_like
c = tf.zeros_like(b)     #等价于 tf.zeros(b.shape)  创建一样shape 的矩阵
print("c:",c)

#fill
#创建矩阵并初始化为指定的数值
d = tf.fill([3,3],5) #初始化为5
print(d)

# 均匀分布  正太分布 初始化权值

#正态分布
e = tf.random.normal([5,5],mean=1,stddev=1)   #[]shape nean均值 stddev方差，默认为标准正态分布  0 1
print(e)

f = tf.random.truncated_normal([2,2],mean=0,stddev=1) #j截断正态分布 截取正太分布的一部分在进行采样
print(f)

# 均匀
g = tf.random.uniform([2,2],minval=3,maxval=6)   #3---6之间采样
print(g)
