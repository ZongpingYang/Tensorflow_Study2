#标量 scalar
#向量 vector
#矩阵 matrix
#tensor : rank >2    所有数据都可叫做tensor

# int float double
#bool   0 1
#string
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import  tensorflow as tf
import  numpy as np
"""
a = tf.constant(1,dtype = tf.int32 )  #constant  常量
print(a)
print(a.device)

with tf.device("cpu"):  #cpu上的tensor只能使用CPU上的操作
    b = tf.range(4)
print(b.device)
print(b)
with tf.device("gpu"):  #Gpu上的tensor只能使用gPU上的操作
    c = tf.range(4)
print(c.device)           #！！！！ 无法将cup上的数据和GPU上的数据混着操作
print(c)

b = b.numpy()   #Tensor类型转换为numpy类型
print(b)
print(b.shape)

print(tf.rank(b))

print(tf.is_tensor(b))   #判断数据是否为tensor
"""
#数据类型转换
a = np.arange(5)
print(a.dtype)
aa =  tf.convert_to_tensor(a)  #将a转换为tensor
print(aa.dtype)
tf.cast(aa,dtype=tf.float32)  #强制类型转换  int double float  bool

b = tf.constant([5,0])            #转换为bool行，非0即为1,true
bb = tf.cast(b,dtype=tf.bool)
print(bb)

bbb = tf.cast(bb,dtype=tf.int32)  # bool转换为整型 true转换为1
print(bbb)

# tf.Variable  类型  具有可求导属性

# tensor  to  numpy

c = tf.ones([])
print(c)
c = c.numpy()
print(c)



