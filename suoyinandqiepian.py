#索引和切片


import numpy as np
import tensorflow as tf
"""
#Basic index
a = tf.ones([1,5,5,3])
print("a:",a) #四维
print("a[0]:",a[0]) #三维
print("a[0][0]:",a[0][0]) #二维
print("a[0][0][0]:",a[0][0][0])  #一维
print("a[0][0][0][0]:",a[0][0][0][0]) #标量
"""
"""
# Numpy_style index
b = tf.random.normal([4,28,28,3])  #四张 28*28的彩色照片 RGB
#print("b:",b)
print("b[1]:",b[1].shape)  #取第2张照片
print("b[1,2]:",b[1,2].shape)  #取第二张照片的第三行
print("b[1,2,3]:",b[1,2,3].shape) #取第二张照片的第三行第四列
print("b[1,2,3,2]:",b[1,2,3,2].shape) #取第二张照片的第三行第四列第三个元素
"""
#[start ： end)切片   正：0 1 2 3 4 ....    反：-1 -2 -3
c = tf.range(10)
print(c)
print(c[1:5])
print(c[-2:9])
print(c[-2:-1])
print(c[:])  #只有冒号 表示当前维度的所有

#start：end：step（步长）
#::step
print(c[0:9:2])

#::-1  逆序 -2

#...表示没写的所有维度

# Select Index  可进行任意采样   指定一个维度index
#tf.gather(a, axis,indices[])  收集  a 数据源 axis维度 indices 收集的具体数据，有顺序
dd = tf.random.normal([4,35,8])
d = tf.gather(dd,axis=1,indices=[1,3,5,7])
print(d.shape)


# tf.gather_nd(a,[]) 可以指定多个维度index   a数据源  []索引号 多维度联合索引  可以嵌套[[],[],[]] 每一个[]表示选出来的数据。之后进行重新组合

e = tf.gather_nd(dd,[0])
print(e.shape)

f = tf.gather_nd(dd,[0,1])
print(f.shape)


#tf.boolean_mask(a, mask,axis)
# a数据源
# mask 维度选择
# axis 轴   mask维度 与轴保持一致



