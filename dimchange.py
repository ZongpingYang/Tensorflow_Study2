import tensorflow as tf
import numpy as np

#tf.reshape(a,[])  a 为tensor 数据源   [] 目标维度    数据量不会变
# 可以变换成多种维度，保证数据量不变即可，不一定都有物理意义

a = tf.random.normal([4,28,28,3])
print("a:",a)

print("a.shape:",a.shape)
print("a.dim:",a.ndim)
aa = tf.reshape(a,[4,784,3])      #   784 =28*28    变换了维度，相当于对a换了一种理解方式
print("aa.shape:",aa.shape )
aaa = tf.reshape(a,[4,-1,3])     #   -1  只等有一个 ，系统自动计算数值 与aa等价

aaaa = tf.reshape(a,[4,28*28*3])  #等价于 tf.reshape(a,[4,-1])
print("aaaa.shape:",aaaa.shape)


#tf.transpose 转置
b = tf.random.normal((4,3,2,1))
print(b)

bb = tf.transpose(b)
print("aa.shape",bb.shape)

#指定某些维度进行转置
bbb = tf.transpose(b,perm=[0,1,3,2])  #将b的维度改变  perm 指定维度顺序
print("bbb.shape",bbb.shape)             #数据互通

#增加维度  axis 参数为正数在前面加轴，负数在后面加
#Expand dim
c = tf.random.normal([4,35,8])
cc = tf.expand_dims(c,axis=0,) #在第0轴前面加上一个轴
print("cc.shape",cc.shape)
ccc = tf.expand_dims(c ,axis=3) #再第三个轴前面加上一个轴
print("ccc.shape",ccc.shape)
cccc = tf.expand_dims(c,axis=-1) # 在最后一个轴后面加上一个轴
print("cccc.shape",cccc.shape)

#减少维度   可以去掉多维中的维度为一的  只能去掉为1 的维度
#Squeeze dim

d = tf.squeeze(tf.zeros([1,2,1,1,3]))
print("d.shape:",d.shape)

e = tf.zeros([1,2,1,3])
ee = tf.squeeze(e,axis=0)  # 由axis指定去掉某个轴
print("ee.shape:",ee.shape)