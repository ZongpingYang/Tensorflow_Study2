#broadcast
#[a,b,c,d]   [d]     两个tensor维度不同 一个tensor左边的维度为大维度 ，又边的维度为小维度
#先将两个tensor又对齐 ，在再行维度的添加 ，扩充到相同的维度
# 小维度必须相同才能进行broadcast ,不同的话无法进行   最后面的维度必须相同
#1可进行扩充 别的不行
#维度少的必须和维度多的相同  1 除外 （可扩充）

#可节省内存空间

import tensorflow as tf
a = tf.random.normal([4,32,32,3])
aa = a + tf.random.normal([3])       #自动broadcast    隐式
print("aa.shape:",aa.shape)

aaa = a + tf.random.normal([4,1,1,1])
print("aaa.shape",aaa.shape)

#tf.broadcast_to

b = tf.random.normal([4,1,1,1])
c = tf.broadcast_to(b,a.shape)   #显式  将b broadcast to a.shape  将一个tensor braodcast to 某个维度
print("c.shape",c.shape)


