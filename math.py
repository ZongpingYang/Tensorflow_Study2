# -*- coding: utf-8 -*-
"""============================================
@date     :      2021/4/14   15:40
@author   :      Yangzp
@file     :      math.py
@IDE      :      PyCharm 
============================================"""

import tensorflow as tf


#1
#element-wise
#元素运算  =-*/

b = tf.fill([2,2],2.)
a = tf.ones([2,2])
print("a+b",a+b)
print("a-b",a-b)
print("a*b",a*b)
print("a/b",a/b)
print("b//a",b//a)  #求商
print("a%b",a%b)    #求余

print("log:",tf.math.log(a))
print("exp:",tf.exp(a))

print("b的三次方:",tf.pow(b,3))
print("b的三次方:",b**3)
print("b的开平方:",tf.sqrt(b))


#2
#matrix-wise
#矩阵运算(乘法) @ matmul
print("矩阵a,b相乘：",a@b)
print("矩阵a,b相乘：",tf.matmul(a,b))
c = tf.ones([4,2,3])
d = tf.fill([4,3,5],2.)
print("矩阵c,d相乘：",c@d)
#利用broadcast 将无法相乘的矩阵 变得可以相乘
e = tf.ones([4,2,3])
f = tf.fill([3,5],2.)
ff = tf.broadcast_to(f,[4,3,5])
print("矩阵e,f broadcast to ff 相乘：",e@ff)
#3
#dim-wise
#维度运算 reduce_mean    max  min  sum


