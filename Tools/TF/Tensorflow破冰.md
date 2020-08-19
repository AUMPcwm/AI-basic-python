# Tensorflow破冰

> 张子豪 2019-11-29

[TOC]

# 安装Tensorflow

# Hello World

```python
# 导入tensorflow，起个小名叫tf
import tensorflow as tf

# 创建一个Tensor对象，赋值给hello_constant
hello_constant = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
```

## Tensor——张量

在Tensorflow中，数据不是以整数、浮点数、字符串形式存储的，而是被封装在Tensor对象中。

Tensor可以有不同维度：

```python
# A是一个零维的int32类型的张量，零维对应标量
A = tf.constant(1234) 

# B是一个一维的int32类型的张量，一维对应向量
B = tf.constant([123,456,789]) 

# C是一个二维的int32类型的张量，二维对应矩阵
C = tf.constant([ [123,456,789], [222,333,444] ])
```

`tf.constant`是Tensorflow运算之一，`tf.constant`返回的tensor是常量tensor，值不会变。

## Session——会话

Tensorflow的API构建在计算图（ computational graph ）概念上，这是对复杂的数学运算可视化的一种方法，我们可以将刚刚的Hello World第一行代码可视化成一个计算图。

 ![img](file:///D:/02019%E7%A7%8B%E5%AD%A3%E5%AD%A6%E6%9C%9F/1124/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%20v2.0.0/Part%2008-Module%2001-Lesson%2008_TensorFlow/img/session.png)

Session的作用是分配GPU和CPU资源，真正让计算图“流动”起来。

开启会话通常使用Python中的会话管理器：

 ```python
with tf.Session() as sess:
    output = sess.run(hello_constant)
 ```

在session里对tensor求值。

 这段代码用 [`tf.Session`](https://www.tensorflow.org/api_docs/python/tf/Session) 创建了一个 `sess` 的 session 实例。然后 [`sess.run()`](https://www.tensorflow.org/api_docs/python/tf/Session#run) 函数对 tensor 求值，并返回结果。 



# Tensorflow基本操作

## 向占位符传入数据

刚刚我们向session传入一个常量tensor并求值，返回结果，那如果是变量应该怎么办？

使用`tf.placeholder()`作为占位符，也就是变量。

然后在运行会话的时候用`feed_dict={}`传入实际的数据。

例子：

```python
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
```

也可以用`feed_dict={}`参数设置多个tensor。

```python
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
```

> 注意：feed_dict中传入的数据类型必须与之前规定的tensor类型相符，否则会报错` ValueError: invalid literal for....`

## 数学运算

官方文档： https://tensorflow.google.cn/api_docs/python/tf/math 

```python
# 加法
x = tf.add(5, 2)  # 7

# 减法
x = tf.subtract(10, 4) # 6

# 乘法
y = tf.multiply(2, 5)  # 10

# 除法 
z = tf.divide(6,2)  # 3

# 矩阵乘法
c = tf.matmul(a,b) # a,b是矩阵
```

> 注意，传入的两个参数必须是同一种张量类型。
>
> `tf.subtract(tf.constant(2.0),tf.constant(1))`会报错：
>
> `ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32: `。
>
> 因为常量2.0是浮点数，常量1是常数，`tf.subtract()`需要传入的参数类型匹配。
>
> 可以把2.0转换成整数再相减。
>
> ```python
> tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))   # 1
> ```

练习：将原生python代码重构为TensorFlow张量计算代码：

```python
x = 10
y = 2
z = x/y - 1
print(z)
```

答案：

```python
x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1), tf.float64))

with tf.Session() as sess:
    output = sess.run(z)
    print(output)
```

# 线性函数

在深度学习的神经网络中，最常见的操作就是线性加权求和，如下图所示：

 ![img](file:///D:/02019%E7%A7%8B%E5%AD%A3%E5%AD%A6%E6%9C%9F/1124/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%20v2.0.0/Part%2008-Module%2001-Lesson%2008_TensorFlow/img/linear-equation.gif) 

W是连接两层神经网络的权重矩阵。b是偏置项。

W和b都是在训练过程中，通过梯度下降不断迭代更新的数。

但我们之前学到的`tf.placeholder()和tf.constant()`只能传入固定的值，不是真正的`变量`，这个时候就需要`tf.Variable`了。

## tf.Variable()

```python
x = tf.Variable(5)
```

像原生python里的变量一样，`tf.Variable`创建的张量值可以改变。

我们需要使用`` tf.global_variables_initializer()`初始化所有可变的tensor。

在session中调用它，就会在 graph 中初始化所有的 TensorFlow 变量 。

官方文档：https://www.tensorflow.org/api_docs/python/tf/global_variables_initializer

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
```

## tf.truncated_normal()

神经网络的权重一般是正态分布随机初始化的，可以用 `tf.truncated_normal()`

```python
# 初始化权重矩阵，输入维度120，输出维度5
weights = tf.Variable(tf.truncated_normal((120, 5)))
```

官方文档：https://www.tensorflow.org/api_docs/python/tf/truncated_normal

偏置项b一般初始化为0。

## tf.zeros()

```python
# 输出层的5个神经元，每个神经元都有一个偏置项，总共有5个偏置项
bias = tf.Variable(tf.zeros(5))
```

# 实战：Tensorflow实现线性回归

见代码

# Tensorflow神经网络基础

## softmax

![image-20191129214753119](C:\Users\40743\AppData\Roaming\Typora\typora-user-images\image-20191129214753119.png)

```python
x = tf.nn.softmax([2.0, 1.0, 0.2])
```

代码实例：

```python
logit_data = [2.0, 1.0, 0.1]

logits = tf.placeholder(tf.float32)
softmax = tf.nn.softmax(logits)

with tf.Session() as sess:
    output = sess.run(softmax, feed_dict={logits: logit_data})
    print(output)
```

运行结果：

```python
[0.6590012  0.24243298 0.09856589]
```



# 实战：Tensorflow搭建线性分类器-MNIST数据集图像分类

见代码文件。

