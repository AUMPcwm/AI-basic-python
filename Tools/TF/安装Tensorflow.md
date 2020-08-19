# 安装Tensorflow和Keras



# 第一步：pip换源

打开命令行，运行这条命令

```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```



# 第二步：安装

在命令行输入以下两条命令，安装Tensorflow的CPU版本和keras

```shell
pip install tensorflow

pip install keras
```

# 验证安装成功

```python
import tensorflow as tf
tf.__version__
```

![image-20191129172205333](C:\Users\40743\AppData\Roaming\Typora\typora-user-images\image-20191129172205333.png)