# Pytorch安装教程



# 预备工作

Python3最新版本安装

# 第一步：给pip换源

打开命令行，运行这行命令

```shell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

# 第二步：访问Pytorch官网

 https://pytorch.org/get-started/locally/ 

![image-20191129171103696](C:\Users\40743\AppData\Roaming\Typora\typora-user-images\image-20191129171103696.png)

比如这里我选择的是稳定版、Windows、pip、python3.7，不使用CUDA。

复制最后一行生成的命令，

```shell
pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

# 验证安装成功

运行以下python脚本

```python
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
```

![image-20191129171724049](C:\Users\40743\AppData\Roaming\Typora\typora-user-images\image-20191129171724049.png)

