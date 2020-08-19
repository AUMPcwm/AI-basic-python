# SVM 支持向量机

标签（空格分隔）： 机器学习

---

### **直观解释**
SVM，Support Vector Machine，它是一种二分类模型，其基本模型定义为特征空间上的间隔最大的线性分类器，其学习策略便是间隔最大化，最终可转化为一个凸二次规划问题的求解。
这里涉及了几个概念，二分类模型，线性分类器，间隔最大化，凸二次规划问题。
> + 二分类模型：给定的各个样本数据分别属于两个类之一，而目标是确定新数据点将归属到哪个类中。
> + 线性分类器：分割样本点的分类器是一个超平面，这也就要求样本线性可分，这是hard-margin SVM的要求，对于后来的soft-margin SVM，放低为近似线性可分，再到后来的核技巧，要求映射到高维空间后要近似线性可分。
> + 线性可分：$D0$和$D1$是$n$维欧氏空间中的两个点集（点的集合）。如果存在 $n$维向量 $w$和实数$b$，使得所有属于$D0$的点 xi 都有 $wx_i+b>0$，而对于所有属于$D1$的点 $x_j$则有 $wx_j+b<0$。则我们称$D0$和$D1$线性可分。
> + 间隔最大化：首先要知道SVM中有函数间隔和几何间隔，函数间隔刻画样本点到超平面的相对距离，几何间隔刻画的是样本点到超平面的绝对距离，SVM的直观目的就是找到最小函数距离的样本点，然后最大化它的几何间隔。
> + 凸二次规划：目标函数是二次的，约束条件是线性的。

### **核心公式**
> + 线性可分训练集：$T=\left\{ \left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{n}, y_{n}\right)\right\}$

> + 学习得到的超平面：$w^{* T} x+b^{*}=0$

> + 相应的分类决策函数：$f(x)=\operatorname{sign}\left(w^{* T} x+b^{*}\right)$

> + SVM基本思想：间隔最大化，不仅要讲正负类样本分开，而且对最难分的点（离超平面最近的点）也要有足够大的确信度将他们分开。

### **函数间隔**

给定一个超平面$（w, b）$，定义该超平面关于样本点 $(x_i,y_i )$ 的函数间隔为：$\widehat{\gamma}{i}=y{i}\left(w^{T} x_{i}+b\right)$ 定义该超平面关于训练集$T$的函数间隔为：$\widehat{\gamma}=\min {i=1,2, \ldots, N} \widehat{\gamma}{i}$

### **几何间隔**

给定一个超平面$（w, b）$，定义该超平面关于样本点 $(x_i,y_i )$ 的几何间隔为：$\gamma_{i}=y_{i}\left(\frac{w^{T}}{|w|} x_{i}+\frac{b}{|w|}\right)$ 定义该超平面关于训练集$T$的几何间隔为：$\gamma=\min {i=1,2, \ldots, N} \gamma{i}$


### **函数间隔与几何间隔的关系**

$\begin{array}{c}{\gamma_{i}=\frac{\hat{\gamma}_{i}}{|w|}, i=1,2, \ldots, N} \ {\gamma=\frac{\hat{\gamma}}{|w|}}\end{array}$

### **间隔最大化**
求得一个几何间隔最大的分离超平面，可以表示为如下的最优化问题： $\begin{array}{c}{\max {w, b} \gamma} \ {\text {s.t.} y{i}\left(\frac{w^{T}}{|w|} x_{i}+\frac{b}{|w|}\right) \geq \gamma, i=1,2, \ldots, N}\end{array}$

考虑函数间隔与几何间隔的关系式，改写为：

$\begin{array}{c}{\max {w, b} \frac{\hat{\gamma}}{|w|}} \ {\text {s.t. } y{i}\left(w^{T} x_{i}+b\right) \geq \hat{\gamma}, i=1,2, \ldots, N}\end{array}$

等价与下式

$\begin{array}{c}{\max {w, b} \frac{1}{|w|}} \ {\text {s.t. } 1-y{i}\left(w^{T} x_{i}+b\right) \leq 0, i=1,2, \ldots, N}\end{array}$

注意到最大化$\frac{1}{|w|}$ 和最小化$\frac{1}{2}|w|^{2}$是等价的，故最优化问题可转化为：

$\begin{array}{c}{\min {w, b} \frac{1}{2}|w|^{2}} \ {\text {s.t. } 1-y{i}\left(w^{T} x_{i}+b\right) \leq 0, i=1,2, \ldots, N}\end{array}$

构造Lagrange函数： $\begin{aligned} L(w, b, \alpha)=& \frac{1}{2}|w|^{2}+\sum_{i=1}^{N} \alpha_{i}\left[1-y_{i}\left(w^{T} x_{i}+b\right)\right] \ \alpha_{i} & \geq 0, i=1,2, \ldots, N \end{aligned}$

令$\theta_{\alpha}(w, b)=\max {\alpha{i} \geq 0} L(w, b, \alpha)$

则有$\theta_{\alpha}(w, b)=\left\{\begin{array}{c}{\frac{1}{2}|w|^{2},当全部约束满足} \ {+\infty，当存在约束不满足}\end{array}\right\}.$

故原问题等价于 $\min \{w, b\} \theta{\alpha}(w, b)=\min _{w, b} \max {\alpha{i} \geq 0} L(w, b, \alpha)$

### **学习的对偶算法**
根据拉格朗日对偶性，上式的对偶问题为： $\min {w, b}  \theta{\alpha}(w, b)= \max {\alpha{i} \geq 0}\min _{w, b} L(w, b, \alpha)$
(1) 求 $\min _{w, b} L(w, b, \alpha)$
$\nabla_{w} L(w, b, \alpha)=w-\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}=0$

$\nabla_{b} L(w, b, \alpha)=-\sum_{i=1}^{N} \alpha_{i} y_{i}=0$

得

$w=\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}$

$\sum_{i=1}^{N} \alpha_{i} y_{i}=0$

将以上两式代入拉格朗日函数中消去，得 $\begin{aligned} L(w, b, \alpha) &=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left\langle x_{i}, x_{j}\right\rangle+\sum_{i=1}^{\mathrm{N}} \alpha_{i} \end{aligned}$

（2）求$\min _{w, b} L(w, b, \alpha)$求对$\alpha$的极大，即是对偶问题

$\begin{aligned} \max {\alpha} &-\frac{1}{2} \sum{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left\langle x_{i}, x_{j}\right\rangle+\sum_{i=1}^{\mathrm{N}} \alpha_{i} \ \text {s.t.} & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \ \alpha_{i} & \geq 0, i=1,2, \ldots, N \end{aligned}$

将极大改为极小，得

${\min {\alpha} \frac{1}{2} \sum{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left\langle x_{i}, x_{j}\right\rangle-\sum_{i=1}^{\mathrm{N}} \alpha_{i}}$

$\sum_{i=1}^{N} \alpha_{i} y_{i}=0$

$\alpha_{i} \geq 0, i=1,2, \ldots, N$

原问题的对偶问题： $\begin{aligned} \min {\alpha} & \frac{1}{2} \sum{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left\langle x_{i}, x_{j}\right\rangle-\sum_{i=1}^{\mathrm{N}} \alpha_{i} \ \text {s.t.} & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \ & \alpha_{i} \geq 0, i=1,2, \ldots, N \end{aligned}$

求解方法： （1）由于该问题为凸优化问题，故可直接求解。 （2）由于该问题与其原问题等价，其原问题满足Slater定理，故该问题的解与KKT条件为充分必要的关系，故只需找到一组解满足KKT条件，即找到了问题的解（充分性）。

关于对偶问题的解$\alpha^{}=\left(\alpha_{1}^{}, \alpha_{2}^{}, \ldots, \alpha_{N}^{}\right)$，是由SMO算法解出来的，这个结合加入松弛变量的情况再讲。

解出对偶问题的解$\alpha^{}=\left(\alpha_{1}^{}, \alpha_{2}^{}, \ldots, \alpha_{N}^{}\right)$后，怎么求原问题的解$w^{}, b^{}$？

由KKT条件可知，原问题和对偶问题均取到最优值的解$\left(w^{}, b^{}, \alpha^{*}\right)$需满足以下四个要求：

对原始变量梯度为0： $\nabla_{w} L\left(w^{}, b^{}, \alpha^{}\right)=w^{}-\sum_{i=1}^{N} \alpha_{i}^{} y_{i} x_{i}=0$ $\nabla_{b} L\left(w^{}, b^{}, \alpha^{}\right)=-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}=0$
原问题可行： $1-y_{i}\left(w^{* T} x_{i}+b^{*}\right) \leq 0, i=1,2, \ldots, N$
不等式约束乘子非负: $\alpha_{i}^{*} \geq 0, i=1,2, \ldots, N$
对偶互补松弛： $\alpha_{i}^{}\left[1-y_{i}\left(w^{ T} x_{i}+b^{*}\right)\right]=0, i=1,2, \dots, N$
由于1中 $\nabla_{w} L\left(w^{}, b^{}, \alpha^{}\right)=w^{}-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i} x_{i}=0$

得到 $w^{}=\sum_{i=1}^{N} \alpha_{i}^{} y_{i} x_{i}$ 这样$w^{*}$就求出来了

用反证法我们可以得到至少有一个$\alpha_{i}^{}>0$，假设所有的$\alpha_{i}^{}=0$，由$w^{}-\sum_{i=1}^{N} \alpha_{i}^{} y_{i} x_{i}=0$知道，$w^{}=0$，而$w^{}=0$显然不是原问题的解，我们要零解一点意义都没有。

接下来，求$b^{}$ 取$\alpha_{i}^{}$ 的一个分量满足$\alpha_{i}^{}>0$ ，则有对应的由4中的 $\alpha_{i}^{}\left[1-y_{i}\left(w^{* T} x_{i}+b^{}\right)\right]=0, i=1,2, \dots, N$，有$1-y_{j}\left(w^{ T} x_{j}+b^{*}\right)=0$

代入$w^{}$和样本点$(x_j,y_j)$，求出 $b^{}=y_{j}-\sum_{i=1}^{N} \alpha_{i}^{*} y_{i}\left\langle x_{i}, x_{j}\right\rangle$

这样超平面的两个参数$(w^{},b^{})$就都求出来了 $w^{}=\sum_{i=1}^{N} \alpha_{i}^{} y_{i} x_{i}$ $b^{}=y_{j}-\sum_{i=1}^{N} \alpha_{i}^{} y_{i}\left\langle x_{i}, x_{j}\right\rangle$

至于为什么SVM叫支持向量机，因为我们发现只有$\alpha_{i}^{}>0$时，对应的样本$(x_i,y_i)$才会对最终超平面的结果产生影响，此时$1-y_{i}\left(w^{ T} x_{i}+b^{*}\right)=0$， 也就是函数间隔为1，我们称这类样本为支持向量，所以这个模型被叫做支持向量机。支持向量的个数一般很少，所以支持向量机只有很少的“重要的”训练样本决定。


$T=\left\{\left(\Phi\left(x_{1}\right), y_{1}\right),\left(\Phi\left(x_{2}\right), y_{2}\right), \ldots,\left(\Phi\left(x_{n}\right), y_{n}\right)\right\}$

### **核技巧**
基本思想：找一个映射Φ（一般为高维映射），将样本点特征x映射到新的特征空间Φ(x)，使其在新的特征空间中线性可分（或近似线性可分），然后利用之前的SVM算法在新的特征空间中对样本进行分类。 
 输入训练集 
$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{n}, y_{n}\right)\right\}$ 
 其中$x_{i} \in R^{n}, y_{i} \in{-1,+1}$ （1）选择合适的映射函数Φ，将训练集?? 映射为 $T=\left\{\left(\Phi\left(x_{1}\right), y_{1}\right),\left(\Phi\left(x_{2}\right), y_{2}\right), \ldots,\left(\Phi\left(x_{n}\right), y_{n}\right)\right\}$ （2）选择惩罚参数C，构造并求解约束最优化问题（原问题的对偶问题） $\min_{\alpha} \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left\langle\Phi\left(x_{i}\right), \Phi\left(x_{j}\right)\right\rangle-\sum_{i=1}^{\mathrm{N}} \alpha_{i}$ $\begin{aligned} \text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \ & 0 \leq \alpha_{i} \leq C, i=1,2, \ldots, N \end{aligned}$ 求得最优解$\alpha^{}=\left(\alpha_{1}^{}, \alpha_{2}^{}, \ldots, \alpha_{N}^{}\right)^{T}$ （3）计算$W^{}, b^{}$: $w^{}=\sum_{i=1}^{N} \alpha_{i}^{} y_{i} \Phi\left(x_{i}\right)$ 选择$a^{}$的一个分量满足$0<\alpha_{i}^{}<C$，计算 $b^{}=y_{j}-\sum_{i=1}^{N} \alpha_{i}^{} y_{i}\left\langle\Phi\left(x_{i}\right), \Phi\left(x_{j}\right)\right\rangle$ （4）求得分离超平面和分类决策函数： $w^{* T} \Phi(x)+b^{}=0$ $f(x)=\operatorname{sign}\left(w^{ T} \Phi(x)+b^{}\right)=\operatorname{sign}\left(\sum_{i=1}^{N} \alpha_{i}^{} y_{i}\left\langle\Phi(x), \Phi\left(x_{i}\right)\right\rangle+ b^{*}\right)$

该算法的问题： （1）合适的映射函数??太难找，几乎找不到 （2）假设找到了映射函数??，由于将数据映射到高维，在高维空间中做运算，计算量太大（维数灾难）

改进： 考虑到算法中如果不需写出分离超平面，即不需写出$w^{?}$，而是直接用$f(x)=\operatorname{sign}\left(w^{* T} \Phi(x)+b^{}\right)=\operatorname{sign}\left(\alpha_{i}^{} y_{i}\left\langle\Phi(x), \Phi\left(x_{j}\right)\right\rangle+ b^{*}\right)$来做预测，同样可以给出分类边界以及达到预测目的。这样的话，算法中需要用到样本的地方全部以内积形式出现，如果我们能够找到一种函数，能够在低维空间中直接算出高维内积，并且该函数对应着某个映射，即解决了以上两个问题。

核函数的本质：用相似度函数重新定义内积运算。

什么样的函数可以作为核函数？ 核函数对应的Gram矩阵为半正定矩阵。

常用的核函数:

1. 线性核函数（linear kernel）$K(x, z)=x^{T} z$
2. 多项式核函数（polynomial kernel function）$K(x, z)=\left(\gamma x^{T} z+r\right)^{p}$
3. 高斯核函数（ Gaussian kernel function ） $K(x, z)=\exp \left(-\gamma|x-z|^{2}\right)$
4. Sigmoid核函数
5. 拉普拉斯核函数
6. 字符串核函数

### **算法十问**
1. SVM 为什么采用间隔最大化
> 当训练数据线性可分时，存在无穷个分离超平面可以将两类数据正确分开。感知机利用误分类最小策略，求得分离超平面，不过此时的解有无穷多个。线性可分支持向量机利用间隔最大化求得最优分离超平面，这时，解是唯一的。另一方面，此时的分隔超平面所产生的分类结果是最鲁棒的，对未知实例的泛化能力最强。可以借此机会阐述一下几何间隔以及函数间隔的关系。


2. 为什么要将求解 SVM 的原始问题转换为其对偶问题
> 一是对偶问题往往更易求解，当我们寻找约束存在时的最优点的时候，约束的存在虽然减小了需要搜寻的范围，但是却使问题变得更加复杂。为了使问题变得易于处理，我们的方法是把目标函数和约束全部融入一个新的函数，即拉格朗日函数，再通过这个函数来寻找最优点。二是可以自然引入核函数，进而推广到非线性分类问题。

3. 为什么 SVM 要引入核函数
> 当样本在原始空间线性不可分时，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分。而引入这样的映射后，所要求解的对偶问题的求解中，无需求解真正的映射函数，而只需要知道其核函数。核函数的定义：K(x,y)=<?(x),?(y)>，即在特征空间的内积等于它们在原始样本空间中通过核函数 K 计算的结果。一方面数据变成了高维空间中线性可分的数据，另一方面不需要求解具体的映射函数，只需要给定具体的核函数即可，这样使得求解的难度大大降低。

4. 为什么SVM对缺失数据敏感
> 这里说的缺失数据是指缺失某些特征数据，向量数据不完整。SVM 没有处理缺失值的策略。而 SVM 希望样本在特征空间中线性可分，所以特征空间的好坏对SVM的性能很重要。缺失特征数据将影响训练结果的好坏。

5. SVM 核函数之间的区别
> 一般选择线性核和高斯核，也就是线性核与 RBF 核。 线性核：主要用于线性可分的情形，参数少，速度快，对于一般数据，分类效果已经很理想了。 RBF 核：主要用于线性不可分的情形，参数多，分类结果非常依赖于参数。有很多人是通过训练数据的交叉验证来寻找合适的参数，不过这个过程比较耗时。 如果 Feature 的数量很大，跟样本数量差不多，这时候选用线性核的 SVM。 如果 Feature 的数量比较小，样本数量一般，不算大也不算小，选用高斯核的 SVM。

6. LR和SVM的联系与区别
> 联系： 1. LR和SVM都可以处理分类问题，且一般都用于处理线性二分类问题（在改进的情况下可以处理多分类问题） 2. 两个方法都可以增加不同的正则化项，如l1、l2等等。所以在很多实验中，两种算法的结果是很接近的。 区别：1、LR是参数模型，SVM是非参数模型。 3. 从目标函数来看，区别在于逻辑回归采用的是logistical loss，SVM采用的是hinge loss，这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。 4. SVM的处理方法是只考虑support vectors，也就是和分类最相关的少数点，去学习分类器。而逻辑回归通过非线性映射，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重。 5. 逻辑回归相对来说模型更简单，好理解，特别是大规模线性分类时比较方便。而SVM的理解和优化相对来说复杂一些，SVM转化为对偶问题后,分类只需要计算与少数几个支持向量的距离,这个在进行复杂核函数计算时优势很明显,能够大大简化模型和计算。 6. logic 能做的 svm能做，但可能在准确率上有问题，svm能做的logic有的做不了。

7. SVM的原理是什么？
> SVM是一种二类分类模型。它的基本模型是在特征空间中寻找间隔最大化的分离超平面的线性分类器。（间隔最大是它有别于感知机） （1）当训练样本线性可分时，通过硬间隔最大化，学习一个线性分类器，即线性可分支持向量机； （2）当训练数据近似线性可分时，引入松弛变量，通过软间隔最大化，学习一个线性分类器，即线性支持向量机； （3）当训练数据线性不可分时，通过使用核技巧及软间隔最大化，学习非线性支持向量机。 注：以上各SVM的数学推导应该熟悉：硬间隔最大化（几何间隔）---学习的对偶问题---软间隔最大化（引入松弛变量）---非线性支持向量机（核技巧）。

8. SVM如何处理多分类问题？
> 一般有两种做法：一种是直接法，直接在目标函数上修改，将多个分类面的参数求解合并到一个最优化问题里面。看似简单但是计算量却非常的大。 另外一种做法是间接法：对训练器进行组合。其中比较典型的有一对一，和一对多。 一对多，就是对每个类都训练出一个分类器，由svm是二分类，所以将此而分类器的两类设定为目标类为一类，其余类为另外一类。这样针对k个类可以训练出k个分类器，当有一个新的样本来的时候，用这k个分类器来测试，那个分类器的概率高，那么这个样本就属于哪一类。这种方法效果不太好，bias比较高。 svm一对一法（one-vs-one），针对任意两个类训练出一个分类器，如果有k类，一共训练出C(2,k) 个分类器，这样当有一个新的样本要来的时候，用这C(2,k) 个分类器来测试，每当被判定属于某一类的时候，该类就加一，最后票数最多的类别被认定为该样本的类。

### **面试真题**
1.核函数的选择就是svm中的难点，也是核心问题（举出不同的例子？）

2.函数间隔/几何间隔是什么，有什么意义？

3.精通svm，那你说一下svm中的难点是什么？以及你是怎么解决这个难点的？（熟悉的问法）

4.相关关键词：硬间隔最大化（几何间隔）、函数间隔、学习的对偶问题、软间隔最大化（引入松弛变量）、非线性支持向量机（核技巧）、Hinge Loss

5.怎么理解SVM的损失函数?

6.使用高斯核函数，请描述SVM的参数C和σ对分类器的影响

7.核函数是什么?高斯核映射到无穷维是怎么回事?

8.SVM和Logistic回归的异同？

9.SVM用于回归问题:SVR

10.SVM框架下引入Logistic函数:输出条件后验概率？

11.SVM可以用来划分多类别吗? 如果可以，要怎么实现？

12.此时的分隔超平面所产生的分类结果是最鲁棒的，对未知实例的泛化能力最强（WHY）？

13.高维一定线性可分？

14.一个核函数都隐式定义了一个成为“再生核希尔伯特空间”的特征空间(iff条件)？

15.感知机的对偶形式和SVM对偶形式的对比

16.为什么要用对偶形式？如何理解对偶函数的引入对计算带来的优势？



### **应用**
1. Sklearn 
> class sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, 
					coef0=0.0, shrinking=True, probability=False, tol=0.001, 
					cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
					decision_function_shape=’ovr’, random_state=None)



: 惩罚参数C。C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

kernel ： string，optional(default =‘rbf’)
核函数类型，str类型，默认为’rbf’。可选参数为：

’linear’：线性核函数
‘poly’：多项式核函数
‘rbf’：径像核函数/高斯核
‘sigmod’：sigmod核函数
‘precomputed’：核矩阵
precomputed表示自己提前计算好核函数矩阵

degree ： int，可选(默认= 3)
多项式核函数的阶数，int类型，可选参数，默认为3。这个参数只对多项式核函数有用，是指多项式核函数的阶数n，如果给的核函数参数是其他核函数，则会自动忽略该参数。

gamma ： float，optional(默认=‘auto’)
核函数系数，float类型，可选参数，默认为auto。只对’rbf’ ,’poly’ ,’sigmod’有效。如果gamma为auto，代表其值为样本特征数的倒数，即1/n_features。

coef0 ： float，optional(默认值= 0.0)
核函数中的独立项，float类型，可选参数，默认为0.0。只有对’poly’ 和,’sigmod’核函数有用，是指其中的参数c。


shrinking ： 布尔值，可选(默认= True)
是否采用启发式收缩方式，bool类型，可选参数，默认为True。


tol ： float，optional(默认值= 1e-3)
svm停止训练的误差精度，float类型，可选参数，默认为1e^-3。


cache_size ： float，可选（默认为200）
内存大小，float类型，可选参数，默认为200。指定训练所需要的内存，以MB为单位，默认为200MB。

class_weight ： {dict，‘balanced’}，可选
类别权重，dict类型或str类型，可选参数，默认为None。给每个类别分别设置不同的惩罚参数C，如果没有给，则会给所有类别都给C=1，即前面参数指出的参数C。如果给定参数’balance’，则使用y的值自动调整与输入数据中的类频率成反比的权重。

verbose ： bool，默认值：False
是否启用详细输出，bool类型，默认为False，此设置利用libsvm中的每个进程运行时设置，如果启用，可能无法在多线程上下文中正常工作。一般情况都设为False，不用管它。

max_iter ： int，optional(默认值= -1)
最大迭代次数，int类型，默认为-1，表示不限制。

decision_function_shape ： ‘ovo’，‘ovr’，默认=‘ovr’
决策函数类型，可选参数’ovo’和’ovr’，默认为’ovr’。’ovo’表示one vs one，’ovr’表示one vs rest。

random_state ： int，RandomState实例或None，可选(默认=无)
数据洗牌时的种子值，int类型，可选参数，默认为None。伪随机数发生器的种子,在混洗数据时用于概率估计。