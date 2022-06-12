---
layout: post
comments: false
title: "Mathematics小知识"
date: 2022-05-30 01:09:00

---

> This post of various machine learning knowledge.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## 1. Fisher Information（费雪信息）的直观意义

Fisher Information的定义：

假设观察到的i.i.d的数据$$X_1, \cdots, X_n$$服从一个概率分布$$f(X;\theta)$$，$$\theta$$是要求的参数，是个标量。那么似然函数（likelihood）就是

$$L(X; \theta) = \Pi_{i=1}^n f(X_i; \theta)$$

为了通过maximum likelihood estimation来解$$\theta$$，需要让log likelihood对于$$\theta$$的一阶导数等于0，然后得到$$\hat \theta_{MLE}$$。

这个log likelihood的一阶导数也叫做score function，

$$S(X;\theta) = \Sigma_{i=1}^{n} \frac{\partial log f(X; \theta)}{\partial \theta}$$

那么Fisher Information，用$$I(\theta)$$表示，的定义就是这个score function的二阶矩（second moment）：

$$I(\theta) = E\left[S(X;\theta)^2\right]$$

一般情况下，很容易证明，$$E\left[S(X;\theta)\right] = 0$$，从而得到

$$I(\theta) = E\left[S(X;\theta)^2\right] - E\left[S(X;\theta)\right]^2 = Var\left[S(X;\theta)\right]$$

于是得到了Fisher Information的第一个数学意义：估计计算MLE的方程的方差。直观表述就是，随着收集的数据越来越多，这个方差由于是independent sum的形式，就会变得越来越大，也就象征得到的信息越来越多。

如果log likelihood二阶可导，在一般情况下可以证明：

$$E\left[S(X;\theta)^2\right] = -E\left[\frac{\partial^2}{\partial \theta^2} log L(X;\theta)\right]$$

于是就有了Fisher Information的第二个数学意义：log likelihood在某个参数$\theta$处的负二阶导数的期望。下面对其进行解释。

一个normalized Bernoulli log likelihood的函数图像如下图所示：

![fisher]({{ '/assets/images/FISHER-1.png' | relative_url }})
{: style="width: 600px; max-width: 100%;"}

对于这样一个log likelihood function，它越平越宽，就代表我们对参数的估计能力越差，它越高越窄，就代表我们对参数的估计能力越好，也就是信息量越大。这个log likelihood function在MLE的结果$$\hat{\theta}$$处（也就是顶点处）的弯曲程度，就是通过在这个点处的负二阶导数来表示的。弯曲程度越大，整个log likelihood的形状就越偏向于高而窄，也就代表掌握的信息越多。

在一般情况下，通过对score function在真实值处的泰勒展开，应用中心极限定理，弱大数定律，依照概率一致收敛，以及Slutsky定理，可以证明MLE的渐进分布的方差是$$I^{-1}(\theta)$$，即$$Var(\hat \theta_{{MLE}}) = I^{-1}(\theta)$$，这也就是Fisher Information的第三条数学意义。这样说不严谨，严格地说，是$$\sqrt{n}(\hat \theta_{MLE} - \theta) \rightarrow N(0, I^{\ast}(\theta)^{-1})$$，这里$$I^{\ast}(\theta)$$是当只观察到一个$$X$$值时候的Fisher Information，当有$$n$$个i.i.d.观察值时，$$I^{\ast}(\theta) = I(\theta)/n$$。这个的直观解释就是，Fisher Information反映了我们对参数估计的准确度，它越大，对参数估计越准确，也即代表了更多的信息。


### 参考文献
* https://www.zhihu.com/question/26561604/answer/33275982


## 2. 矩阵特征向量和特征值的意义

一般来说，长期趋势往往体现的是内禀的性质。

长远来看，矩阵的特征值表示矩阵的缩放比。

如果要计算矩阵$$A^{100}$$，矩阵的乘法计算量大，那能否利用一个标量$$\lambda$$来代替矩阵，只计算$$\lambda^{100}$$就行呢？在很多应用里，比如说马尔可夫随机场模型里经常要计算矩阵的高阶乘法，比如说$$A^{100}$$。

我们通过斐波那契数列和黄金分割比来说明特征值和特征向量的意义。

斐波那契数列的表达式为：$$F_n = F_{n-1} + F_{n-2}$$。写成矩阵的形式：

$${\left[ \begin{array}{cc} F_n \\ F_{n-1} \end{array} \right]} = {\left[ \begin{array}{cc} 1 & 1 \\ 1 & 0 \end{array} \right]} {\left[ \begin{array}{c} F_{n-1} \\ F_{n-2} \end{array} \right]}$$

我们将利用特征值来计算斐波那契额数列的通项公式，从而说明矩阵特征值能体现矩阵长期趋势的原因。

定义：$$u_n = {\left[ \begin{array}{c} F_n \\ F_{n-1} \end{array} \right]}$$，$$A = {\left[ \begin{array}{cc} 1 & 1 \\ 1 & 0 \end{array} \right]}$$，那么

$$u_n = Au_{n-1}$$

初始条件：$$u_0 = {\left[ \begin{array}{ccc} 1 \\ 0 \end{array} \right]}$$

$$u_1 = Au_0$$

$$u_2 = Au_1$$

$$\cdots$$

$$u_n = A^nu_o$$

因为矩阵$$A$$是满秩的，所以其可以进行eigenvalue decomposition，也就是说$$A = S\Lambda S^{-1}$$，其中$$\Lambda$$是对角矩阵，对角线上的值是可以重复出现的特征值，而$$S$$的每一列则是对应特征值的特征向量，这些特征向量是线性无关的（因为矩阵满秩，所以做得到），$$S$$是orthonormal矩阵。

从而我们利用特征值和特征向量来求$$A^n$$：

$$A^2 = S \Lambda S^{-1} S \Lambda S^{-1} = S \Lambda^2 S^{-1}$$

从而我们可以知道：

$$A^k = S \Lambda^k S^{-1}$$

对所有的正整数$$k$$都成立。

我们来求一下$$\Lambda$$。令$$det \left |\lambda I - A \right | = 0$$，得到

$$\lambda_1 = \frac{1+\sqrt{5}}{2}, \lambda_2 = \frac{1-\sqrt{5}}{2}$$

如果我们将相邻的两个斐波那契数列的项分别作为一个点的$$x,y$$坐标，如上面所示。我们可以知道相邻两个点就可以用乘以矩阵$$A$$得到。我们对于这些点做线性回归，结果是斜率约等于$$\lambda_1$$。也就是说相邻的两个斐波那契额数相差$$\lambda_1$$倍，与黄金分割比相同。也就是说，矩阵最大的特征值，就是黄金分割比的值。

从这个例子可以直觉上理解：一个矩阵的$$n$$次方可以类比于这个矩阵最大的特征值的$$n$$次方。

更神奇的是，如果任意选取坐标$$(x,y)$$，在不断的乘以矩阵$$A$$，最终坐标仍然会收敛到最大的特征值对应的特征向量附近，而且当乘上的矩阵逼近于无穷时，会收敛到这个特征向量。也就是说，上面所得出的黄金分割比，与初始值无关，而是斐波那契数列的推导式的内禀性质，也就是矩阵$$A$$的内禀性质，也就是其最大的特征值。

我们计算矩阵$$A$$两个特征值对应的特征向量，可以得到$$\lambda_1$$对应的特征向量$$v_1 = \left[\lambda_1, 1 \right]^T$$，$$\lambda_2$$对应的特征向量$$v_2 = \left[\lambda_2, 1 \right]^T$$，再将$$v_1$$和$$v_2$$归一化，得到了$$v_1 = \left[\sqrt{\frac{5+\sqrt{5}}{10}}, \sqrt{\frac{5-\sqrt{5}}{10}}\right]^T = \sqrt{\frac{5-\sqrt{5}}{10}} \left[\lambda_1, 1 \right]^T $$，以及$$v_2 = \left[\sqrt{\frac{5-\sqrt{5}}{10}}， \sqrt{\frac{5+\sqrt{5}}{10}}\right]^T = \sqrt{\frac{5+\sqrt{5}}{10}} \left[\lambda_2, 1 \right]^T$$。因为eigenvalue decomposition并不需要对应的eigenvector是归一化的，我们就直接使用$$v_1 = \left[\lambda_1, 1 \right]^T$$和$$v_2 = \left[\lambda_2, 1 \right]^T$$了。$$v_1，v_2$$分别是矩阵$$A$$的eigenvalue decomposition里的$$S$$的两列：

$$S = {\left[ \begin{array}{cc} \lambda_1 & \lambda_2 \\ 1 & 1 \end{array} \right]}$$

而$$S^{-1}$$则是

$$S^{-1} = \frac{1}{\sqrt{5}} {\left[ \begin{array}{cc} 1 & -\lambda_2 \\ -1 & \lambda_1 \end{array} \right]}$$


所以我们得到：

$$u_k = A^ku_0 = S \Lambda^k S^{-1}u_0 = \frac{1}{\sqrt{5}} S \Lambda^k {\left[ \begin{array}{cc} 1 & -\lambda_2 \\ -1 & \lambda_1 \end{array} \right]} u_0$$

$$ = \frac{1}{\sqrt{5}} {\left[ \begin{array}{cc} \lambda_1 & \lambda_2 \\ 1 & 1 \end{array} \right]} {\left[ \begin{array}{cc} \lambda_1 & 0 \\ 0 & \lambda_2 \end{array} \right]}^k {\left[ \begin{array}{c} 1 \\ 0 \end{array} \right]}$$

$$ = \frac{1}{\sqrt{5}} {\left[ \begin{array}{cc} \lambda_1 & \lambda_2 \\ 1 & 1 \end{array} \right]} {\left[ \begin{array}{cc} \lambda_1^k & 0 \\ 0 & \lambda_2^k \end{array} \right]} {\left[ \begin{array}{c} 1 \\ -1 \end{array} \right]}$$

$$ = \frac{1}{\sqrt{5}} {\left[ \begin{array}{cc} \lambda_1 & \lambda_2 \\ 1 & 1 \end{array} \right]} {\left[ \begin{array}{c} \lambda_1^k \\ -\lambda_2^k \end{array} \right]}$$

$$ = \frac{1}{\sqrt{5}} {\left[ \begin{array}{c} \lambda_1^{k+1} - \lambda_2^{k+1} \\ \lambda_1^{k}-\lambda_2^k \end{array} \right]}$$

而$$(\lambda_1^{k+1} - \lambda_2^{k+1}) / \lambda_1^{k}-\lambda_2^k \approx \lambda_1$$，也就是说，斐波那契数列前后两项的比值约等于黄金分割比。而在$$k$$趋近于无穷的时候，上述约等于则无限趋近于等于。


### 参考文献

* https://zhuanlan.zhihu.com/p/354102331


## 3. 信息熵公式的来由

信息熵定义为$$-\Sigma p_i log (p_i)$$并不是靠直觉得到，而是基于信息熵的三个基本规范要求，通过数学推导得到的唯一函数表达式。

**Lemma 1**

如果$$f(x)$$是$$x$$的单调增加函数，且对于任意给定的正整数$$m$$和$$n$$，$$f(mn) = f(m) + f(n)$$成立，则$$f(x)$$有唯一的函数形式：$$f(x) = K log(x)$$，其中$$K$$是一个正数。

信息熵$$H$$作为对随机试验不确定程度的度量，满足三个规则：

* $$H$$是概率$$p$$的连续函数：$$p$$的微小变动不会给$$H$$带来大的变化；
* 对于结果有$$n$$种可能的等概率的随机试验，$$H$$是$$n$$的单调递增函数：掷骰子$$n=6$$带来的不确定程度比扔硬币$$n=2$$要大；
* 组合可加性：把一个随机试验分成两个，未分之前的$$H$$是之后的$$H$$的加权和，形式化表达为：$$H_m(p_1,p_2,\cdots,p_m) = H_{m-1}((p_1+p_2), p_3,\cdots,p_m) + (p_1+p_2)H_2(\frac{p_1}{p_1+p_2}, \frac{p_2}{p_1+p_2})$$。

在以上三个约束下，来求信息熵的表达式。

记函数$$g(n) = H(1/n, 1/n, \cdots, 1/n)$$，由上面第二条可知，$$g(n)$$是$$n$$的单调递增函数。

假设随机试验有等概率$$s$$个结果，不失一般性，令$$s=mn$$，$$m$$和$$n$$是正整数，我们将这$$s$$个结果分为$$m$$个有等概率$$n$$个结果的实验，那么由上述第三条知道：

$$g(s) = g(mn) = g(m) + m(\frac{1}{m}g(n)) = g(m) + g(n)$$

由Lemma可知，$$g(n) = H(1/n, 1/n, \cdots, 1/n) = K logn$$。

我们得到了等概率结果的信息熵$$H$$的表达式，但这不是最一般的表达式。

为此，我们再将$$s$$个等概率结果进行如下处理：把$$s$$分成随机的$$n>1$$堆，每堆包含$$n_i$$个结果：

$$s = \Sigma_{i=1}^n n_i$$
$$p_i = \frac{n_i}{s}, i=1,2,\cdots,n$$

可以把$$p_i$$堪称概率，这个随机试验可以划分为两个连续的随机试验：

* 1)首先的实验是以概率$$p_i$$出现事件$$A_i(i=1,2,\cdots,n)$$
* 2)在$$A_i$$已经发生的基础上，探究它是$$n_i$个等概率事件里的具体哪一个发生了。

根据上面性质第三条，我们得到：

$$K log s = H(p_1, p_2, \cdots, p_n) + K \Sigma_{i=1}^n p_i log n_i$$

因此，

$$H(p_1, p_2, \cdots, p_n) = K log s - K\Sigma_{i=1}^n p_i log n_i = -K \left[ \Sigma_{i=1}^n p_i(log n_i - log s) \right] $$

$$= -K \left[ \Sigma_{i=1}^n p_i (log (\frac{n_i}{s}) \right] = -K \Sigma_{i=1}^n p_i log p_i$$

这样就得到了信息熵的一般表达式。

常数$$K>0$$如何选取不关键，而且log的底也可以任意选取。


### 参考文献

* https://www.zhihu.com/question/30828247/answer/2146861892



## 3. 矩阵的eigenvalue，eigenvector，algebraic multiplicity和geometric multiplicity和相似对角化

$$Ax = \lambda x$$，$$A \in R^{n \times n}$$为$$n$$阶方阵，$$x$$叫做$$A$$的特征向量，eigenvector，$$\lambda$$叫做特征值，eigenvalue。

寻找矩阵的特征值，是通过求解$$A$$的行列式$$det(\lambda I - A)=0=(\lambda - \lambda_1)^{n_1} (\lambda - \lambda_2)^{n_2} \cdots (\lambda - \lambda_r)^{n_r}$$，$$n_1 + n_2 + \cdots + n_r = n$$，一共$$r$$个不同的特征值。多项式$$det(\lambda I - A)$$的根有$$n$$个，不同的根有$$r$$个，根$$\lambda_1$$有$$n_1$$重，$$\lambda_2$$有$$n_2$$重，以此类推。

找到特征值之后，就可以找到特征值对应的特征向量了。特征值$$\lambda_i$$对应的特征向量的求解方式为，寻找方程$$Ax = \lambda_i x$$的解，也就是$$(A-\lambda_i I) x = 0$$的解。也就是说特征值对应的特征向量在方阵$$A-\lambda_i I$$的null space，$$N(A-\lambda_i I)$$里。

每个特征值对应的特征向量都有无数个，问题是特征向量构成的特征空间，eigenspace有几维，也就是$$N(A - \lambda_i I)$$有几维。假设$$N(A - \lambda_i I)$$有$$m_i$$维，那么其是否等于特征值$$\lambda_i$$的重根的个数呢？并且$$A$$的$$r$$个eigenspace的维度加起来是否等于$$n$$呢？

我们容易验证对角矩阵的所有eigenspace的维数加起来是$$n$$，并且对应不同特征值的特征向量是线性无关的。假设$$A$$的特征值$$\lambda_1$$的eigenspace，$$N(A - \lambda_1 I)$$维度为$$m_1$$，$$\lambda_2$$的eigenspace的维度为$$m_2$$，以此类推。从而我们有$$m_1 + m_2 + \cdots + m_r = n$$。每个eigenspace，$$N(A - \lambda_k I)$$里，分别取$$m_k$$个线性无关的向量，$$1 \leq k \leq r$$，然后将所有的这样的向量综合起来，记为$$x_1, \cdots, x_n$$，则其为$$R^n$$的一组基。从而：

$$A(x_1, x_2, \cdots, x_n) = (\lambda_1 x_1, \lambda_2 x_2, \cdots, \lambda_n x_n) = (x_1, x_2, \cdots, x_n) \begin{pmatrix} \lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

令$$P = (x_1, x_2, \cdots, x_n) \in R^{n \times n}$$，那么上面的式子就是：

$$AP = P \begin{pmatrix} \lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

因为$$P$$里的列向量都是线性无关的，也就是$$n$$阶方阵$$P$$的秩$$rank(P) = n$$，所以上面式子两侧乘以$$P^{-1}$$，就将$$A$$进行了相似对角化。

如果$$m_1 + m_2 + \cdots + m_r \neq n$$，而是$$m_1 + m_2 + \cdots + m_r = m < n$$，那么矩阵$$P$$就是$$n \times m$$的矩阵，$$P$$不再是方阵，$$rank(P) = m < n$$，从而矩阵$$P$$不可逆，也就无法通过$$P^{-1}AP$$来对矩阵进行相似对角化。

因此，只有当所有特征值对应的特征空间合起来的维度为$$n$$时，矩阵才可以相似对角化。而且不同的$$(x_1, x_2, \cdots, x_n)$$的书写顺序，对应的对角矩阵里面$$\lambda_i$$的顺序也不一样。

我们有$$det(\lambda I - A)=0=(\lambda - \lambda_1)^{n_1} (\lambda - \lambda_2)^{n_2} \cdots (\lambda - \lambda_r)^{n_r}$$，$$n_1 + n_2 + \cdots + n_r = n$$，一共$$r$$个不同的特征值。多项式$$det(\lambda I -A)$$中，$$\lambda - \lambda_1$$的次数为$$n_1$$，记为特征值$$\lambda_1$$的代数重数，algebraic multiplicity。$$\lambda_1$$的eigenspace的维数$$m_1$$，叫做特征值$$\lambda_1$$的几何重数，geometric multiplicity。那么每个特征值的几何维数是否等于代数维数呢？答案是代数维数不小于几何维数，即$$n_1 \geq m_1$$。

从而$$m_1 \leq n_1, m_2 \leq n_2, \cdots, m_r \leq n_r$$，我们有$$m_1 + m_2 + \cdots + m_r = m \leq n = n_1 + n_2 + \cdots + n_r$$。因为$$m \leq n$$，所以所有的特征空间的线性无关的特征向量合起来的矩阵$$P \in R^{n \times m}$$，在等号不成立的时候不是方阵，从而上述矩阵相似对角化操作无法实行。

Jordan Canonical Form Theorem告诉我们由于$$r$$个特征空间的维数的和$$m \leq n$$，所以矩阵无法进行相似对角化，但可以退而求其次，将矩阵相似变换为$$m$$个Jordan block的对角形式：

$$\begin{pmatrix} J_1 & 0 & 0 & \cdots & 0 \\ 0 & J_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & J_m \end{pmatrix} \in R^{n \times n}$$

其中第$$i, i=1,2,\cdots, r$$个约当块为

$$J_i = \begin{pmatrix} \lambda_i & 1 & 0 & \cdots & 0 \\ 0 & \lambda_i & 1 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & 1 \\ 0 & 0 & 0 & \cdots & \lambda_i \end{pmatrix}$$

那么问题是，将矩阵$$A$$变换为约当块对角型的$$n$$阶矩阵$$P \in R^{n \times n}$$是什么样呢？

$$J_A = \begin{pmatrix} J_1 & 0 & 0 & \cdots & 0 \\ 0 & J_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & J_m \end{pmatrix} \in R^{n \times n} = p^{-1} A P$$。

可进行相似对角化的矩阵$$A$$，在对角化之后的每个对角元$$\lambda_i$$都对应矩阵$$P$$的某个特征值的eigenspace里的一个基向量，特征值$$\lambda_1$$就对应矩阵$$P$$的$$m_1=n_1$$个基，特征值$$\lambda_2$$就对应$$P$$矩阵的$$m_2 = n_2$$个基，以此类推。那么对于不可相似对角化，而只能相似约当块对角化的矩阵$$A$$，每个Jordan Block也对应某个特征值的eigenspance里的一个基向量。所以如果特征值$$\lambda_1$$的eigenspace是三维的，即$$m_1 = 3$$，那么$$\lambda_1$$就对应三个Jordan block。所以如果将$$A$$进行约当块对角化，那么约当块的个数为$$m$$个，也就是所有eigenspace的维度合起来的值，并不等于特征值的个数，也不等于代数重数的和。

假设特征值$$\lambda_1$$的eigenspace，$$N(A - \lambda_1 I)$$有$$m_1 = 2$$维，就对应有2个约当块，那么它们的size应该是多大呢？

我们有，因为$$A$$和$$J_A$$相似，所以$$det(A - \lambda I) = det(J_A - \lambda I) = (\lambda - \lambda_1)^{n_1} (\lambda - \lambda_2)^{n_2} \cdots (\lambda - \lambda_r)^{n_r}$$，所以特征值$$\lambda_1$$的$$m_1$$个约当块的size之和一定等于$$n_1$$，从而我们知道了代数重数和几何重数的关系。显然，如果所有的Jordan block的size都是1，那么代数重数等于几何重数。

我们有$$dim \left[ N(A - \lambda_1 I) \right] = dim \left[ P^{-1} N(A - \lambda_1 I) P \right] = dim \left[ N(P^{-1} A P - \lambda_1 P^{-1} I P \right] = dim \left[ J_A - \lambda_1 I \right]$$。而且：

$$J_A - \lambda_1 I = \begin{pmatrix} 0 & \ast & 0 & \cdots & \cdots & \cdots & \cdots & \cdots & 0 \\ 0 & 0 & \ast & 0 & \cdots & \cdots & \cdots & \cdots & 0 \\ \vdots & 0 & \vdots & \ast & 0 & \cdots & \cdots & \cdots & 0 \\ 0 & \cdots & 0 & 0 & 0 & \cdots & \cdots & \cdots & 0 \\ 0 \cdots & \cdots & 0 & \lambda_1 - \lambda_2 \neq 0 & \ast & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\ 0 & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots &  \lambda_1 - \lambda_r \neq 0 \end{pmatrix}$$

其中$$\ast$$可能是1也可能是0。假设特征值的代数重数为11，那么$$J_A - \lambda_1 I$$的对角线上有11个0。假设特征值$$\lambda_1$$的几何重数为5，所以对应有5个约当块，这五个约当块加起来的size为11。

对于任意某个特征值$$\lambda_i$$的eigenspace的某个基$$x_1$$，我们有$$(A-\lambda_i I)x_1 = 0$$。如果找到$$v_1$$使得$$(A - \lambda_i I)v_1 = x_1$$有解，也就是$$(A - \lambda_i I) (A - \lambda_i I) v_1 = 0$$。也就是$$(A - \lambda_i I) ^2 v_1 = 0$$的解至少有1维，也就是$$(A - \lambda_i I) ^2$$的零空间维数至少为1，说明对应于$$x_1$$的Jordan Block的size至少为2。如果进一步找到$$(A - \lambda_i I) w_1 = v_1$$，那么$$(A - \lambda_i I) ^3 w_1 = 0$$，从而对应$$x_1$$的Jordan block的size至少为3。一直这么找下去，直到没有解为止。如果最大size的Jordan block为$$l$$，我们设它对应的最底层的变量为$$u$$，也就是找不到$$y$$使得$$(A - \lambda_i I) y = u$$。往回推，因为$$u$$是由$$(A - \lambda_i I) u = u_1$$，(A - \lambda_i I) u_1 = u_2, \cdots, (A - \lambda_i I) u_k = u_{k+1}$$这样一集一集推来的，从而$$(A - \lambda_i I) u = u_1, (A - \lambda_i I) (A - \lambda_i I) u = u_2, \cdots, (A - \lambda_i I) ^{l-1}u = x$$，其中$$x$$是$$\lambda_i$$的eigenspace里的某个基，$$(A - \lambda_i I) x = 0$$。$$(A - \lambda_i I) ^l u =0$$。

假设矩阵$$A$$有两个特征值$$\lambda_1$$和$$\lambda_2$$。对于特征值$$\lambda_1$$，假设$$(A - \lambda_1 I) x = 0$$的解有2维，也就是$$\lambda_1$$的eigenspace是2维，所以$$\lambda_1$$对应的Jordan block有两个。任意选取$$\lambda_1$$的eigenspace的一组基$$x_1, x_2$$。假设对于$$x_1$$的Jordan block的size是3，也就是能找到$$(A - \lambda_1 I) v_1 = x_1, (A - \lambda_1 I) w_1 = v_1$$，但无法找到$$z_1$$满足$$(A - \lambda_1 I) z_1 = w_1$$，所以

$$J_1 = \begin{pmatrix} \lambda_1 & 1 & 0 \\ 0 & \lambda_1 & 1 \\ 0 & 0 & \lambda_1 \end{pmatrix}$$

我们进行以下操作：

$$(x_1, v_1, w_1) J_1 = (x_1, v_1, w_1) \begin{pmatrix} \lambda_1 & 1 & 0 \\ 0 & \lambda_1 & 1 \\ 0 & 0 & \lambda_1 \end{pmatrix} = (\lambda_1 x_1, x_1 + \lambda_1 v_1, v_1 + \lambda_1 w_1) = A (x_1, v_1, w_1) \tag{1}$$

也就是$$\lambda_1 x_1 = A x_1, x_1 + v_1 \lambda_1 = A v_1, v_1 + \lambda_1 w_1 = A w_1$$，从而$$(A - \lambda_1 I) x_1 = 0, (A - \lambda_1 I )v_1 = x_1, (A - \lambda_1 I) w_1 = v_1$$。满足条件，从而等式1成立。

假设对于$$x_2$$来说，无法找到$$v_2$$满足$$(A - \lambda_1 I) v_2 = x_2$$，即Jordan block的size是1。那么$$J_2 = \begin{pmatrix} \lambda_1 \end{pmatrix}$$。

对于特征值$$\lambda_2$$，假设$$(A - \lambda_2) x = 0$$的eigenspace只有1维，从而也就只有一个Jordan block。任意选取一个基$$y_1$$，假设对于$$y_1$$我们发现的Jordan block的size为2，也就是能找到$$r_1$$满足$$(A - \lambda_2 I) r_1 = y_1$$，却无法找到$$s_1$$满足$$(A - \lambda_2 I)s_1 = r_1$$。从而

$$J_3 = \begin{pmatrix} \lambda_2 & 1 \\ 0 & \lambda_2 \end{pmatrix}$$。

最后，我们使得$$A$$能够约当块对角化的矩阵$$P$$设置为$$(x_1, v_1, w_1, x_2, y_1, r_1)$$，可以使得：

$$AP = P \begin{pmatrix} J_1 & 0 & 0 \\ 0 & J_2 & 0 \\ 0 & 0 & J_3 \end{pmatrix}$$




## 4. 实对称矩阵特性、相似变换与SVD分解

如果$$n$$阶方阵可以对角化，那么$$P$$阵是$$n$$个线性无关的特征向量组成的：

$$AP = P \begin{pmatrix} \lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

$$P = (x_1, x_2, \cdots, x_n)$$。

但是一般的矩阵并不会是一个方阵，同时是一个$$m \times n$$的矩阵，那么这样的矩阵是否能进行某种对角化？

为了实现非方阵$$A \in R^{m \times n}$$的对角化，我们对$$A A^T \in R^{m \times m}$$或者$$A^T A \in R^{n \times n}$$进行对角化。但是为了仍然可以根据$$A A^T$$和$$A^T A$$推导出$$A$$的一个分解形式，假设这个分解形式为$$A = U \Sigma V^T$$，$$U \in R^{m \times m}, \Sigma \in R^{m \times n}, V \in R^{n \times n}$$，那么：

$$A^T A = (V \Sigma^T U^T) (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T$$

$$A A^T = (U \Sigma V^T) (V \Sigma^T U^T) = U \Sigma V^T V \Sigma^T U^T$$

我们希望中间的$$U^T U = I$$以及$$V^T V = I$$，也就是说$$U,V$$是orthonormal的，即单位正交矩阵。

首先我们来看方阵$$A^T A$$，对于任意特征值$$A^T A x_i = \lambda_i x_i$$，两边除以特征向量的模，等式依然成立$$A^T A \frac{x_i}{\lVert x_i \rVert} = \lambda_i \frac{x_i}{\lVert x_i \rVert}$$。我们令$$P_1 = (\frac{x_1}{\lVert x_1 \rVert}, \frac{x_2}{\lVert x_2 \rVert}, \cdots, \frac{x_n}{\lVert x_n \rVert})$$，同样也有：

$$A^TAP_1 = P_1 \begin{pmatrix} \lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

且有$$P_1^TP_1 = I$$。

现在问题是，矩阵$$A^TA \in R^{n \times n}$$就一定可以进行相似对角化么？答案是肯定的，因为其是实对称矩阵，就一定能够可以找到$$n$$个相互垂直的特征向量。

为什么实对称矩阵一定可以被相似对角化呢？因为任何方阵都可以被Jordan block对角化，但是其就会失去对称性，从而矛盾，所以对称矩阵一定可以被相似对角化。

而且实对称矩阵的不同eigenspace里的向量正交（对于普通方阵，其不同eigenspace里的向量只是线性无关，但不一定正交）。

所以实对称矩阵有非常好的性质，其所有的特征值都是实数，还可以被分为$$r$$个相互独立的特征空间，然后将这些特征空间里的特征向量都单位化，就可以得到这个矩阵的相似对角阵的$$P$$，且$$P^{-1}P = I$$。

因此，$$A^T A \in R^{n \times n}$$可以相似对角化，它存在有$$n$$个相互正交的单位向量组成的矩阵$$P_1$$使得：

$$A^TA = P_1 \begin{pmatrix} \lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \lambda_n \end{pmatrix}P_1^{-1} = P_1 \begin{pmatrix} \lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \lambda_n \end{pmatrix} P_1^T$$

类似的，$$A A^T \in R^{m \times m}$$可以相似对角化，它存在有$$m$$个相互正交的单位向量组成的矩阵$$P_2$$使得：

$$A A^T = P_2 \begin{pmatrix} \Lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \Lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \Lambda_m \end{pmatrix}P_2^{-1} = P_2 \begin{pmatrix} \Lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \Lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \Lambda_m \end{pmatrix} P_2^T$$

取$$V = P_1$$，$$U = P_2$$，那么$$A^T A = V \Sigma^T U^T U \Sigma V^T = P_1 \Sigma^T \Sigma P_1^T \in R^{n \times n}$$，我们有：

$$\begin{pmatrix} \lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \lambda_n \end{pmatrix} = \Sigma^T \Sigma$$

同样的$$A A^T = (U \Sigma V^T) (V \Sigma^T U^T) = U \Sigma V^T V \Sigma^T U^T = P_2 \Sigma \Sigma^T P_2^T \in R^{m \times m}$$，我们有：

$$\begin{pmatrix} \Lambda_1 & 0 & 0 & \cdots & 0 \\ 0 & \Lambda_2 & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & 0 & \cdots & \Lambda_m \end{pmatrix} = \Sigma \Sigma^T$$

然后，因为$$A^TA$$和$$AA^T$$具有相同的eigenvalues，从而上述的$$\lambda_i$$们和$$\Lambda_i$$们实际上是相同的（多出来的就是0）。从而我们就可以根据上面的式子解出$$\Sigma$$的值了。

于是$$A = U \Sigma V^T = P_2 \Sigma P_1^T \in R^{m \times n}$$。

假设$$m < n$$，那么

$$\Sigma = \begin{pmatrix} \sqrt{\Lambda_1} & 0 & \cdots & 0 & 0 & \cdots & 0 \\ 0 & \sqrt{\Lambda_2} & \cdots & 0 & 0 & \cdots & 0 \\ \vdots & \vdots & \cdots & \vdots & 0 & \cdots & 0 \\ 0 & 0 & \cdots & \sqrt{\Lambda_m} & 0 & \cdots & 0 \end{pmatrix}$$

其中右边的$$n-m$$列都是0。

如果$$m > n$$，那么

$$\Sigma = \begin{pmatrix} \sqrt{\Lambda_1} & 0 & \cdots & 0 \\ 0 & \sqrt{\Lambda_2} & \cdots & 0 \\ \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & \cdots & \sqrt{\Lambda_n} \\ 0 & 0 & \cdots & 0 \\ \vdots & \vdots & \cdots & \vdots \\ 0 & 0 & \cdots & 0 \end{pmatrix}$$

其中下面的$$m-n$$行都是0。


SVD分解实际上是通过将一个非方阵的$$A \in R^{m \times n}$$矩阵，通过$$A^TA$$和$$AA^t$$得到方阵，分别找到使得$$A^TA$$和$$AA^T$$对角化的矩阵$$P_1$$和$$P_2$$，利用$$A^TA$$和$$AA^T$$正好是对称矩阵，其拥有正交的$$P_1$$和$$P_2$$，而且$$A^TA$$和$$AA^T$$拥有相同的特征值，从而得到了使得非方阵$$A$$对角化的方法：$$A = P_2 \Sigma P_1^T$$，其中$$\Sigma$$是一个以$$A^TA$$的特征值的开方为对角元素的对角矩阵。
















---
