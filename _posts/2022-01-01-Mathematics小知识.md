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


**参考文献**
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

我们来求一下$$\Lambda$$。令$$det \lVert \lambda I - A \rVert = 0$$，得到

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


**参考文献**

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

**参考文献**

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

$$J_A - \lambda_1 I = \begin{pmatrix} 0 & \ast & 0 & \cdots & \cdots & \cdots & \cdots & \cdots & 0 \\ 0 & 0 & \ast & 0 & \cdots & \cdots & \cdots & \cdots & 0 \\ \vdots & 0 & \vdots & \ast & 0 & \cdots & \cdots & \cdots & 0 \\ 0 & \cdots & 0 & 0 & 0 & \cdots & \cdots & \cdots & 0 \\ 0 & \cdots & \cdots & 0 & \lambda_1 - \lambda_2 \neq 0 & \ast & 0 & \cdots & 0 \\ \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\ 0 & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \cdots & \lambda_1 - \lambda_r \neq 0 \end{pmatrix}$$

其中$$\ast$$可能是1也可能是0。假设特征值的代数重数为11，那么$$J_A - \lambda_1 I$$的对角线上有11个0。假设特征值$$\lambda_1$$的几何重数为5，所以对应有5个约当块，这五个约当块加起来的size为11。

对于任意某个特征值$$\lambda_i$$的eigenspace的某个基$$x_1$$，我们有$$(A-\lambda_i I)x_1 = 0$$。如果找到$$v_1$$使得$$(A - \lambda_i I)v_1 = x_1$$有解，也就是$$(A - \lambda_i I) (A - \lambda_i I) v_1 = 0$$。也就是$$(A - \lambda_i I) ^2 v_1 = 0$$的解至少有1维，也就是$$(A - \lambda_i I) ^2$$的零空间维数至少为1，说明对应于$$x_1$$的Jordan Block的size至少为2。如果进一步找到$$(A - \lambda_i I) w_1 = v_1$$，那么$$(A - \lambda_i I) ^3 w_1 = 0$$，从而对应$$x_1$$的Jordan block的size至少为3。一直这么找下去，直到没有解为止。如果最大size的Jordan block为$$l$$，我们设它对应的最底层的变量为$$u$$，也就是找不到$$y$$使得$$(A - \lambda_i I) y = u$$。往回推，因为$$u$$是由$$(A - \lambda_i I) u = u_1，(A - \lambda_i I) u_1 = u_2, \cdots, (A - \lambda_i I) u_k = u_{k+1}$$这样一级一级推来的，从而$$(A - \lambda_i I) u = u_1, (A - \lambda_i I) (A - \lambda_i I) u = u_2, \cdots, (A - \lambda_i I) ^{l-1}u = x$$，其中$$x$$是$$\lambda_i$$的eigenspace里的某个基，$$(A - \lambda_i I) x = 0$$。$$(A - \lambda_i I) ^l u =0$$。

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


## 5. 凸优化简介

在统计学和机器学习领域，绝大部分想做的事都是一种优化问题。所以面对具体的应用问题，要做的事情可以概括为如下图所示：

![1]({{ '/assets/images/OPTIM-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

即：如何将头脑中的conceptual idea转换写成寻找决策变量$$x$$的最优化问题$$min_{x \in D} f(x)$$。

所以，学习optimization，就是学习：

* 如何把具体问题formulate成为一个优化问题
* 现有算法是如何具体解算这个优化问题，$$P: min_{x \in D} f(x)$$
* 面对不同的具体问题如何选择已有算法或者甚至设计新的合适的算法来解决

绝大部分情况下，只需要做到第一点，就是将问题转换为凸优化问题，并写成标准形式，剩下的就交给已有的优化工具箱去解决。第二和第三点是做优化相关研究工作者的前沿工作。

optimization问题是十分热的问题，现有算法还有很大的优化提升空间，而且还有很多问题没有得到很好的解决（machine learning领域就有很多）

**5.1 Convexity**

历史上，优化问题通常聚焦在linear programming（线性规划）。最初人们认为，优化问题是线性还是非线性，是不同优化问题的根本区别。（参考Dr. Margaret Wright所写的notes：[Fast Times in Linear Programming: Early Success, Revolutions, and Mysteries]）。

但现在人们认为，优化问题是凸还是非凸彩色不同的优化问题之间的根本区别。因为有些问题，虽然是非线性，但是很好解决（因为是凸的），而有些问题虽然说线性，但是很难解决，因为是非凸的。

![2]({{ '/assets/images/OPTIM-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**5.2 Convex Set**

集合$$C$$是$$R^n, n=1,2,3,\cdots$$的子集，如果集合$$C$$满足$$\forall x,y \in C$$，有$$tx + (1-t)y \in C$$对于$$0 \leq t \leq 1$$都成立，那么集合$$C$$就是凸集。

下图第一行是凸集，第二行是非凸集：

![3]({{ '/assets/images/OPTIM-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

常见的convex sets有hyperplane（比原空间小一个维度的子空间，比如说三维空间里的任意平面）, halfspace（hyperplane将原空间分为的两部分就是halfspaces）, polytope，polyhedra，ellipsoid，non-negative orthant，norm ball，norm cone，positive (semi)definite matrix构成的集合等等。

凸集之间的某些运算能保证结果仍然是凸的。判断一个集合是不是凸集，一般就是两种方法：

* 验证定义中的条件是否满足
* 验证其是不是某些已知凸集的运算，该运算是不是能够保证凸性

**5.3 Convex Function**

如果函数$$f: R^n \longrightarrow R$$满足：

* 1. $$dom(f) \subset R^n$$是凸集，其中$$dom(f)$$是$$f$$的定义域
* 2. $$f(tx + (1-t)y) \leq tf(x) + (1-t)f(y)$$，$$\forall x,y \in dom(f), 0 \leq t \leq 1$$

下图是一个凸函数的例子：

![4]({{ '/assets/images/OPTIM-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

除了上述的定义，还有等价的通过epigraph的定义方法：如果$$f(x)$$的epigraph $$\lbrace (x,t): f(x) \leq t \rbrace$$是凸集，那么$$f(x)$$是凸函数。或者，如果$$-f(x)$$是concave的，那么$$f(x)$$是convex的。如果$$f(x)$$是一阶或者二阶可微，还可以通过一阶或者二阶条件来定义。

常见的convex function有$$ax+b$$，$$e^{ax}$$，$$x^a(x>0, a \geq 1$$或者$$a \leq 0$$，$$\lvert x \rvert ^p (p \geq 1)$$，$$xlogx (x>0)$$，$$L_p norm (p \geq 1)$$，$$\Sigma e^{x_i}$$，$$f(X) = Trace(AX) + b$$（$$A$$是一个矩阵），$$\sigma_{max}(X)$$（$$X$$是个矩阵， $$\sigma_{max}$$是最大的特征值）等等。凸函数的某些运算或者变量替换仍然会是凸的。判断一个函数是不是凸的，一般也是两种方法：

* 按定义判断条件是否满足
* 其是否是某些已知凸函数之间的运算，以及该运算是否能够保证凸性

**5.4 Optimization Problem**

$$min_{x \in D} f(x)$$

subject to $$g_i (x) \leq 0, i=1,\cdots, m$$

$$h_j(x) = 0, j=1,\cdots, r$$

其中，$$D = dom(f) \cap \Cap_{i=1}^m dom(g_i) \cap \Cap_{j=1}^r dom(h_j)$$，也就是所有的条件$$g_i$$和$$h_j$$以及$$f$$的定义域的交集。$$x$$被称为决策变量（decision variable），$$f(x)$$称作目标函数（objective function），是一个标量；$$g_i$$（$$m$$个）称作不等式约束（inequality constraints），是个标量；$$h_j$$（$$r$$个）称作等式约束（equality constraints），是个标量。一般都统一将优化函数写为minimize（因为maximize就相当于minimize原函数乘以负1）。

我们面对具体的问题，要做的就是怎么把问题写成如同上述的形式，并且设计算法寻找$$x$$，使得$$x$$在满足那些约束的前提下让objective function $$f$$最小。如果某个点$$x \in D$$，且将$$x$$代入到函数$$g_i$$（$$m$$个）和$$h_j$$（$$r$$个）中分别都满足$$g_i(x) \leq 0$$和$$h_j(x) = 0$$，那么$$x$$就称作该优化问题的可行解（feasible point）。feasible point可能根本不存在，或者存在一个，或者存在多个甚至无穷个。如果存在多个或者无穷个feasible points，我们需要找到这些feasible points里，能让$$f(x)$$最小的那个点$$x^{\ast}$$，那$$x^{\ast}$$就是该优化问题的最优解，optimal。

函数$$f$$，$$g_i$$和$$h_j$$可以是$$x$$的任意形式的函数（线性或者非线性），某些形式会让问题好解决，有些形式会让问题很难解决，甚至至今也没什么好办法。

不等式约束$$g_i \leq 0$$和等式约束$$h_j=0$$，从根本上说是约束了可行的求解域，如下图所示：

![5]({{ '/assets/images/OPTIM-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

以上定义了什么是优化问题，那么什么是凸优化问题呢？

**5.5 Convex Optimization Problem**

如果上述优化问题中的函数$$f$$，$$g_i$$（$$m$$个）是凸的，而且$$h_j$$（$$r$$个）是affine的（因为$$h_j$$是等式，即$$h_j(x) = a_j^T x + b_j, j=1,2,\cdots,r$$，那么该问题就是一个凸优化问题。

>为什么等式约束$$h_j$$需要是affine的？因为affine的等式约束保证了可行域仍然是凸集（直线是凸集，而曲线则是非凸集），非affine的约束就是曲线了，可行域就非凸了。

所以某个问题是凸优化问题，需要满足以下四个条件：

* 1. $$dom(f)$$是凸集
* 2. objective function $$f(x)$$是凸函数
* 3. $$g_i$$（$$m$$个）inequality constraints是凸函数
* 4. $$h_j$$（$$r$$个）equality constraints是affine函数

如之前所说，不等式约束和等式约束实际上就是约束了可行的求解域。那么凸优化问题中，$$g_i \leq 0$$导致的可行域是凸的，$$h_j=0$$是affine的，最终的可行域$$D$$是这些函数以及$$f$$的可行域的交集，而凸集的交集一定是凸的，所以最终的可行域也是凸的。

对于优化问题，convex到non-convex是由易到难，constrained到non-constrained也是由易到难。

![6]({{ '/assets/images/OPTIM-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**5.6 Convex vs. Non-convex**

为什么研究者大多会把问题转换为凸优化问题？因为对于凸优化问题，local minima也就是global minima。

![7]({{ '/assets/images/OPTIM-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

上图里的左边，因为问题是凸的，所以不管起始点是红色，蓝色，绿色，黄色还是紫色，只要沿着下降的方向（下降的方向也可以有很多种，但是只要是下降的就行），永远都会收敛到同一个最颠，而且就是全局最低点。

上图里的右边，因为问题是非凸的，如果起始点在红线和黄线的圆圈处，沿着下降的方向，会收敛到右图中右边的local minima。而如果起始点在绿线、蓝线和紫线的圆圈处，沿着下降的方向，会收敛到左边的local minima。最终收敛到哪里，依赖于系统的起始点在哪里。而且这两个local minima的值是不一样的。这个性质是很不好的。非凸优化的难点，就在于如何在这种性质的情况下，仍然能找到全局最优点，或者尽量小的局部最优点。


**5.7 为什么要把问题写成standard form**

因为这样可以让人们方便利用计算机最快速的求解convex optimization问题，通常需要将问题重新写成standard form。求解优化问题如果要利用计算机来进行的，常用的convex optimization tools，包括cvx，yalmip(matlab)，cvxpy，picos(python)等求解优化问题是分为两步的：

* 1. 检验问题是不是凸的
* 2. 把问题转化成这些tools内部的solver能够方便求解的标准形式

Margaret Wright的notes，[Fast Times in Linear Programming: Early Success, Revolutions, and Mysteries]()里，写出了从最初的simplex algorithm（基于最优点一定在corner point的intuition而设计的算法）到内点法的各种历史八卦和新算法设计的思考出发点。

**5.8 几类标准问题**

凸优化的标准问题有四类：

* 1. linear programming（LP）
* 2. quadratic programming（QP）
* 3. semi-definite programming（SDP）
* 4. cone programming（CP）

这四类问题有包含关系，如下所示：

![8]({{ '/assets/images/OPTIM-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

**Linear Programming**

如果凸优化问题里的$$f(x)=c^T x$$，$$g_i(x) = D_i^T x - d_i$$，$$h_j = A_j^T x - b_i$$，也就是：

$$min_{x \in D} c^T x$$

subject to $$Dx \leq d$$

$$Ax = b$$

其中矩阵$$D$$每行都是$$D_i^T$$，而矩阵$$A$$每行都是$$A_j^T$$，$$d$$和$$b$$是向量。

那么上述这种特殊情况下的凸优化问题，叫做linear programming问题。下图给出了在二维情况下polytope的可行域内的情况，图中的objective function由虚线表示，同一条虚线上的$$f(x)$$的值是相等的，$$f(x)$$的最优先在最下面的点$$x^{\ast}$$。

![0]({{ '/assets/images/OPTIM-0.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


**Quadratic Programming**

quadratic programming的形式如下：

$$min_{x \in D} c^T x + \frac{1}{2} x^T Q x$$

subject to $$Dx \leq d$$

$$Ax = b$$

如果矩阵$$Q$$是positive definite，那么上述就是一个凸优化问题，如果$$Q$$是negative definite的，那就不是凸优化问题了。下图是$$Q$$是positive definite和negative definite的两种情况：

![9]({{ '/assets/images/OPTIM-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

我们指的QP问题，就默认是$$Q$$是semi-positive definite的。从以上定义看出，如果$$Q=0$$，那么问题就变成了LP问题，所以LP问题是QP问题的特例。

下图给出了quadratic programming的例子，虚线是目标函数（同一条虚线上的$$f(x)$$的值是相等的，可以看出与linear programming的不同，一个是直线，而这个是曲线），最优值在$$x^{\ast}$$。图中目标函数是椭圆因为其是quadratic的。

![10]({{ '/assets/images/OPTIM-10.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

举一个典型的例子：SVM，support vector machine算法

在监督学习里，SVM是一种分类算法，它是一个QP问题。已知下图中的红点类和绿点类是两个不同的类，怎么找到一个hyperplane将两类数据分开，并使得该hyperplane到两类数据的margin最大，也就是找到图中的黑线：$$w^T x + b$$，使得绿色类的所有点（假设有$$p$$个）满足$$w^T x_{green} + b \geq 1$$，红色类的所有的点（假设有$$q$$个）满足$$w^T x_{red} + b \leq -1$$。

![11]({{ '/assets/images/OPTIM-11.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

图中灰色的部分就是margin，有两个support vectors（离黑线最近的两个点）。如果要最大化两个类的margin，就相当于：

$$max \frac{2}{\lVert w \rVert}$$

而最大化$$\frac{2}{\lVert w \rVert}$$等价于最小化$$\frac{1}{2} \lVert w \rVert ^2$$，于是问题formulate为：

$$min_{w \in D} \frac{1}{2} w^T w$$

subject to $$w^T x_{green} + b \geq 1$$

$$-w^T x_{red} - b \geq 1$$

其中决策变量是$$w$$，而$$x_{green}$$和$$x_{red}$$都是已知的点，代入上述条件即可，因为绿色有$$p$$个点，从而得到$$p$$个不等式，而同理红色的点对应了$$q$$个不等式。

但在很多实际的情况下，两个类之间并没有上图那样的明确的界限，而是像下图这样没有明确界限，但仍然可分：

![12]({{ '/assets/images/OPTIM-12.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

这时需要在将两类分开的前提下，同时尽量减少错误的分类。我们可以通过引入一个额外的参数来实现，将两个条件$$w^T x_{green} + b \geq 1$$，和$$w^T x_{red} + b \leq -1$$放宽为：$$w^T x_{green} + b \geq 1 - \epsilon_i$$，$$i=1,\cdots,p$$，$$-w^T x_{red} - b \geq -1 - \xi_j$$，$$j=1,\cdots,q$$。所以优化问题变为：

$$min_{w, \epsilon, \xi} \frac{1}{2} w^T w + C(\Sigma_{i=1}^p \epsilon_i + \Sigma_{j=1}^q \xi_{j})$$

subject to $$w^T x_{green} + b \geq 1 - \epsilon_i$$

$$-w^T x_{red} - b \geq 1 - \xi_j$$

$$\epsilon \geq 0$$

$$\xi \geq 0$$

从而决策变量不仅仅是$$w$$，而是$$(w, \epsilon, \xi)$$。objective function是决策变量的quadratic的形式，而且不等式约束也是线性的，所以是QP问题。

下图给出了选取不同参数$$C$$的分类效果图，$$C$$越大，灰色阴影部分的margin就越小，错误分类的概率就越小。$$C$$不是决策变量，而是手动调节的参数，这类参数称为超参数，hyper-parameter。

![13]({{ '/assets/images/OPTIM-13.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


**Semi-definite Programming**

semi-definite programming (SDP）的形式为：

$$min_{x \in D} c^T x$$

subject to $$x1 F_1 + x_2 F_2 + \cdots + x_n F_n \leq F_0$$

$$Ax = b$$

其中$$x = (x_1, x_2, \cdots, x_n)^T \in R^n$$，$$F_i \in R^{d \times d}$$是对称矩阵，$$A \in R^{m \times n}$$。


## 6. 浅谈变分原理

对付数学物理中的极值问题，变分法（variational principle）是非常重要的工具。此post介绍了变分法的基本思想。

1696年6月，瑞士科学家Johann Bernoulli在Acta Eruditorum上向全世界的科学家发起一项挑战。他提出了一个问题：让一个物体从静止开始沿着一个光滑无摩擦的轨道下滑到一个固定点，如果要求下滑过程耗时最短，轨道应该是什形状？

这个问题被称作最速降曲线问题（the brachistochrone problem）。Johann本文利用光学原理类比给出了一种解法，他哥哥Jacob Berboulli想到了另一种解法，而Leibniz、L’Hospital等都给出了各自的解法。

而Newton提出了一种新颖的方法解决此问题，即为变分法。

假设我们有两个定点$$(a,p)$$和$$(b,q)$$，连接这两点的任意曲线的方程$$y=y(x)$$都将满足如下边界条件：

$$y(a) = p, y(b) = q \tag{1}$$

现在考虑如下形式的定积分：

$$I = \int_{a}^{b} f(y, y^{'}) dx \tag{2}$$

其中$$f(y,y^{'})$$是关于$$y(x)$$和其一阶导数$$y^{'}(x)$$的函数，我们期望找到一个具体的$$y(x)$$，使得$$I$$到达极值（极大或极小）。

注意在一般的极值问题中，我们考察的是自变量$$x$$的变化：$$x$$取值多少时，函数会有极值。而现在这个新问题的不同之处是，我们考察的是函数$$y(x)$$的变化：$$y(x)$$是什么形式时，$$I$$会有极值（高级的叫法，$$I$$称为函数$$y(x)$$的泛函）。然而这两类问题仍有共同之处：当$$I$$取极值时，对$$y(x)$$作微小的变化，$$I$$在一级近似下应该保持不变。

![v1]({{ '/assets/images/V-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

如果$$y(x)$$有微小改变$$\delta y(x)$$（$$\delta y(x)$$称作函数$$y(x)$$的变分），那么$$f(y,y^{'})$$的变化为：

$$\delta f = \frac{\partial f}{\partial y} \delta y + \frac{\partial f}{\partial y^{'}} \delta y^{'} \tag{3}$$

$$I$$相应的变化为：

$$\delta I = \int_{a}^{b} \left[ \frac{\partial f}{\partial y} \delta y + \frac{\partial f}{\partial y^{'}} \delta y^{'} \right] dx \tag{4}$$

公式4方括号里的第二项可以改写为$$\frac{\partial f}{\partial y^{'}} \frac{d(\delta y)}{dx}$$，然后我们可以分部积分：

$$\int_{a}^{b} \frac{\partial f}{\partial y^{'}} \delta y^{'} dx = \int_{a}^{b} \frac{\partial f}{\partial y^{'}} d(\delta y)$$

$$=\frac{\partial f}{\partial y^{'}} \delta y \vert_a^b - \int_{a}^{b} \delta y \frac{d}{dx}(\frac{\partial f}{\partial y^{'}})dx$$

由于$$y(x)$$的边界条件固定，$$\delta y(a) = \delta y(b) = 0$$，所以上述分部积分第一项为零。从而代回公式4中，可得：

$$\delta I = \int_{a}^{b} \left[ \frac{\partial f}{\partial y} - \frac{d}{dx} (\frac{\partial f}{\partial y^{'}}) \right]\delta y(x) dx \tag{5}$$

如果$$I$$有极值，对任意满足边界条件的$$\delta y(x)$$，都必须有$$\delta I = 0$$，从而：

$$\frac{\partial f}{\partial y} - \frac{d}{dx}(\frac{\partial f}{\partial y^{'}}) = 0 \tag{6}$$

这就是Euler-Lagrange方程，它是变分法的核心定理。有了这个定理，就可以找出所求的极值函数$$y(x)$$。

通常来说，Euler-Lagrange方程会是一个二阶的微分方程，$$y(x)$$的通解中含有的两个待定常数刚好可以通过两个边界条件确定。

**例1：两点间的最短路径**

给定平面上两点$$(a,p)$$和$$(b,q)$$，连接它们的长度最短的曲线是什么？

曲线的长度计算为：

$$S = \int_{a}^{b} \sqrt{1 + y^{'2}} dx $$

现在希望$$S$$有最小值，取$$f(y,y^{'}) = \sqrt{1 + y^{'2}}$$，运用Euler-Lagrange方程来找到$$y(x)$$。

注意到$$\frac{\partial f}{\partial y} = 0$$，$$\frac{\partial f}{\partial y^{'}} = \frac{y^{'}}{\sqrt{1 + y^{'2}}}$$

代入Euler-Lagrange方程里，可得：

$$\frac{d}{dx}\frac{y^{'}}{\sqrt{1 + y^{'2}}}=0$$

从而$$\frac{y^{'}}{\sqrt{1 + y^{'2}}}$$是个常数，也就是$$y^{'}$$也是个常数，从而$$y(x)=kx+c$$是个直线，又要通过$$(a,p)$$和$$(b,q)$$两点。


**例2：最速降曲线**

我们将坐标系的y轴向下，斜向下的轨道就可以由$$y(x)$$给出，其中轨道的起点和终点分别设为$$(0,0)$$和$$(a,b)$$，从而来求最速降曲线的函数式。

![v2]({{ '/assets/images/V-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

当物体下滑到$$(x,y)$$位置时，结合物理定律：

$$\frac{1}{2}mv^2 = mgy$$

$$v = \sqrt{2gy}$$

根据定义：

$$v = \frac{ds}{dt} = \sqrt{1 + y^{'2}} \frac{dx}{dt}$$

将上面两个式子联立，得：

$$dt = \sqrt{\frac{1 + y^{'2}}{2gy}}dx$$

积分之后就可以得到总时间：

$$T = \frac{1}{\sqrt{2g}} \int_{0}^{a} \sqrt{\frac{1 + y^{'2}}{y}}dx$$

为了找出让$$T$$取得极小的$$y(x)$$，我们可以取$$f(y,y^{'}) = \sqrt{\frac{1 + y^{'2}}{y}}dx$$，再利用Euler-Lagrange公式来计算：

$$\frac{1}{2} \sqrt{\frac{1+y^{'2}}{y^3}} + \frac{d}{dx}(\frac{y^{'}}{\sqrt{y(1 + y^{'2})}}) = 0$$

化简得：

$$\frac{1}{2} (1+y^{'2}) + yy^{''}=0$$

两侧乘以$$2y^{'}$$，可以得到：

$$\frac{d}{dx} \left[y(1+y^{'2})\right] = 0$$

$$y(1+y^{'2}) = k$$

$$y^{'} = \sqrt{\frac{k-y}{y}}$$

其中$$k$$是一个常数。

从而：

$$x = \int dx = \int \sqrt{\frac{y}{k-y}} dy$$

做三角换元，设$$y = ksin^2 \theta$$，则：

$$x = k \int (1-cos2\theta) d\theta$$

从而：

$$x = k \theta - \frac{1}{2} k sin2 \theta + c$$

其中$$c$$是积分常数。再将$$\theta$$替换为$$y$$，就可以得到：

$$x(y) = k arcsin \sqrt{\frac{y}{k}} - \sqrt{y(k-y)} + c$$

轨道起点是$$(0,0)$$，可以得到$$c=0$$，于是：

$$x(y) = k arcsin \sqrt{\frac{y}{k}} - \sqrt{y(k-y)}$$

轨道终点是$$(a,b)$$，所以上述上述$$k$$满足：

$$karcsin \sqrt{\frac{b}{k}} - \sqrt{b(k-b)} = a$$

这是一条摆线。


**参考文献**

* https://zhuanlan.zhihu.com/p/139018146


## 7. Singular Value Decomposition（SVD分解）

我们将会学习一个最普适的方法来“对角化”（diagonalize）一个矩阵。这个方法就叫做singular value decomposition。

对于方矩阵来说，其有着很好的性质。特别是对于对称实方矩阵，其可以直接进行eigendecomposition。对于一般的方矩阵，也有类似的分解存在，可以参考machine learning知识点和技巧那里的SVD内容。

但是对于一般的矩阵，其大小为$$m \times n$$，甚至没有办法定义eigenvalue和eigenvector。但是我们发现，对于任意的矩阵来说，$$A^T A$$和$$A A^T$$是方阵，而且还是对称矩阵，而且还是positive semi-definite的，其具有很好的性质：可以进行eigendecomposition。

从而我们可以利用$$A^T A$$和$$A A^T$$来构造$$A$$的SVD分解。

假设矩阵$$A$$的大小为$$m \times n$$，秩为$$r$$。那么矩阵$$A A^T$$的大小为$$m \times m$$，$$A^T A$$的大小为$$n \times n$$，且这两个矩阵的秩也都是$$r$$。

>这是因为矩阵$$A$$的秩可以理解为矩阵$$A$$的列向量构成的向量子空间的维度，而矩阵$$A A^T$$的每一列都是矩阵$$A$$的列的线性组合（$$A A^T$$的每一列的组成的线性系数是$$A$$的每一行），所以说矩阵$$A A^T$$的列向量组成的空间的维度不大于$$r$$。但这不足以证明$$A A^T$$和$$A$$的秩一样。完整的证明方法是，证明$$A$$的null space和$$A A^T$$的null space的相同，参考这里：https://math.stackexchange.com/questions/349738/prove-operatornamerankata-operatornameranka-for-any-a-in-m-m-times-n

矩阵$$A A^T$$和$$A^T A$$的秩都是$$r$$，而且都是positive semidefinite的，从而它们有$$r$$个eigenvalues（可能有重复），而且都是正的。并且也有$$r$$个线性无关的特征向量。因为这两个矩阵还都是对称的，所以它们的eigenvectors是orthogonal的，我们还可以将其归一化为orthonormal的。

>$$A A^T$$和$$A^T A$$的特征值是相同的，这是因为:https://math.stackexchange.com/questions/1087064/non-zero-eigenvalues-of-aat-and-ata

对于这些eigenvalues，将$$A^T A$$相对于这些eigenvalues的eigenvectors记为$$v_1, v_2, \cdots, v_r$$，这些将会构成矩阵$$A$$的SVD分解的行空间。而$$A A^T$$相对于这些eigenvalues的eigenvectors记为$$u_1, u_2, \cdots, u_r$$，这些将会构成矩阵$$A$$的SVD分解的列空间。

而这些向量有着非常好的性质：

$$A v_1 = \sigma_1 u_1, A v_2 = \sigma_2 u_2, \cdots, A v_r = \sigma_r u_r$$

其中$$\sigma_1, \sigma_2, \cdots, \sigma_r$$是一些正数，被称为矩阵$$A$$的singular values。

而上述的这些关系，可以让我们有如下的公式：

$$A
\begin{pmatrix}
v_1 & \cdots & v_r
\end{pmatrix}

=

\begin{pmatrix}
u_1 & \cdots & u_r
\end{pmatrix}

\begin{pmatrix}
\sigma_1 &  &  \\
  & \ddots &  \\
  &  &  \sigma_r
\end{pmatrix}
$$

将上述公式记为$$AV = U \Sigma$$。

根据之前的构造，矩阵$$V$$是orthonormal的，从而我们可以在两边乘以$$V^T$$，得到：$$A = U \Sigma V^T$$。这就是矩阵$$A$$的singular value decomposition。

而如果我们希望将上述分解里的$$V$$和$$U$$都变成方阵也很简单。已知矩阵$$A$$的秩是$$r$$，从而$$A$$的null space的维度是$$n-r$$（因为$$Ax=0$$的解$$x$$表明$$x$$和矩阵$$A$$里的每一行都垂直，从而矩阵的null space就是和矩阵行向量构成的空间相垂直的空间，也就是维度为$$n-r$$），而$$A$$的left null space的维度是$$m-r$$（因为$$A^T x = 0$$的解$$x$$表明$$x$$和矩阵$$A$$里的每一列都垂直，从而矩阵的left null space就是和矩阵列向量构成的空间相互垂直的空间，也就是维度为$$m-r$$）。

有了上述基础，我们可以将矩阵$$V$$再扩充$$n-r$$列，新增的列都和之前的列垂直，而且是归一化的，这些列都在$$A$$的null space里（根据上述基础，这是能做得到的），从而新的矩阵$$V$$大小为$$n \times n$$。实际上直接将之前的$$V$$的列继续扩充到$$n$$维，构成线性无关的向量，那么新增的向量也就是$$A$$的null space里的向量。接下来，也将矩阵$$U$$的列向量继续扩充，增加线性无关的向量，直到$$U$$变成$$m \times m$$，新增加的向量也就构成了矩阵$$A$$的left null space的一个基。而且我们还需要将$$\Sigma$$扩充为$$m \times n$$维，扩充方法就是增加列和行，里面都是0。

矩阵$$A$$的SVD分解还可以写成：

$$A = U \Sigma V^T = u_1 \sigma_1 v_1^T + u_2 \sigma_2 v_2^T + \cdots + u_r \sigma_r v_r^T$$

每个$$u_i \sigma_i v_i^T$$的秩都是1。如果我们将那些singlar value很小的项去掉，就构成了原矩阵的一个近似，这有很广的应用。


下面给出上述SVD分解的一个证明。

因为矩阵$$A A^T$$和矩阵$$A^T A$$都是positive semidefinite的，其非零的特征值都是正的。

假设$$\lambda_i$$是矩阵$$A^T A$$的一个非零的特征值，对应的特征向量为$$v_i$$，那么$$A^T A v_i = \lambda_i v_i$$，我们记$$\sigma_i = \sqrt{\lambda_i}$$，从而$$A^T A v_i = \sigma_i^2 v_i$$。

从而，$$v_i^T A^T A v_i = \sigma_i^2 v_i^T v_i$$，也就是$$\sigma_i^2 = \sigma_i^2 v_i^T v_i = v_i^T A^T A v_i = (A v_i)^T (A v_i) = \lVert A v_i \rVert^2$$，从而我们有$$\lVert A v_i \rVert = \sigma_i$$。

而且，因为$$A^T A v_i = \sigma_i^2 v_i$$，所以$$A A^T A v_i = \sigma_i^2 A v_i$$，也就是说$$A v_i$$是矩阵$$A A^T$$的相对于特征值$$\sigma_i^2$$的特征向量，将这个特征向量归一化之后就有$$u_i = A v_i / \lVert A v_i \rVert = A v_i / \sigma_i$$，也就是说，$$A v_i = \sigma_i u_i$$。证明了之前$$v_i$$和$$u_i$$之间的关系。



## 8. 随机向量（random vector）以及variance-covariance matrix

**Definition 1** 一个random vector $$\vec{X}$$是由联合分布表示的随机变量们构成的向量，$$\vec{X} = (X_1, X_2, \cdots, X_p)$$。这些随机变量$$X_1, \cdots, X_p$$由某个联合分布描述，其可能是独立的，可能是某部分和某部分是独立的，也可能有着很复杂的依赖关系。

**Definition 2** 一个随机向量$$\vec{X} = \left[ X_1, X_2, \cdots, X_p \right]^T$$的期望$$E \vec{X}$$的计算方法为：$$E \vec{X} = \left[ E X_1, E X_2, \cdots, E X_p \right]$$。

对于任何$$k \times p$$矩阵$$A$$和$$1 \times j$$矩阵$$B$$，我们有：$$E (A \vec{X}) = A E \vec{X}$$以及$$E(\vec{X} B) = (E \vec{X}) B$$。

**Definition 3** 一个随机向量$$\vec{X}$$的variance-covariance matrix（或者简称covariance matrix）的定义为：$$Cov(\vec{X}) = E \left[ (\vec{X} - E \vec{X})(\vec{X} - E \vec{X})^T \right]$$。

**Proposition 1** $$Cov(\vec{X}) = E \left[ \vec{X} \vec{X}^T \right] - E \vec{X} (E \vec{X})^T$$

**Proposition 2** 

$$Cov(\vec{X}) = 
\begin{pmatrix}
Var(X_1) & Cov(X_1, X_2) & \cdots & Cov(X_1, X_p) \\
Cov(X_2, X_1) & Var(X_2) & \cdots & Cov(X_2, X_p) \\
\vdots & \vdots & \ddots & \vdots \\
Cov(X_p, X_1) & Cov(X_p, X_2) & \cdots & Var(X_p)
\end{pmatrix}
$$

因此，$$Cov(\vec{X})$$是一个对称矩阵。


**8.1 随机变量的线性组合**

考虑我们有随机变量$$X_1, X_2, \cdots, X_p$$。我们来考虑获取这些随机变量的一个线性组合的期望和方差。这些随机变量的线性组合写为：$$L(X_1, X_2, \cdots, X_p) = \Sigma_{i=1}^p a_i X_i$$。也就是$$L(\vec{X}) = \vec{a}^T \vec{X}$$，其中$$\vec{a}^T = \left[a_1, \cdots, a_p \right]$$。

从而我们有：$$E \left[ L(\vec{X}) \right] = E \left[ vec{a}^T \vec{X} \right] = \vec{a}^T E \vec{X}$$。

而

$$Var \left[ L(\vec{X}) \right] = E \left[ \vec{a}^T \vec{X} \vec{X}^T \vec{a} \right] - E \left[ \vec{a}^T \vec{X} \right] (E \left[ \vec{a}^T \vec{X} \right]^T$$

$$ = \vec{a}^T E \left[\vec{X} \vec{X}^T \right] \vec{a} - \vec{a}^T E \left[\vec{X} \right] (E \left[\vec{X} \right])^T \vec{a}$$

$$ = \vec{a}^T (E \left[ \vec{X} \vec{X}^T \right] - E \left[\vec{X} \right] (E \left[\vec{X} \right])^T ) \vec{a} = \vec{a}^T Cov(\vec{X}) \vec{a}$$

也就是说，如果知道了$$ E \left[ \vec{X} \right]$$和$$Cov(\vec{X})$$，那么就可以计算$$X_1, X_2, \cdots, X_p$$的线性组合的期望和方差。


**Proposition 1** 如果$$\Sigma$$是一个随机向量的variance-covariance matrix，那么对于任意的向量$$\vec{a}$$，我们都有：$$\vec{a}^T \Sigma \vec{a} \geq 0$$。也就是说，$$\Sigma$$是一个positive semi-definite矩阵。

证明很简单，因为$$\vec{a}^T \Sigma \vec{a}$$是一个随机变量的方差（可以先从随机向量找到$$p$$个随机变量，而这个随机变量就是这些随机变量的线性组合，系数由$$\vec{a}$$表示）。

而且上述定理的反方向也是成立的，也就是说给定一个positive semi-definite的矩阵，其就是某个随机向量的variance-covariance matrix。




---
