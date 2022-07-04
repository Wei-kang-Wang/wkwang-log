---
layout: post
comments: false
title: "Mathematics"
date: 2022-05-30 01:09:00

---

> This post of Mathematics learning.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## 1. 简单数理统计（浙大，概率论与数理统计）

这系列的lecture notes是对统计学知识点的一个梳理（不包含概率论部分），侧重理解而不是严格证明。

### 1.1 Lecture 1

### 1.1.1 大数定律

假设每次独立重复的做同一个实验（比如掷骰子），当实验次数足够多时，每个子事件发生的频率都会无限接近它的概率。大数定律证明了这种现象的客观真实性。

**弱大数定理（Khinchin）**
设$$X_1, X_2, \cdots$$为相互独立且服从同一分布的随机变量序列，且具有数学期望$$\mathbb E(x_k) = \mu (k=1,2,\cdots)$$，计算前$$n$$个变量的算术平均值$$\frac{1}{n} \Sigma_{k=1}^n X_k$$，那么对于$$\forall \epsilon > 0$$，有$$lim_{n \rightarrow \infty} P \lbrace \lvert \frac{1}{n} \Sigma_{k=1}^n X_k - \mu \rvert < \epsilon \rbrace = 1$$。

这个公式很好理解，就是说$$n$$无穷大的时候，这个统计量会和$$\mu$$无限接近。毕竟$$\frac{1}{n}\Sigma_{k=1}^n X_k$$是$$\mu$$的无偏估计。

给出证明之前先介绍一个引理：

**Lemma（Chebyshev inequality）**
$$E(X) = \mu$$，$$D(X) = \sigma^2$$，then $$P \lbrace \vert X - \mu \vert \geq \epsilon \rbrace \leq \frac{\sigma^2}{\epsilon^2}$$

证明：$$P \lbrace \vert X - \mu \vert \geq \epsilon \rbrace = \int_{\vert x - \mu \vert \geq \epsilon} f(x) dx \leq \int_{\vert x - \mu \vert \geq \epsilon} \frac{\vert x - \mu \vert^2}{\epsilon^2} f(x) dx \leq \frac{1}{\epsilon^2} \int_{-\infty}{\infty} (x-\mu)^2 f(x) dx = \frac{\sigma^2}{\epsilon^2}$$。

设$$D(x_k) = \sigma^2$$，然后由于

$$E(\frac{1}{n} \Sigma_{k=1}^n X_k) = \frac{1}{n} \Sigma_{k=1}^n E(X_k) = \frac{1}{n} n \mu = \mu$$

$$D(\frac{1}{n} \Sigma_{k=1}^n X_k) = \frac{1}{n^2} \Sigma_{k=1}^n D(X_k) = \frac{1}{n^2} n \sigma^2 = \frac{\sigma^2}{n}$$

从而，结合Lemma，有：

$$1 - \frac{\sigma^2}{n \epsilon^2} \leq P \lbrace \lvert \frac{1}{n} \Sigma_{k=1}^n X_k - \mu \rvert < \epsilon \rbrace \leq 1$$

令$$n \rightarrow \infty$$，即证。

根据弱大数定理，可以得到下面的推论：

**Bernoulli大数定理**

设$$f_A$$为$$n$$次独立重复实验中事件$$A$$发生的次数，$$p$$为事件$$A$$在每次实验中发生的概率，那么对于$$\forall \epsilon > 0$$，有：

$$lim_{n \rightarrow \infty} P \lbrace \lvert \frac{f_A}{n} - p \rvert < \epsilon \rbrace = 1$$


### 1.1.2 中心极限定理

这个定理说的是，大量的相互独立的随机因素的综合影响形成的结果往往近似的服从正态分布。

**独立同分布的中心极限定理**

设随机变量$$X_1, X_2, \cdots, X_n, \cdots$$相互独立且服从同一分布，有$$E(X_k) = \mu$$，$$D(X_k) = \sigma^2$$，$$k=1,2,\cdots$$，那么随机变量之和$$\Sigma_{k=1}^n X_k$$的标准化变量$$\frac{\Sigma_{k=1}^n X_k - n\mu}{\sqrt{n}\sigma}$$的分布函数$$F(x)$$对于$$\forall x$$，有：

$$lim_{n \rightarrow \infty} F_n(x) = \lim_{n \rightarrow \infty} P \lbrace \frac{\Sigma_{k=1}^n X_k - n\mu}{\sqrt{n}\sigma} \leq x \rbrace = \Phi(x)$$

其中$$\Phi(x)$$是标准正态分布的CDF$$。

下面是一个推论：

**De Moivre-Laplace Theorem**

设随机变量$$\eta_n (n=1,2,\cdots)$$服从参数为$$n,p$$的二项分布，则对于任意$$x$$，有：

$$lim_{n \rightarrow \infty} P \lbrace \frac{\eta_n - np}{\sqrt{np(1-p)}} \leq x \rbrace = \Phi(x)$$


### 1.1.3 分位数和异常值

我们假设存在一个样本$$x_1,x_2,\cdots,x_n$$，样本$$p$$分位数（$$0 < p < 1$$）我们记为$$x_p$$，我们要求它具有以下的性质：

* 至少有$$np$$个观察值小于等于$$x_p$$
* 至少有$$n(1-p)$$个观察值大于等于$$x_p$$

所以箱线图的几个数据是：$$Min, Q_1, M, Q_3, Max$$，其中$$Q_1 = x_{0.25}, Q_3 = x_{0.75}$$

异常值在箱线图中被定义为小于$$Q_1-1.5IQR$$和大于$$Q_3+1.5IQR$$的数据，其中$$IQR = Q_3-Q_1$$。在箱线图中会被以特殊符号标记。


### 1.1.4 常见统计量一览

首先给出统计量的定义。

设$$X_1, X_2, \cdots, X_n$$为来自于总体$$X$$的一个样本，$$g(X_1,X_2,\cdots,X_n)$$为$$X_1,X_2,\cdots,X_n$$的函数，如果$$g$$中不含未知参数，则其被称为是一个统计量。

常见的统计量列举如下：

* $$\bar X = \frac{1}{n} \Sigma_{i=1}^n X_i$$（样本平均值）
* $$S^2 = \frac{1}{n-1} \Sigma_{i=1} (X_i - \bar X)^2 = \frac{1}{n-1} (\Sigma_{i=1}^n X_i^2 - n \bar X^2)$$（样本方差）
* $$S = \sqrt{\frac{1}{n-1} (\Sigma_{i=1}^n X_i^2 - n \bar X^2)}$$（样本标准差）
* $$A_k = \frac{1}{n} \Sigma_{i=1}^n X_i^k, k=1,2,\cdots$$（$$k$$阶原点矩）
* $$B_k = \frac{1}{n} \Sigma_{i=1}^n (X_i - \bar X)^k, k=1,2,\cdots$$（$$k$$阶中心矩）

只需要将对应的观察值$$x_i$$替换回去，就可以得到每一个统计量的对应的样本观察值的表达式。

为什么上述样本方差的分母是$$n-1$$，主要是因为无偏性。

**无偏性**

对于一个参数$$\theta$$的估计量$$\hat \theta$$，如果满足条件$$E(\hat theta) = \theta$$，那么就说这个估计是无偏的。


### 1.1.5 三大抽样分布

我们在做统计推断的时候，总是不可避免的需要使用这些抽样统计量来描述样本的分布。常见的抽样分布都是基于正态分布的样本的。

**卡方分布（$$\Xi^2$$）**

设$$X_1,X_2,\cdots,X_n$$为来自总体$$N(0,1)$$的样本，则称统计量$$\xi^2 = X_1^2 + X_2^2 + \cdots + X_n^2$$服从自由度为$$n$$的$$\Xi^2分布（称为卡方分布），记为$$\Xi^2 \sim \Xi^2(n)$$。

如下是卡方分布的密度函数图像：




### 1.7 参考文献

* https://zhuanlan.zhihu.com/p/29068570
* https://zhuanlan.zhihu.com/p/29091290
* https://zhuanlan.zhihu.com/p/29135727
* https://zhuanlan.zhihu.com/p/29154307
* https://zhuanlan.zhihu.com/p/29208753
* https://zhuanlan.zhihu.com/p/29480270
* https://zhuanlan.zhihu.com/p/29758751


## 2. 抽象代数

### 2.12 参考文献

* https://zhuanlan.zhihu.com/p/30384157
* https://zhuanlan.zhihu.com/p/30496016
* https://zhuanlan.zhihu.com/p/30668738
* https://zhuanlan.zhihu.com/p/30797590
* https://zhuanlan.zhihu.com/p/30845557
* https://zhuanlan.zhihu.com/p/31441459
* https://zhuanlan.zhihu.com/p/31516301
* https://zhuanlan.zhihu.com/p/31909442
* https://zhuanlan.zhihu.com/p/32152134
* https://zhuanlan.zhihu.com/p/32418036
* https://zhuanlan.zhihu.com/p/32530225

## 3. 实分析（Stein, Real Analysis）

### 3.10 参考文献

* https://zhuanlan.zhihu.com/p/31982152
* https://zhuanlan.zhihu.com/p/33180264
* https://zhuanlan.zhihu.com/p/33350896
* https://zhuanlan.zhihu.com/p/33480239
* https://zhuanlan.zhihu.com/p/33616656
* https://zhuanlan.zhihu.com/p/33761437
* https://zhuanlan.zhihu.com/p/33834703
* https://zhuanlan.zhihu.com/p/33866656
* https://zhuanlan.zhihu.com/p/33879863

## 4. 实分析（周民强，实变函数论第三版）

### 4.8 参考文献

* https://zhuanlan.zhihu.com/p/34191290
* https://zhuanlan.zhihu.com/p/34893497
* https://zhuanlan.zhihu.com/p/35378076
* https://zhuanlan.zhihu.com/p/36367639
* https://zhuanlan.zhihu.com/p/36755508
* https://zhuanlan.zhihu.com/p/36921064
* https://zhuanlan.zhihu.com/p/37123013

## 5. 拓扑学（Munkres，Topology）

### 5.7 参考文献

* https://zhuanlan.zhihu.com/p/33143298
* https://zhuanlan.zhihu.com/p/33249330
* https://zhuanlan.zhihu.com/p/33311995
* https://zhuanlan.zhihu.com/p/33444380
* https://zhuanlan.zhihu.com/p/33692808
* https://zhuanlan.zhihu.com/p/33789792

## 6. 拓扑学（尤承业，基础拓扑学讲义）

## 6.8 参考文献

* https://zhuanlan.zhihu.com/p/34986735
* https://zhuanlan.zhihu.com/p/35050567
* https://zhuanlan.zhihu.com/p/35125375
* https://zhuanlan.zhihu.com/p/35177017
* https://zhuanlan.zhihu.com/p/36029065
* https://zhuanlan.zhihu.com/p/35864474
* https://zhuanlan.zhihu.com/p/36722787
* https://zhuanlan.zhihu.com/p/37070641


## 7. 数值优化（Nocedal, Wright, Numerical Optimization (Second Edition)）

### 7.13 参考文献

* https://zhuanlan.zhihu.com/p/118443321
* https://zhuanlan.zhihu.com/p/121001066
* https://zhuanlan.zhihu.com/p/129844645
* https://zhuanlan.zhihu.com/p/143103337
* https://zhuanlan.zhihu.com/p/143535012
* https://zhuanlan.zhihu.com/p/144736223
* https://zhuanlan.zhihu.com/p/158206612
* https://zhuanlan.zhihu.com/p/161390629
* https://zhuanlan.zhihu.com/p/163527928
* https://zhuanlan.zhihu.com/p/165930639
* https://zhuanlan.zhihu.com/p/174245041
* https://zhuanlan.zhihu.com/p/181718998


## 8. 凸优化（CMU Course: Ryan Tibshirani的Convex Optimization）

### 8.13 参考文献

* https://zhuanlan.zhihu.com/p/194308254
* https://zhuanlan.zhihu.com/p/210252556
* https://zhuanlan.zhihu.com/p/226381487
* https://zhuanlan.zhihu.com/p/237582181
* https://zhuanlan.zhihu.com/p/258138843
* https://zhuanlan.zhihu.com/p/259399116
* https://zhuanlan.zhihu.com/p/260819137
* https://zhuanlan.zhihu.com/p/264515249
* https://zhuanlan.zhihu.com/p/265785675
* https://zhuanlan.zhihu.com/p/266625103
* https://zhuanlan.zhihu.com/p/267542995
* https://zhuanlan.zhihu.com/p/268912334


## 9. 随机过程（Richard Durrett, Essentials of Stochastic Processes）

### 9.16 参考文献

* https://zhuanlan.zhihu.com/p/334739650
* https://zhuanlan.zhihu.com/p/335796325
* https://zhuanlan.zhihu.com/p/337130907
* https://zhuanlan.zhihu.com/p/337993832
* https://zhuanlan.zhihu.com/p/339121930
* https://zhuanlan.zhihu.com/p/339972698
* https://zhuanlan.zhihu.com/p/341164317
* https://zhuanlan.zhihu.com/p/341873401
* https://zhuanlan.zhihu.com/p/343585535
* https://zhuanlan.zhihu.com/p/344841436
* https://zhuanlan.zhihu.com/p/345820722
* https://zhuanlan.zhihu.com/p/347895833
* https://zhuanlan.zhihu.com/p/352464021
* https://zhuanlan.zhihu.com/p/353246777
* https://zhuanlan.zhihu.com/p/355142950


## 10. 高等数理统计（茆诗松，概率论与数理统计教程和韦博成，参数统计教程）

### 10.15 参考文献

* https://zhuanlan.zhihu.com/p/70018601
* https://zhuanlan.zhihu.com/p/85026175
* https://zhuanlan.zhihu.com/p/87520809
* https://zhuanlan.zhihu.com/p/87796459
* https://zhuanlan.zhihu.com/p/89868315
* https://zhuanlan.zhihu.com/p/91730443
* https://zhuanlan.zhihu.com/p/94011595
* https://zhuanlan.zhihu.com/p/94590802
* https://zhuanlan.zhihu.com/p/100749535
* https://zhuanlan.zhihu.com/p/104237281
* https://zhuanlan.zhihu.com/p/102550823
* https://zhuanlan.zhihu.com/p/104721146
* https://zhuanlan.zhihu.com/p/105360052
* https://zhuanlan.zhihu.com/p/105714434

---
