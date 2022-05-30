---
layout: post
comments: false
title: "Machine Learning知识点和炼丹技巧"
date: 2021-12-18 01:09:00

---

> This post of various machine learning knowledge.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## 1. Fisher Information（费雪信息）的直观意义

Fisher Information的定义：

假设观察到的i.i.d的数据$X_1, \cdots, X_n$服从一个概率分布$f(X;\theta)$，$\theta$是要求的参数，是个标量。那么似然函数（likelihood）就是$$L(X; \theta) = \Pi_{i=1}^n f(X_i; \theta)$$为了通过maximum likelihood estimation来解$\theta$，需要让log likelihood对于$\theta$的一阶导数等于0，然后得到$\hat{\theta}_{MLE}$。

这个log likelihood的一阶导数也叫做score function，$$S(X;\theta) = \Sigma_{i=1}^{n} \frac{\partial log f(X; \theta)}{\partial \theta}$$

那么Fisher Information，用$I(\theta)$表示，的定义就是这个score function的二阶矩（second moment）：$$I(\theta) = E\left[S(X;\theta)^2\right]$$

一般情况下，很容易证明，$E\left[S(X;\theta)\right] = 0$，从而得到$$I(\theta) = E\left[S(X;\theta)^2\right] - E\left[S(X;\theta)\right]^2 = Var\left[S(X;\theta)\right]$$

于是得到了Fisher Information的第一个数学意义：估计计算MLE的方程的方差。直观表述就是，随着收集的数据越来越多，这个方差由于是independent sum的形式，就会变得越来越大，也就象征得到的信息越来越多。

如果log likelihood二阶可导，在一般情况下可以证明：$$E\left[S(X;\theta)^2\right] = -E\left[\frac{\partial^2}{\partial \theta^2} log L(X;\theta)\right]$$

于是就有了Fisher Information的第二个数学意义：log likelihood在某个参数$\theta$处的负二阶导数的期望。下面对其进行解释。

一个normalized Bernoulli log likelihood的函数图像如下图所示：

对于这样一个log likelihood function，它越平越宽，就代表我们对参数的估计能力越差，它越高越窄，就代表我们对参数的估计能力越好，也就是信息量越大。这个log likelihood function在MLE的结果$\hat{\theta}$处（也就是顶点处）的弯曲程度，就是通过在这个点处的负二阶导数来表示的。弯曲程度越大，整个log likelihood的形状就越偏向于高而窄，也就代表掌握的信息越多。

在一般情况下，通过对score function在真实值处的泰勒展开，应用中心极限定理，弱大数定律，依照概率一致收敛，以及Slutsky定理，可以证明MLE的渐进分布的方差是$I^{-1}(\theta)$，即$Var(\hat{\theta}_{MLE}) = I^{-1}(\theta)$




---
