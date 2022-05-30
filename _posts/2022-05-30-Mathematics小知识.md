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




---
