---
layout: post
comments: false
title: "[书]Pattern Recognition and Machine Learning"
date: 2021-12-17 01:09:00
tags: book-reading
---

> 这是*PATTERN RECOGNITION AND MACHINE LEARNING*的读书笔记.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## Mathmatical Notation

calculus，linear algebra和probability theory是阅读这本书的数学基础。

在这本书里，向量vectors使用小写加粗罗马字母表示，比如 $$\pmb{x}$$ ，所有的vectors都是列向量。$$\pmb{x}^T$$ 表示行向量，因为$$T$$表示转置。大写的加粗罗马字母表示矩阵，比如 $$\pmb{M}$$ 。$$(w_1, \cdots, w_M)$$表示有$$M$$个元素的行向量，从而对应的列向量为 $$\pmb{w} = (w_1, \cdots, w_M)^T$$。

记号$$\left[a, b \right]$$表示从$$a$$到$$b$$的闭区间，而$$(a,b)$$则表示从$$a$$到$$b$$的开区间。

$$M \times M$$的identity矩阵（也叫做unit矩阵）记为$$\pmb{I_M}$$。

一个functional被记为$$f\left[y\right]$$，其中$$y(x)$$也是一个function。

$$g(x) = O(f(x))$$表示$$\lVert f(x) / g(x) \rVert$$在$$x \longrightarrow \infty$$的时候是有界的。

函数$$f(x,y)$$关于随机变量$$x$$的期望被记为$$\mathbb{E_x} \left[f(x,y)\right]$$。如果不造成歧义的话，可以省去下标，比如$$\mathbb{E} \left[g(x)\right]$$。如果$$x$$的分布是在另一个随机变量$$z$$的条件下的，那么这个条件期望被记为$$\mathbb{E_x} \left[f(x) \vert z\right]$$。类似的，variance被记为$$var\left[f(x)\right]$$。而对于向量型的随机变量，covariance则被记为$$cov\left[\pmb{x}, \pmb{y}\right]$$。我们使用$$cov\left[\pmb{x}\right]$$来表示$$cov\left[\pmb{x}, \pmb{x}\right]$$。

如果我们对于某个$$D$$维的向量$$\pmb{x} = (x_1, \cdots, x_D)^T$$有$$N$$个值$$\pmb{x_1}, \cdots, \pmb{x_N}$$，我们可以将其放在一个矩阵$$\pmb{X}$$里，其第$$n$$行就是向量$$\pmb{x_n}^T$$。如果是一维的随机变量，我们上述的矩阵就变成了向量，记为**x**，其是一个列向量，第$$n$$个位置的元素是$$x_n$$。注意到**x** 和$$\pmb{x}$$是不一样的，前者是$$N$$个随机向量放在一起组成的列向量，后者自身就是一个随机向量。


## 1. Introduction

考虑一个识别手写数字图像的例子，如fig1所示。每个数字图像都是一个$$28 \times 28$$的pixel矩阵，我们将其拉成一个长度为784的向量$$\pmf{x}$$。我们的目的是设计一个算法来自动学习不同的手写数字图片的特征，从而能自动判定输入的手写数字的值。hand-crafted特征是可以用来辨别手写数字图片的，但其对于复杂的情况效果不好，而且需要大量手动设计的features。

用来训练上述这个自动检测的模型的参数的数据的集合$$\lbrace \pmb{x_1}, \cdots, \pmb{x_N} \rbrace$$叫做训练集，training set。







---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
