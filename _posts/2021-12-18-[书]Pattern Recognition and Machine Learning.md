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

在这本书里，向量vectors使用小写加粗罗马字母表示，比如 $$\pmb{x}$$ ，所有的vectors都是列向量。$$\pmb{x}^T$$ 表示行向量，因为$$T$$表示转置。大写的加粗罗马字母表示矩阵，比如 $$\pmb{M$}$ 。$$(w_1, \cdots, w_M)$$表示有$$M$$个元素的行向量，从而对应的列向量为 $\pmb$w}$$ = $$(w_1, \cdots, w_M)^T$$。

记号$$\left[a, b \right]$$表示从$$a$$到$$b$$的闭区间，而$$(a,b)$$则表示从$$a$$到$$b$$的开区间。

$$M \times M$$的identity矩阵（也叫做unit矩阵）记为$$\pmb{I_M}$$。

一个functional被记为$$f\left[y\right]$$，其中$$y(x)$$也是一个function。

$$g(x) = O(f(x))$$表示$$\lVert f(x) / g(x) \rVert$$在$$x \longrightarrow \infty$$的时候是有界的。

函数$$f(x,y)$$关于随机变量$$x$$的期望被记为$$\mathbb{E_x} \left[f(x,y)\right]$$。






---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
