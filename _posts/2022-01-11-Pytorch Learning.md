---
layout: post
comments: false
title: "Pytorch Learning"
date: 2022-01-11 01:09:00

---

> This post is about Pytorch learning.


<!--more-->

---

## A Talk About Pytorch Internals

这篇文章是ezyang的一篇[talk](http://blog.ezyang.com/2019/05/pytorch-internals/)的翻译并加上自己的理解。此talk主要讲了Pytorch的内部机理。

![Title]({{ '/assets/images/slide1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

这篇文章是为了那些已经用过PyTorch并且希望能够给PyTorch代码库做出贡献却被PyTorch背后的C++代码唬住的人。实话说，有时候PyTorch的代码确实比较难懂。这篇文章的目的是给你一份指南，告诉你tensor 

