---
layout: post
comments: false
title: "PyTorch Learning"
date: 2022-01-11 01:09:00

---

> This post is about PyTorch learning.


<!--more-->

---

## A Talk About PyTorch Internals

![1]({{ '/assets/images/slide1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

这篇文章是ezyang的一篇[talk](http://blog.ezyang.com/2019/05/pytorch-internals/)的翻译并加上自己的理解。此talk主要讲了PyTorch的内部机理。

![2]({{ '/assets/images/slide2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

这篇文章是为了那些已经用过PyTorch并且希望能够给PyTorch代码库做出贡献却被PyTorch背后的C++代码唬住的人。实话说，有时候PyTorch的代码确实比较难懂。这篇文章的目的是给你一份指南，告诉你tensor library that supports automatic differentiation的基本概念框架，并且教会你如何找到自己的方式来了解学习代码库。这篇文章假设你已经写过一些PyTorch代码，但是还没有完全明白机器学习代码库是怎么写的。

![3]({{ '/assets/images/slide3.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

这篇文章主要分为两部分（Concepts和Mechanics）。在第一部分里，我将会介绍tensor library的基本知识。从所熟知的tensor数据类型开始，并且仔细说明这个数据类型到底为我们提供了哪些功能和内容。这可以让我们对于代码背后到底发生了什么有更好的理解。我们还会讨论extension points的一体三面，layout，device以及dtype，这可以让我们更深的理解tensor类的扩展。最后我还会讨论一些autograd的内容。在第二部分里，我们将会讨论PyTorch代码里复杂难搞的细节。我会告诉你如何在大片的autograd代码里找到需要的部分，以及哪一部分是核心代码、PyTorch给你提供的写扩展代码的工具。

## 第一部分：Concepts

![4]({{ '/assets/images/slide4.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

### Tensor

![5]({{ '/assets/images/slide5.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

Tensor是PyTorch里最核心的数据结构。你可能对一个tensor长什么样有着良好的直觉猜想：一个n维的数据结构，包含着很多标量类型的数据，比如floats，ints等。我们可以认为一个tensor是一个含有一些数据，同时也含有一些metadata用来描述这个tensor的大小，所含的数据的类型（dtype），这个tensor分布在什么设备上（CPU memory或者是CUDA memory）等的综合体。

![6]({{ '/assets/images/slide6.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

有些metadata你可能不是很了解：stride。Strides实际上是PyTorch一个很独特的特征，所以值得单独讨论一下。

![7]({{ '/assets/images/slide7.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

一个tensor是一个数学概念。但是为了在计算机上表达它，我们需要定义某种physical representation。最常见的representation是将tensor的每个值连续的部署在memory里，将每一行都写在memory里，正如上图所示的那样。在上述例子中，我规定了tensor包含32bit的整数类型，所以你可以看到每个整数部署在一个物理地址里，每一个物理地址都相隔4bytes。为了记住tensor的实际维度，我们还需要将每个维度的size记录下来作为metadata。（部署在memory的时候，是将高维的tensor按某种规则展开成了一维）

所以说，stride起到了什么作用呢？

![8]({{ '/assets/images/slide8.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

假设我们想要获得tensor$$\left[0,1\right]$$位置的值，我们该如何将这个逻辑上的index$$\left[0,1\right]$$转换到memory的物理地址呢？stride告诉我们如何实现：为了找到tensor每个位置的元素的物理地址，我们将逻辑上的index分别乘以该维度上的stride，并加在一起。在上图里，第一个维度涂了蓝色，第二个维度涂了红色，从而可以清晰的看到计算过程。在上例中，tensor$$\left[0,1\right]$$所得到的sum是2，而且该tensor的部署方式是连续的，所以tensor$$\left[0,1\right]$$应该在从头数两个bytes之后，位于第三个byte的物理地址上。

在这篇talk之后，我还会介绍TensorAccessor，是一个类，用来解决index计算。当你看到了TensorAccessor而不是原始指针，那index的计算就已经被算好了。

stride是PyTorch实现为用户提供操作tensor的一个重要基础。比如，我们想要看看上述tensor第二行所表示的tensor长什么样，

![9]({{ '/assets/images/slide9.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

用上层高级index代码，我可以直接用tensor$$\left[1,:\right]$$来获取这一行。当我写这个代码的时候，我并没有真的创建了一个新的tensor，而是返回了一个原有tensor部分数据的一个新的view。这表明如果我在这个view里编辑了数据，我将会改变原有tensor该位置的值。







