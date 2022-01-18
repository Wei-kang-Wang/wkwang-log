---
layout: post
comments: false
title: "Deep Learning with PyTorch"
date: 2021-11-29 01:09:00
tags: book-reading
---

> 这是Deep Learning with PyTorch这本书的翻译版，加上了个人一些拙略的理解。


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

![Whole Process]({{ '/assets/images/DLP-0-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. The Whole Process of Deep Learning.*

## 0.1 序

当我们于2016年中旬开始PyTorch项目的时候，我们还是网上相识的一群搞开源代码的hackers，都想着写一个更好的深度学习软件。这本书的两个作者，Luca Antiga和Thomas Viehmann，在开发PyTorch时贡献了很大的力量并使得它变成今天这样成功。

我们开发PyTorch的目标是建立一个最flexible的框架来实现深度学习算法。PyTorch从Torch7（由Ronan Colobert等人于2007年开始开发）那里继承了很大一部分代码库，并且借用了Lush语言（由Yann LeCun和Leon Bottou开发）的基础概念。这些本就已经有的基础让我们可以更加专注于改进，而不是从头开始造轮子。

PyTorch的成功很难归咎于一个单纯的原因。这个项目提供了很好的用户体验，良好的debug环境以及很强的可塑性，使得用户能够很高效的编程。PyTorch的大量使用造成了一个良好的社区环境，从而促使其更进一步的发展，形成一个良性循环。

这本书Deep Learning with PyTorch提供了一个让我们原作者介绍PyTorch的机会。它涵盖了很多基础的细节，包括tensor，neural network如何实现的细节等等。而且它还涵盖了一些高级的项目，以及如何将PyTorch高效部署等高级知识。

而且这本书还包括了很多应用实例，这样可以让读者更好的理解PyTorch而且在应用中更加熟练。Eli Stevens在软件方面的技巧加上Thomas对PyTorch核心的理解，可以让应用的实现部分更加易懂且深刻。


## 0.2 关于这本书

这本书目的在于介绍PyTorch的基础以及它在日常项目中的应用。我们立志于提供深度学习的关键知识并且展现PyTorch如何使得用户将这些知识应用到现实。在这本书里，我们试图提供支持后续研究的直觉，并且说明代码背后的原理。

Deep Learning with PyTorch这本书并不是想要做成字典类的参考书，而是做成一本介绍概念的书，使得读者能够独立开发自己的项目。所以，我们集中于介绍PyTorch某一方面的知识。最显著的没有介绍的内容就是recurrent neural network。

### 0.2.1 谁该看这本书

这本书是为了想要成为深度学习开发者或者想要了解PyTorch的人所准备的。我们认为这本书典型的读者包括计算机科学家，数据学家，或者软件工程师，或者相关专业的学生。

我们认为读者对命令式编程和面向对象编程有一些基本的了解。因为这本书用的是Python，读者需要对其语法和操作环境有所理解。了解如何安装Python包以及运行脚本是所需要掌握的前行知识。了解Numpy将会很有帮助。同时了解一些linear algebra的基础也是有必要的。

### 0.2.2 这本书是怎么组织的：线路图

Deep Learning With PyTorch包括了三个部分。Part I涵盖了基础知识；Part II从头到尾介绍了一个项目，运用了Part I里的基础知识并加上了一些高级的内容；短小的Part III简单介绍了PyTorch如何部署在设备上作为结尾。

#### 0.2.2.1 Part I

在Part I里，我们开始了认识PyTorch的第一步，建立起理解PyTorch项目的基础知识并为以后写自己的项目做准备。我们介绍的内容涵盖了PyTorch API以及实现PyTorch库的内部机理，并且介绍了如何训练一个分类网络。在Part I结束的时候，我们就已经做好了研究现实项目的能力。

Chapter1 介绍PyTorch是一个库，并介绍了其在深度学习革命中的作用，并且介绍了PyTorch之所以区分于其他深度学习框架的原因。

Chapter2 通过运行pretrained example代码实例来真正接触PyTorch；这章显示了如何在PyTorch Hub里下载以及运行模型。

Chapter3 介绍了PyTorch一个基本的模块：tensor（张量），介绍了它的API以及它背后的实现细节。

Chapter4 阐述了不同类型的数据怎么用tensor来表示，以及各种深度学习模型需要什么格式的tensor。

Chapter5 介绍了通过gradient descent学习的理论，并且介绍了PyTorch如何用automatic differentiation（自动微分）实现它。

Chapter6 用PyTorch里的nn和optim模块介绍了如何构建和训练一个用于回归的模型。

Chapter7 基于之前的知识建立了一个用于图像分类的全连接网络，并且扩展了对PyTorch API的介绍。

Chapter8 介绍了convolutional neural network（卷积神经网络）并介绍了一点构建神经网络模型的高级知识以及它们的PyTorch实现。


#### 0.2.2.2 Part II

在Part II里，每一chapter都让我们对于解决如何自动检测肺癌这个问题更进一步。我们用这个困难的问题作为解释现实中如何解决类似检测肺癌这种复杂问题的动机。

Chapter9 描述了基于CT图像的end-to-end的自动肺癌检测的方法。

Chapter10 加载了具有人工标注的CT照片，并使用标准PyTorch API将相关的信息转化为tensor。

Chapter11 介绍了第一个利用Chapter10中所生成的数据作为输入的分类模型。我们训练这个模型，并记录一些基本的performance metrics。我们同时也介绍了如何用TensorBoard来监控训练过程。

Chapter12 研究并落实了标准performance metrics，并利用这些metrics来鉴别之前训练所体现出的问题。之后我们用一个经过了data balancing and augmentation（数据平衡和扩充）的更好的训练集来解决这些训练设计上的漏洞。

Chapter13 描述了segmentation（分割）任务，利用一个pixel-to-pixel的模型结构来生成一个表示瘤子可能出现位置的p大小和原输入图片一样大的heatmap。这个heatmap可以用来在没有人工标注的CT图像上找到瘤子。

Chapter14 实现了最后的end-to-end项目：在分类任务后接分割模型用来诊断癌症病人。

#### 0.2.2.3 Part III

Part III只有一个章节，用来解释PyTorch的部署。Chapter15 提供了怎么把PyTorch模型部署在网络服务上、将其嵌入一个C++项目或将其部署在移动设备上的一个overview。


### 0.2.3 关于代码

这本书所有的代码都用Python3.6或之后的版本来写。

这本书大量使用Jupyter Notebook来写代码。

下面这段在所有的Jupyter Notebook代码里都被当成第一个模块。

```python
# In[1]:
%matplotlib inline
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)
```


## Part I Core PyTorch





---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

