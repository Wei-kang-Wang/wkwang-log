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

## 序

当我们于2016年中旬开始PyTorch项目的时候，我们还是网上相识的一群搞开源代码的hackers，都想着写一个更好的深度学习软件。这本书的两个作者，Luca Antiga和Thomas Viehmann，在开发PyTorch时贡献了很大的力量并使得它变成今天这样成功。

我们开发PyTorch的目标是建立一个最flexible的框架来实现深度学习算法。PyTorch从Torch7（由Ronan Colobert等人于2007年开始开发）那里继承了很大一部分代码库，并且借用了Lush语言（由Yann LeCun和Leon Bottou开发）的基础概念。这些本就已经有的基础让我们可以更加专注于改进，而不是从头开始造轮子。

PyTorch的成功很难归咎于一个单纯的原因。这个项目提供了很好的用户体验，良好的debug环境以及很强的可塑性，使得用户能够很高效的编程。PyTorch的大量使用造成了一个良好的社区环境，从而促使其更进一步的发展，形成一个良性循环。

这本书Deep Learning with PyTorch提供了一个让我们原作者介绍PyTorch的机会。它涵盖了很多基础的细节，包括tensor，neural network如何实现的细节等等。而且它还涵盖了一些高级的项目，以及如何将PyTorch高效部署等高级知识。

而且这本书还包括了很多应用实例，这样可以让读者更好的理解PyTorch而且在应用中更加熟练。Eli Stevens在软件方面的技巧加上Thomas对PyTorch核心的理解，可以让应用的实现部分更加易懂且深刻。


## 关于这本书

这本书目的在于介绍PyTorch的基础以及它在日常项目中的应用。我们立志于提供深度学习的关键知识并且展现PyTorch如何使得用户将这些知识应用到现实。在这本书里，我们试图提供支持后续研究的直觉，并且说明代码背后的原理。

Deep Learning with PyTorch这本书并不是想要做成字典类的参考书，而是做成一本介绍概念的书，使得读者能够独立开发自己的项目。所以，我们集中于介绍PyTorch某一方面的知识。最显著的没有介绍的内容就是recurrent neural network。

### 谁该看这本书

这本书是为了想要成为深度学习开发者或者想要了解PyTorch的人所准备的。我们认为这本书典型的读者包括计算机科学家，数据学家，或者软件工程师，或者相关专业的学生。

我们认为读者对命令式编程和面向对象编程有一些基本的了解。因为这本书用的是Python，读者需要对其语法和操作环境有所理解。了解如何安装Python包以及运行脚本是所需要掌握的前行知识。了解Numpy将会很有帮助。同时了解一些linear algebra的基础也是有必要的。

### 这本书是怎么组织的：线路图

Deep Learning With PyTorch包括了三个部分。Part I涵盖了基础知识；Part II从头到尾介绍了一个项目，运用了Part I里的基础知识并加上了一些高级的内容；短小的Part III简单介绍了PyTorch如何部署在设备上作为结尾。

#### Part I







---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

