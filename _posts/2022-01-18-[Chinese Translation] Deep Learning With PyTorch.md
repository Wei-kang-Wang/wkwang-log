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

这是本书的第一部分，从此我们开始正式介绍PyTorch的内容，学习理解PyTorch的机理并理解PyTorch项目的机制。

在Chapter1里我们将会第一次接触PyTorch，理解它是什么，它用来解决什么问题，以及它和别的深度学习框架是什么联系的。Chapter2将会带领读者盘一些已经pretrained的model。Chapter3进一步接触深入的内容，并说明了PyTorch项目里所用的最基础的数据结构：tensor。Chapter4 教会读者如何将不同领域问题的数据表达成PyTorch tensor的形式。Chapter5 展示了深度学习模型是如何从实例中学习到知识的，并说明PyTorch是怎么实现这个过程的。Chapter6 介绍了神经网络组成的基础知识，以及如何用PyTorch构建一个神经网络。Chapter7 介绍了一个利用神经网络实现简单的图像分类任务。Chapter8 介绍了Chapter7里同样的任务如何用卷积神经网络更聪明的实现。

在Part I学习结束之后，在part II里我们就可以用PyTorch应对真实的问题了。

## Chapter 1 Introducing deep learning and the PyTorch Library

>这章包括的内容
>* 深度学习是怎么改变我们处理机器学习任务的方式的。
>* 理解为什么PyTorch很适合深度学习。
>* 检查一个典型的深度学习项目。
>* PyTorch所需的硬件要求。


人工智能是个定义的很差的词，联系到很多的研究，检查，疑惑，fancy的宣传，以及科幻性的制造恐慌。现实当然是更加令人安心的。很难说现在的机器已经可以达到人类的思考水平，更多的是我们发现了一系列通用的算法，它们可以很高效近似复杂的、非线性的过程，从而替代人类自动解决之前只有人类可以解决的问题。

在某种程度上，我们认识到intelligence（智能）是一个和self-awareness（自我意识）相同的概念，而自我意识对于能否解决这些任务来说肯定是不必要的。到了后来，计算机智能这个问题甚至已经变得不重要了。

>Edsger W. Dijkstra认为机器是否能思考这个问题就像是问潜水艇是否会游泳。

我们这本书所要讨论的一大类通用的算法是AI下面一个叫做深度学习的子类，它旨在利用具有指导性的样本来训练深度神经网络（一个数学上的概念）。深度学习利用大量的数据来近似复杂的函数，该函数的input（输入）和output（输出）可以差的很远，比如输入可以是一张图片，而输出是一行描述该输入的文字。这种学习能力让我们可以构建具有处理人类才能解决的问题的代码。

### 1.1 深度学习革命

为了理解深度学习造成了什么样的改变，我们先往前退一步来看。直到上一个十年（2010年左右），绝大部分的机器学习系统还很依赖feature engineering（特征工程）。特征是为了下游任务而对输入数据做的transformation（变形或转换），下游任务比如说一个可以给出正确输出的分类器。特征工程核心在于如何选择一个能使得下游任务能被解决的合适的变形。比如，为了能够区分手写数字0和1的图片，我们可以利用一系列filters（过滤器）来估计图像中边缘的方向，然后训练一个分类器利用边缘方向的分布规律来给出预测结果。另一种可能有效的特征是记录图中围成的洞的个数（0有一个，1有0个）。

相对比来说，深度学习为了成功解决一个任务，直接从原始数据中自动的寻找这样的representation（特征表示）。在判断输入手写数字图片是0还是1的任务里，过滤器在训练的过程中被成对的带有标签的数据反复迭代的更新。这并不是说feature engineering在深度学习中就没有用了；我们经常需要在learning system里插入一些prior knowledge。然而，neural networks基于数据实例摄取数据而提取有用的representation的能力是深度学习最有力的地方。深度学习实践者所关注的重点并不是如何handcrafting这些representations，而是如何使得neural network能基于训练数据来自动的找到这样的representations。而这些自动生成的特征比那些手动的特征往往要更好！因为更多其它技术上的革新，这个事实使得整个机器学习community的想法和观点都有所改变了。


![Revolution]({{ '/assets/images/DLP-1-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Deep Learning exchanges the need to handcraft features for an increase in data and computational requirements.*

如上图右边所示，一个操作员忙碌的定义features并将它们给一个learning algorithm作为输入；而这个任务效果的好坏取决于这个操作员所定义的features的好坏。再上图左侧，利用深度学习，原始的数据直接给了一个algorithm，从而自动得到一系列hierarchical的特征，而这个过程由优化这个任务本身的performance来指导；这个任务效果的好坏将会取决于操作员如何设计这个algorithm。

从上图左侧我们已经初步看到了成功执行一个深度学习任务所需要的东西：
* 我们需要一个方法来ingest手中所拥有的data。
* 我们需要定义一个deep learning machine。
* 我们必须有一个自动的训练过程，用来获得有用的representations以及使得算法输出所需要的输出。

我们更进一步的来看一下训练的过程。在训练过程中，我们需要用一个criterion（尺度，标准），这是一个以模型输出和训练数据为输入的实值函数，我们需要用它来对所想要的输出和实际的输出之前的差值给出一个数值上的数据（一般来说这个值越低越好）。训练就是使得这个值在不断的调整模型参数的过程中越来越低，甚至在训练中没看到的数据上仍然很低（这样就达到了学习的效果）。


### 1.2 用于深度学习的PyTorch

PyTorch是一个用于构建深度学习项目的Python program的库。它强调了flexibility并且允许以Python风格来编写深度学习模型。这种易接近性和易于实用性在研究者团体中找到了早期的追随者，然后在第一版本面世过去的几年里，发展成为了一个具有广泛应用的主流深度学习框架。

正如Python对于programming，PyTorch给深度学习提供了一个很好的介绍。同时，PyTorch也被证明够格用于真实世界里的专业任务。我们相信PyTorch的清晰的语法，合理的API，以及容易上手的debugging使得它是深度学习框架的不二之选。我们十分建议将PyTorch作为第一个深度学习代码库来学习。至于这是不是你将来最后选择使用的库这完全取决于你自己。

在figure1左侧的deep learning machine是一个很复杂的函数。为了能表达这个函数，PyTorch提供了一个核心的数据结构，tensor。tensor是一个和Numpy ndarray很类似的multi-dimensional array。在这个基础数据结构（tensor）周围，PyTorch提供了可以部署在复杂硬件上进行各种加速数学运算的features，使得我们可以很容易的设计neural network的架构以及在单个机器或者并行计算资源上进行训练。

这本书立志于为软件工程师、数据科学家以及熟练掌握Python的学生提供一个学会利用PyTorch构建深度学习项目的起点。我们希望这本书可以既清晰易懂由足够有用，而且我们希望读者能理解书中概念并应用于实际项目中。为了达到这个目标，我们用了一种hands-on的方法并且希望读者已经准备好时刻在电脑上将书中的实例实现。在这本书的基础上，再加上优秀的official documentation，我们希望读者能够自己设计并实现各种项目。

为了能更好学习这本书的内容，你需要做到以下两点：
* 熟悉Python编程。
* 愿意在自己电脑上写代码，get your hands dirty。

这本书由三个不同的部分构成。Part I涵盖了基础知识，将PyTorch如何用代码实现figure1左侧的流程的细节进行了解释。Part II带读者从头到尾过了一个end-to-end的项目：找到并分类CT扫描图片里的肿瘤。此项目基于Part I里的基础知识而构建，并加了一些更多的高级的内容。短暂的Part III简短的介绍了PyTorch怎么把深度学习模型部署在设备上。

深度学习是一个很大的话题。在这本书里，我们仅仅包含了其中很小的一部分：用PyTorch以图片作为输入实现小规模的分类和分割任务。这本书重点在于PyTorch的实践，目的在于提供足够读者能解决实际问题的知识，比如cv，nlp，等。


### 1.3 为什么要选择PyTorch

正如之前所说，深度学习通过学习数据让我们能够解决一系列复杂的问题，比如机器翻译、玩策略类游戏等。为了能在实践中实现问题的解决，我们需要flexible的工具，所以他们可以被应用于一系列范围很广的问题里；而且可以利用大量的数据进行很多次的训练，而且我们还需要训练的网络对于各种各样的输入都能够表现得很好。基于上面这些目的，让我们来看看为什么我们需要用PyTorch。

PyTorch足够简单。很多研究员和程序员觉得它很容易学而且上手。PyTorch是Pythonic的，而且对于复杂的领域，他也有着充足的文档，对错误有所警告，也给出了实例。更具体地说，在PyTorch里编码深度学习框架是十分自然地。PyTorch给了我们一种data type（数据类型），Tensor，来处理numbers，vectors，matrices，或者最广义的arraies。而且，PyTorch还提供了能够操作Tensor的函数。我们可以就像在Python里一样应用这些函数。如果读者了解Numpy，那这些就显得很自然。

PyTorch还提供了两个特征使得其非常适合深度学习：首先，它支持利用GPU进行加速计算，经常比在CPU上进行同样的计算要快几十倍。其次，PyTorch支持对一般数学表达式进行数值优化，这被用于深度学习的训练。我们可以注意到，这两个特征对于一般的scientific computing都很有用，不仅仅只是针对于深度学习问题。实际上，我们完全可以将PyTorch描述为一个支持优化和科学计算的高性能的Python库。

一个PyTorch设计的驱动力是它的表达性，允许开发者实现足够复杂的模型，而不需要因为库本身的复杂性而困扰。PyTorch毫无疑问的提供了提供了一种在深度学习的范围内，将idea用Python代码实现的手段。

PyTorch在从研究和开发转向工业界的路上仍有一些很有意思的故事。因为它起初是为了研究而开发的，PyTorch一开始是基于C++而开发的，这样它就可以利用C++的快速性而且不需要依赖Python库，而且训练和部署都是用C++做的。

#### 1.3.1 十分具有竞争力的深度学习框架领域

在PyTorch最初beta版本发行的时候：
* Theano和TensorFlow已经成为了low-level库的翘楚，它们都是由用户定义一个computational graph来定义模型。
* Lasagne和Keras是Theano high-level的wrapper，而且Keras还是wrapper了TensorFlow和CNTK。
* Caffe，Chainer，DyNet，Torch（基于Lua，是PyTorch的先驱），MXNet，CNTK，DL4J都是当时流行的框架。

在接下来的两年里，这个领域发生了很大的变化。PyTorch和TensorFlow成为了仅有的两大巨头，其它的框架仅仅在特定的领域才会被使用到。
* Theano：第一个深度学习框架，已经停止了开发。
* TensorFlow：完全吸收了Keras，并将其作为TensorFlow最高层的API；提供了一个中间层，eager mode，类似于PyTorch的计算逻辑；发行了TensorFlow 2.0。
* JAX：也是Google开发的库，完全独立于TensorFlow，可以被认为是具有autograd，JIT和GPU版本的Numpy三种功能的库。
* PyTorch：吸收了Caffe2；更换了绝大多数用Lua写的Torch项目的底层代码；加上了一个graph mode，叫做TorchScript。

TensorFlow拥有一个健壮的可用于生产的pipeline，一个广泛的工业界的环境，以及大量的市场占有率。PyTorch因为易于上手获得了很多研究员、学生老师的青睐，并且随着他们进入工业界从而开始流行。TorchScript和TensorFlow的eager mode可以认为是它们双方互相向对方学习的点。


### 1.4 PyTorch如何支持deep learning项目的综述

我们已经暗示了一些PyTorch的building blocks。现在我们正式的构建一个PyTorch high-level的map。我们可以从一个深度学习项目需要从PyTorch中得到什么来研究这个问题。

首先，PyTorch以从Python中继承的Py作为开头，但是仍然有很多底层代码并不是由Python所写。实际上，为了performance，绝大多数PyTorch的底层代码都是用C++和CUDA（一个由NVIDIA开发的类似于C++的语言，被编译后可以进行GPU上大型的并行计算）写的。PyTorch提供了用C++直接运行的方式，将在Chapter15里说。提供这种能力的一个原因是它使得在设备上部署模型成为可能。然而绝大多数时间我们还是从Python的角度来研究PyTorch，包括构建模型，训练，以及用训练好的模型来解决实际问题。

PyTorch的Python API是解释PyTorch作用以及将其与Python生态系统结合的地方。

正如我们已经了解的，在它的核心，PyTorch提供了multi-dimensional arrays（或者在PyTorch的语言里，tensors）的库（我们将会在Chapter3里仔细研究细节），也提供了tensors上各种不同的operations的库，在torch模块里。tensors和他们的operations都可以在CPU和GPU上进行运算。将计算从CPU迁移到GPU上用PyTorch实现只需要一到两行代码。PyTorch另一个核心就是它给tensor提供了一种能力，这种能力可以让每个tensor追踪定义在其上的运算，并且给出任何一个output关于其任意一个input的derivative。这个将会被用于数值优化，这些derivative由tensor来提供，但实际上由PyTorch的autograd模块来进行计算，并提供给tensor。

通过拥有tensor和autograd-enabled tensor标准库，PyTorch可以被用于物理，渲染，优化，仿真，建模，以及更多。但是PyTorch最重要的就是它是一个深度学习库，所以它提供了所有构建以及训练一个neural networks所需要的building blocks。下图展示了一个标准的过程，包括了数据loading，模型训练，以及部署模型。

![Framework]({{ '/assets/images/DLP-1-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Basic, high-level structure of a PyTorch project, with data loading, training, and deployment to production.*

构建neural networks的最核心的PyTorch module被放在torch.nn里，这个module提供了常见的neural networks layer以及其它的架构组成部分。fully connected layer，convolutional layer，activation function，loss function也可以在这个module里被找到。这些组成部分可以被用于构建以及初始化未被训练的模型（figure2里中间部门的untrained model）。为了训练我们的模型，我们需要额外几件事情：一个source of training data，一个optimizer用来将这个模型适用于这些training data，以及将模型和数据部署到硬件设备上的方法（这样才能真正的开始训练和计算）。











---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

