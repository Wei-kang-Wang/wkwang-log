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


### 1.4 综述：PyTorch如何支持deep learning项目

我们已经暗示了一些PyTorch的building blocks。现在我们正式的构建一个PyTorch high-level的map。我们可以从一个深度学习项目需要从PyTorch中得到什么来研究这个问题。

首先，PyTorch以从Python中继承的Py作为开头，但是仍然有很多底层代码并不是由Python所写。实际上，为了performance，绝大多数PyTorch的底层代码都是用C++和CUDA（一个由NVIDIA开发的类似于C++的语言，被编译后可以进行GPU上大型的并行计算）写的。PyTorch提供了用C++直接运行的方式，将在Chapter15里说。提供这种能力的一个原因是它使得在设备上部署模型成为可能。然而绝大多数时间我们还是从Python的角度来研究PyTorch，包括构建模型，训练，以及用训练好的模型来解决实际问题。

PyTorch的Python API是解释PyTorch作用以及将其与Python生态系统结合的地方。

正如我们已经了解的，在它的核心，PyTorch提供了multi-dimensional arrays（或者在PyTorch的语言里，tensors）的库（我们将会在Chapter3里仔细研究细节），也提供了tensors上各种不同的operations的库，在torch模块里。tensors和他们的operations都可以在CPU和GPU上进行运算。将计算从CPU迁移到GPU上用PyTorch实现只需要一到两行代码。PyTorch另一个核心就是它给tensor提供了一种能力，这种能力可以让每个tensor追踪定义在其上的运算，并且给出任何一个output关于其任意一个input的derivative。这个将会被用于数值优化，这些derivative由tensor来提供，但实际上由PyTorch的autograd模块来进行计算，并提供给tensor。

通过拥有tensor和autograd-enabled tensor标准库，PyTorch可以被用于物理，渲染，优化，仿真，建模，以及更多。但是PyTorch最重要的就是它是一个深度学习库，所以它提供了所有构建以及训练一个neural networks所需要的building blocks。下图展示了一个标准的过程，包括了数据loading，模型训练，以及部署模型。

![Framework]({{ '/assets/images/DLP-1-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Basic, high-level structure of a PyTorch project, with data loading, training, and deployment to production.*

构建neural networks的最核心的PyTorch module被放在torch.nn里，这个module提供了常见的neural networks layer以及其它的架构组成部分。fully connected layer，convolutional layer，activation function，loss function也可以在这个module里被找到。这些组成部分可以被用于构建以及初始化未被训练的模型（figure2里中间部门的untrained model）。为了训练我们的模型，我们需要额外几件事情：一个source of training data，一个optimizer用来将这个模型适用于这些training data，以及将模型和数据部署到硬件设备上的方法（这样才能真正的开始训练和计算）。

在figure2的左侧，我们可以看到在training data到达模型之前，需要进行一些data processing。这种data processing并不是preprocessing，而是对数据的处理，从而使得能够适应PyTorch的训练模型，这在真实的项目中占有十分重要的位置（原始数据必须变成PyTorch能操作的tensor才可以）。首先，我们需要physically获取数据，大多数情况下是从某种存储设备中获取的。然后我们需要将数据里的每一个sample都转换成PyTorch能够实际上操作的东西：tensor。这个在用户定义的数据（可以是任意形式的）和标准的PyTorch tensor中充当桥梁作用的是定义在torch.utils.data module里的Dataset类。这个过程对于每个任务都不太一样，我们需要自己来实现。本书将会在Chapter4里展示各种不同类型的数据该如何表示为tensor的形式。

因为数据存储通常都很慢，特别是由于access latency，我们通常希望并行化处理data loading。但是Python并不包含并行处理的函数，所以我们需要额外的步骤来操作，将数据整合为batches（批次）：包含多个samples的tensor。这样做很细致，但同样也足够一般化，所以PyTorch在DataLoader类中提供了该方法。这个类的实例可以大量产生子过程用来从dataset里load data，之后就可以被training loop使用了。

Dataset和DataLoader的内容将会在Chapter7中介绍。

>DataLoader会用在Dataset处理之后。Dataset负责将各种类型的数据转化为tensor，DataLoader负责将转化后的tensor整合为batch，仍然是个tensor。

当可以生成sample batches的部分定义好之后，我们就可以来处理training loop本身了。一般情况下，这个training loop都是一个标准的Python for循环。在最简单的情况下，模型在本地CPU或单个GPU上完成计算。这种情况也是我们这本书里所假设的情况。

在training loop的每一步，我们都要用从data loader里获取的sample来评价我们的模型。我们可以用某种criterion或者loss function来比较模型的输出和想要的输出之间的差别。就像PyTorch为我们提供了构建model的函数，他也给我们提供了很多loss function函数或类。这些同样在torch.nn模块里。在我们比较完模型的输出和想要的输出后，我们需要将模型调整一点从而使得它的输出更加靠近所想要的输出。正如之前所提到的，这就是PyTorch autograd所起作用的地方；但我们仍然需要一个optimizer来做更新（因为autograd只是整合了gradient），这些optimizer被放在了torch.optim里。我们将会在Chapter5里开始用loss functions和optimizers，在Chapter6-8里进一步熟悉这些内容，然后在Part II里正式接触大的项目。

现在用更复杂精细的硬件系统变得越来越常见了，比如多块GPU，或者多个设备共同训练一个大的模型，如figure2下面所显示的那样。在这些情况里，torch.nn.parallel.Distributed-DataParallel和torch.distributed子模块可以被用来处理复杂硬件的部署。

training loop可能是整个深度学习任务里最无聊但耗时最长的部分。在训练结束后，我们将会得到一个模型，它的参数已经在我们的任务中优化好了：如figure2里的trained model。为了让这个模型变得有用，我们需要将它合理的放置部署。这个过程在figure2的右边进行了描述，一般涉及将模型放在服务器或上传到云端，或者将其与一个更大的项目相整合，或者集成到手机等等。

放置部署训练好的模型里特殊的一种是将训练好的模型导出（export the model）。正如之前所提到的，PyTorch默认导出到一个立即可以执行的模型（eager mode）。每当一个有关PyTorch的指令在Python interpreter里出现了，相对应的操作就会通过C++或者CUDA立刻执行。随着更多的tensor上的指令提出，更多的底层的操作将会被执行。

PyTorch还提供了一种方法可以预先编译模型：TorchScipt。用TorchScript，PyTorch可以将模型序列化成一系列指令，这些指令可以在与Python解释器完全独立的情况下被调用：比如在C++代码里或者在移动设备上。我们可以将其想象成一种拥有有限指令集的虚拟机，特别用于tensor operations。这可以让我们将模型正常的导出，也可以导出为一种叫ONNX的标准形式。这些内容是PyTorch在生产部署的基础，将在Chapter15里说。





## Chapter 2 Pretrained networks

>这章包括的内容：
>* 运行预训练好的图像识别模型
>* GAN和CycleGAN的一个介绍
>* 生成图片文字描述的Captioning模型
>* 通过Torch Hub分享模型

computer vision毋庸置疑是一个被深度学习深深影响的领域，其中有很多的原因：分类或者从原始图片中理解图像内容的需求是存在的，大量的数据变得可以获得，以及新的网络结构比如convolutional layer被发明出来，而且GPU被应用于加速计算。

我们将会学习如何下载和运行其他的研究者已经在大规模数据库上训练好的模型。我们可以把训练好的模型理解成为一段program，能够输入input并输出相应的output。这段program的行为被网络结构、训练数据以及输出所需要满足的条件所精心设计。

在这章里，我们将会三种流行的预训练模型：一个可以根据图像内容给图像贴上标签的模型，一个可以从一张真实图片生成新图片的模型，以及一个可以用句子描述图像内容的模型。我们将会学习如何在PyTorch里下载并运行这些模型。我们也会介绍PyTorch Hub，它是一个工具集，PyTorch模型可以在这个标准化的用户界面上被处理。沿着这个方向，我们将会讨论data source，定义如label这种的术语。

学习如何用PyTorch来运行一个训练好的模型是很有用的，尤其是当这个模型在一个很大的数据集上被训练过。我们将会习惯于给真实世界的数据设计并运行neural networks，不论这个networks是否经过了训练。

### 2.1 一个能够识别图像主体内容的预训练模型

作为我们对deep learning的第一个了解，我们将会运行一个deep neural network，这个网络被object recognition任务预训练过。有很多预训练的网络可以被源代码仓库所获取。研究者们随着论文的发表而公布自己的源代码是很常见的，而有些研究团队还会提供已经预训练好的网络。用这些预训练好的模型可以让我们直接利用这些模型完成一些任务，或者进行进一步的设计。

我们这里所用的预训练模型是在ImageNet这个数据集上预训练的。ImageNet是Stanford University做的一个超过14 000 000张图片的数据集，它的每张图片都有分层级的名词标签，这些标签来自于WordNet数据集，这是一个英文的大型词汇数据集。

ImageNet数据集，和其它的公共数据集一样，是在academic competition里发展出来的。这些competitions传统上是很多研究院和公司里的研究员打比赛的地方。ImageNet Large Scale Visual Recognition Challege（ILSVRC）从2010年开赛以来就获得了很广泛的关注。这个competition基于一些任务，这些任务每年都不一样，比如image classification（告诉图像所含的物体是什么类别的），object localization（给出图片中物体的位置），object detection（鉴别图片中的物体并给出它们的类别），scene classification（给整张图片分类），以及scene parsing（将图片分割为有语义的区域，比如牛，房子，奶酪，帽子等）。我们细说image classification，这个任务是以一张图片作为输入，并在1000个类别中给出5个类别，按置信度高低排序，来描述图像中的内容。

ILSVVRC的训练集包括12 000 000张图片，这些图片标记有唯一的名词（1000个类别），这个名词叫做图像的class。在这个任务里，我们将label和class混用，表达同一个意思。figure 1给出了ImageNet里一些图片。

![ImageNet]({{ '/assets/images/DLP-2-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. A small sample of ImageNet images.*

我们这个任务（image classification）的最终的效果是，给训练好的模型一张新的图片，它能给这张图片返回一个预测的label的列表（在这个例子里列表里包含五个类别），从而我们可以通过这个label list来了解图像的内容。如figure 2所示。

![Inference]({{ '/assets/images/DLP-2-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. The inference process.*

输入的图片将会先被处理为一个multidimensional array class torch.Tensor的实例。这是一张标有高度和宽度的RGB图片，所以这个tensor将会有三个维度：RGB三个通道，以及两个有指定大小的图片spatial dimensions（tensor将会在下一章Chapter 3里详说）。我们的模型将会以上述的处理后的图片作为输入，并给出每个class的分数。之后每个class被一一对应到class label上。输出被存在一个拥有1000个element的torch.Tensor里，每个element表达该class的分数。

在我们正式开始之前，我们先了解一下这个网络本身，看看它的结构，以及学习一下在模型使用前该如何处理数据。

#### 2.1.1 为image recognition任务获取一个预训练的模型

我们现在去找一个在ImageNet上预训练好的网络。[TorchVision](https://github.com/pytorch/vision)项目包括了一些computer vision领域表现最好的模型，比如AlexNet，ResNet，Inception V3等等。在Pytorch里同样很容易获取如ImageNet这样的数据集。之后我们将会介绍更多这样的PyTorch里已经包括或者容易获得的外部资源。现在，我们下载两个网络：AlexNet和ResNet。

预训练好的模型可以在torchvision.models里获得：

```python
# In [1]:
from torchvision import models

# In [2]:
dir(models)

# Out [2]:
['AlexNet',
 'DenseNet',
 'InceptionV3',
 'ResNet',
 'SqueezeNet',
 'VGG',
...
 'alexnet',
 'densenet',
 'densenet121',
...
 'resnet',
 'resnet101',
 'resnet152',
...
]
```

上面首字母大写的那些名字都是Python的class，每一个class都对应着一系列模型，这些模型的内部结构都有着细小的区别。而小写字母的名字是function，它们会返回大写字母对应的那些class里的实例，这样有时候会更方便。比如，resnet101会返回一个ResNet的实例，其有101层，resnet18则会返回一个有18层的ResNet的实例。我们现在将注意力放在AlexNet上。


#### 2.1.2 AlexNet

AlexNet赢了2012 ILSVRC比赛，比第二名高了很多，top-5 error rate是15.4%，第二名是26.2%（第二名不是基于deep Learning的）。这是computer vision发展史上值得记住的时刻：人们意识到了deep learning对于vision任务的巨大潜力。从此开启了对这个比赛持续性的基于deep learning的改进，最低能使得top-5达到3%。

以今天的衡量标准来看，AlexNet是个相当小的网络。但是这是对刚了解neural network以及想上手操作的人一个非常好的例子。

AlexNet的结构如figure 3所示。在图中，输入image从最左侧开始，经过了五个filters，每个给出一系列输出images（我们可以把这些filters理解成一系列加法乘法并最后加上一个activation函数的filter，以一系列图片为输入，一系列图片为输出）。在经过每个filter之后，images的大小被缩小了。最后一个filter输出的image被展平成4096个element的一维向量，并再加上几层全连接层来产生1000个输出概率，每个对应着一个class。

![Architecture]({{ '/assets/images/DLP-2-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. The AlexNet Architecture.*

为了运行AlexNet，我们可以构建AlexNet class的一个instance：

```python
# In [3]:
alexnet = models.AlexNet()
```

现在alexnet就是一个具有着上图中AlexNet结构的object。通过给alexnet一些准确sized的输入数据，我们可以运行forward pass，将这个输入数据通过alexnet。从实际操作来说，就是我们有input object，这样可以运行forward pass：output = alexnet(intput)。

但是如果我们这样做的话，是没有意义的。因为整个网络参数是没有被学习过的。我们需要训练这个网络或者从预训练的网络上加载学习好的参数。我们回到models模块。我们知道大写字母开头的对应着computer vision领域流行的网络结构，它们每个都是一个class。小写字母的那些是functions，输出的是某个class的instance，这个instance有给定好的网络层数，并且可以将预训练好的参数加载到这个instance里。注意到，用这些小写字母代表的函数没有什么特别的，他只是使得生成一个满足条件的instance变得更容易而已（要不然还需要挨个定义层数等等）。


#### 2.1.3 ResNet

用resnet101 function，我们可以实例化一个101层的convolutional neural network。在2015年residual networks发明之前，训练一个很深的网络是十分困难的。residual networks克服了训练很深的网络的困难，并在提出的那一年在多项比赛上打败了其它的网络结构。

我们来构建一个ResNet的实例。我们给resnet101 function传递一个argument，来告诉这个function去下载已经在ImageNet上预训练好的参数：

```python
# In [4]:
My_resnet = models.resnet101(pretrained=True)
```


#### 2.1.4 Ready, Set, Almost run

我们来看一下一个resnet101到底是什么样的。我们可以通过printing所返回的实例来看。结果是一个文字化的表述，描述了网络结构的细节。

```python
# In [5]:
My_resnet

# Out [5]
ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                    bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                       track_running_stats=True)
    (relu): ReLU(inplace)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1,
                        ceil_mode=False)
    (layer1): Sequential(
       (0): Bottleneck(
...
       )
    )
    (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
    (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```

上面所示的每一行都是一个模块。这个模块和Python里的module是不同的概念，这里每个模块代表着一种运算，是neural networks的组成部分。这些模块也成为neural networks的layers。

resnet101 function所返回的实例里，有很多重复的Bottleneck的模块（实际上有101个），每个都包含了convolutions以及其它的运算。resnet101是个典型的深度神经网络结构：由众多的层堆积而成，并以fully connected layer收尾，从而输出一个向量来代表每个种类的分数。

My_resnet变量可以被当作一个function来调用，以一张或多张图片作为输入并输出1000个类别的分数。在我们用它之前，我们还需要对输入图片做一些preprocess，从而使得这些图片有正确的size以及它们的pixel value大致落在一个合理的区间内。为了实现这样的preprocessing，torchvision module提供了transforms，可以允许我们很快定义包含基础preprocessing function的pipeline：

```python
# In [6]:
from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
```

在这种情况下，我们定义了一个preprocess function，这个function将输入图片scale到$$256 \times 256$$的大小，按中心区域crop一块$$224 \times 224$$的区域，transform这个图片成一个tensor（一个PyTorch multidimensional array：在这种情况下，是一个3D array，维度分别代表color，height和width），并沿着color维度normalize，从而使得RGB维度的tensor都具有指定的mean和standard devariations。在Chapter7的7.1.3里我们将会更加深入的介绍transforms。

我们现在可以找一张图片，preprocess它，再将它给到My_resnet，来看看ResNet会给出什么样的结果。我们可以用Python的Pillow module来上传一张本地文件系统里的照片，Pillow是个很好用的图片处理module：

```python
# In [7]:
from PIL import Image
img = Image.open("../data/p1ch2/bobby.jpg")
```

如果我们用的是Jupyter Notebook，那我们还可以inline的看一下照片是什么样的（在实际中，照片会显示出来，替代下面<PIL.JPegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at 0x1B1601360B8>的部分）：

```python
# In [8]:
img

# Out [8]:
<PIL.JPegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at 0x1B1601360B8>
```

或者我们还可以用show() method，这样会弹出一个带有viewer的窗口，就可以直接看到图片：

```python
>>> img.show()
```

![Bobby]({{ '/assets/images/DLP-2-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. The picture of Bobby, the output of img.show().*

然后，我们就可以把输入的图片通过我们的preprocessing pipeline函数：

```python
# In [9]:
img_t = preprocess(img)
```

我们还可以reshape，crop，normalize这个input tensor，在后两章会详细说。

```python
# In [10]:
import torch
batch_t = torch.unsqueeze(img_t)
```

#### 2.1.5 Run!

对输入的新数据运行一个已经训练好的模型，这个过程叫做inference。为了inference，我们需要将这个network的mode变成eval：

```python
# In [11]:
My_resnet.eval()

# Out [11]:
ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                    bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True,
                       track_running_stats=True)
    (relu): ReLU(inplace)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1,
                        ceil_mode=False)
    (layer1): Sequential(
       (0): Bottleneck(
...
       )
    )
    (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
    (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```

如果我们忘记将其mode改成eval，一些内部的模块，比如batch_normalization和dropout，将会给出没有意义的结果，因为它们内部的机制使得它们在训练和eval模式下运行的不一样。现在模型的mode已经是eval了，我们就可以开始inference了：

```python
# In [12]:
out = My_resnet(batch_t)
out

# Out [12]:
tensor([[ -3.4803, -1.6618, -2.4515, -3.2662, -3.2466, -1.3611,
          -2.0465, -2.5112, -1.3043, -2.8900, -1.6862, -1.3055,
...
          2.8674, -3.7442, 1.5085, -3.2500, -2.4894, -0.3354,
          0.1286, -1.1355, 3.3969, 4.4584]])
```

我们现在需要找到上面输出的1000维的向量里哪一个有着最高的score。这将会告诉我们这个模型在这张输入图片中看到了什么。

为了获得这个网络到底给这张图片标了什么标签，我们需要下载一个text file，这个text file列出了所有的可能的类别，而且这些类别的排列顺序和网络训练时输出的排列顺序是对应的。几乎所有的image recognition任务都有着和这个问题相似的输出结构。

我们下载一下含有1000个标签的ImageNet数据集的text file：

```python
# In [13]:
with open(../data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
```

现在我们需要找到out这个tensor里最大score所对应的index是什么。我们可以用PyTorch里的max function来做，这个function会输出一个tensor的最大值以及这个最大值的位置：

```python
# In [14]:
_, index = torch.max(out, 1)
```

我们现在可以用这个index来确定标签了。但是上述得到的index是一个PyTorch的tensor，一个one-element，one dimensional的tensor，而并不是一个Python的number（实际上应该是tensor($$\left[207\right])$$）。所以我们需要得到这个index的实际值，可以通过index$$\left[0\right]$$来得到。我们再通过torch.nn.functional.softmax来normalize输出到$$\left[0,1\right]$$的范围。softmax的结果有点像模型对于每个类别给出一种置信度。在我们的例子里，模型有96%的自信输入的图片是一只金毛：

```python
# In [15]:
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()

# Out [15]:
('golden retriever', 96.29334259033203)
```

既然模型输出了长度为1000的向量来对每个类别都给了分数，我们还可以找到分数排名第二高，第三高等等所代表的标签。我们可以用sort function来对原out进行排序，这个function会输出排序后的序列，并且会输出排序后序列里每个element在原序列里的位置：

```python
# In [16]:
_, indices = torch.sort(out, descending=True)
[(labels[idx], percentage[idx].item() for idx in indices[0][:5]]

# Out [16]:
[('golden retriever', 96.29334259033203),
('Labrador retriever', 2.80812406539917),
('cocker spaniel, English cocker spaniel, cocker', 0.28267428278923035),
('redbone', 0.2086310237646103),
('tennis ball', 0.11621569097042084)]
```

我们可以看到前四名都是狗，之后就变得奇怪了。tennis ball是第五个结果。这是一个显示neural networks和人类在vision领域的判断区别有多大的例子。


### 2.2 A pretrained model that fakes it until it makes it

假设现在我们是想要模仿知名画家的画的职业罪犯。我们不是画家所以并不会画画。如果我们只是自己在不断地练习，很难达到模仿画家绘画的水平。但是如果我们请一个对所要模仿的画家很了解的专家来评判我们画的画，对于不像的部分进行改进，那我们的绘画水平就会朝着该画家的绘画风格而进步。

而GAN所要做的事情，和上述故事有着很大的相似之处。我们可以制作假的数据，而其风格和我们所看到的数据很类似，不管是image还是video都可以。

#### 2.2.1 The GAN game
在deep learning圈子里，我们刚讨论的故事就是GAN game。这个game有两个networks，一个作为绘画者，而另一个则是评判专家，他们两相互比赛来超过对方，绘画者要画出让专家尽可能无法判断真伪的画，而专家要尽可能地判断正确画的真伪。GAN是generative adversarial network的简称，其中generative表示某些东西被创造了出来（在GAN里就是假画），adversarial表示两个networks在互相竞争。GAN是近些年deep learning领域最重要的原创性工作之一。

记住我们最终的目标是生成尽可能像真data的假example。当和真data混合在一起的时候，有经验的专家也无法分辨他们的真伪。

generator网络扮演绘画者的角色，任务是创造尽可能以假乱真的照片，输入是一个任意值。discriminator网络扮演评判专家的角色，需要判断给的画是出自于绘画者还是来自真的画的集合。这样的两个网络设计是非典型的，但是在GAN这个game里，它们将会给出很惊人的结果。

figure5粗略的展示了GAN的结构。generator的最终目标就是迷惑discriminator使得它无法分辨真实和假的图画。discriminator的最终目标就是能判断画的真假。在最开始，generator画的画很容易被判断出来是假的。随着训练过程的进行，从discriminator返回的信息被generator利用来精进绘画技巧。在训练结束后，generator可以画出以假乱真的画，这时discriminator已经无法分辨画的真假了。

![GAN]({{ '/assets/images/DLP-2-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. Concept of a GAN game.*

值得注意的是，discriminator和generator并不是真的在比赛。这两个网络分别基于对方的输出而进行训练，对方网络的输出驱动了它们各自网络参数的优化。

GAN game可以使得generators仅仅从噪声和conditional signal（比如某种attribute，对于脸来说，young，female，glass等）或者从另一个image来生成足以以假乱真的图像。

#### 2.2.2 CycleGAN

GAN的一个有趣的变种是CycleGAN。CycleGAN可以将一个domain内的图片变成另一个domain内的图片，而不需要我们在训练集中提供这两个domain图像的pairs。

在figure6里，我们展示了将马变成斑马，以及将斑马变成马的网络结构。这个网络里有两个分开的generators，并且都有着各自的discriminators.

![CYCLEGAN]({{ '/assets/images/DLP-2-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 6. A CycleGAN trained to the point that it can fool both discriminator networks.*

如figure6所示，第一个generator要学习如何生成满足一个target distribution（在这个例子里是斑马）的图片，而其出发点则是满足属于另一个distribution（在这个例子里是马）的图片，从而它对应的discriminator无法分辨这个图片是从一张马的图片生成的还是本身就是一个斑马的图片。同时，这也是名字里Cycle的由来，这个生成的假的斑马图片作为另一个generator的输入，这个generator要学会如何从输入的斑马图片生成马的图片，并且也有一个相应的discriminator来分辨真假。构建这样一个循环，能够使得GAN的训练更加稳定，这也是GAN在实际应用中很重要也很常见的一个问题（难以训练）。

有意思的是，我们并不需要输入马/斑马一对照片作为输入。从一系列没有联系的马和斑马照片，就足以让generators学会各自的生成，这个已经超过了单纯的supervised learning的范围。这个模型所蕴含的道理甚至还有更深：generator学习如何选择性的改变物体的appearance而并不需要告诉他物体是什么。并没有任何信息告诉generator鬃毛和腿该如何对应，以及它们分别是这个图里的哪部分，但generator仍然可以对应的将其转换。

#### 2.2.3 A network that turns horses into zebras

CycleGAN被一系列没有关联的从ImageNet里提取出来的马和斑马的图片数据集所训练。这个网络学会以一张或者多张马的图片作为输入并将它们全部转换为斑马的图片，并保证图片中除了马或斑马的部分保持不变（环境不变）。运行已经预训练好的CycleGAN可以让我们更进一步了解一个network——在这个网络里，一个generator——是如何被implemented的。

我们定义一个ResNetGenerator class。实例化这个class，使用default parameters就可以：

```python
# In [2]:
netG = ResNetGenerator()
```

netG model被创建了，但是它的参数仍然是随机的。我们可以加载已经预训练好的这个model的参数。model的参数被存在一个.pth的文件里，这个文件是一个包含着模型参数的pickle file。我们可以用model的load_state_dict method将它们加载到ResNetGenerator里：

```python
# In [3]:
model_path = '../data/p1ch2/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)
```

从而现在netG已经获得了预训练所得的所有的知识。注意到，此处加载预训练好的模型参数和2.1.3里给resnet101加载预训练参数是一样的，只不过torchvision.resnet101 function隐藏了加载参数的过程。

如同resnet101例子一样，我们将模型mode改成eval：

```python
# In [4]:
netG.eval()

# Out [4]:
ResNetGenerator(
    (model): Sequential(
...
    )
)
```

这个网络以一张图片为输入，识别图片里有几只马，之后分别将这些马所对应位置的像素进行更改，从而将原图片里的马变成斑马输出。

我们现在可以随意输入一张关于马的图片，来看看我们的generator会输出什么。首先我们需要PIL和torchvision模块：

```python
# In [5]:
from PIL import Image
from torchvision import transforms
```

之后我们做一些输入的transformations，从而使得输入能够匹配网络：

```python
# In [6]:
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])
```

打开一张马的照片：

```python
# In [7]:
img = Image.open("../data/p1ch2/horse.jpg")
img
```

![horse]({{ '/assets/images/DLP-2-7.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 7. A man riding on a horse.*

之后，我们将照片通过preprocess function得到恰当shape的变量：

```python
# In [8]:
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t)
batch_out = netG(batch_t)

# batch_out是generator的输出，我们可以将其转换回一张图片

out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)

out_img.save("../data/p1ch2/zebra.jpg")
out_img

# Out [8]:
<PIL.Image.Image image mode=RGB size=316x256 at 0x23B24634F98>
```

上述代码里输出的<PIL.Image.Image image mode=RGB size=316x256 at 0x23B24634F98>实际上是下图：

![zebra]({{ '/assets/images/DLP-2-8.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 8. A man riding on a zebra.*


### 2.3 A pretrained network that describes scenes

我们已经见过了识别图片的模型，生成新图片的模型，而这个则是融入了自然语言的computer vision的模型。我们将会介绍一个预训练好的image captioning model，NeuralTalk2 model。这个model的输入是一张图片，而输出则是对这张图片所含内容的英文描述，如figure9所示。这个模型被一个很大的数据集训练，这个数据集里的数据是图片和描述图片的文字pair所构成的。

![caption]({{ '/assets/images/DLP-2-9.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 9. Concept of a captioning model.*

这个model由两个互相连接的部分所组成。一个部分负责输出具有代表性的描述性的representations，作为另一部分的输入。另一部分是一个recurrent neural network，可以生成句子。这两个部分在image/captioning pair上共同被训练。

这个模型的第二部分叫recurrent是因为它以一个接一个的方式生成每一个output（individual word），每一次的input都包含了上一次的输出。这样就使得后面的word都和前面的word有关联，而这在生成句子的任务里正是我们所需要的。

#### 2.3.1 NeuralTalk2

[NeuralTalk2](https://github.com/deep-learning-with-pytorch/ImageCaptioning.pytorch)可以以如下的方式来运行，我们要先将一些图片放在data文件夹下，然后运行脚本：

```python
python eval.py --model ./data/FC/fc-model.pth --infos_path ./data/FC/fc-infos.pkl --image_folder ./data
```

试一下2.2里的马的图片，结果是 A person riding a horse on a beach。很准确。


### 2.4 Torch Hub

预训练好的模型在deep learning发展的早期便被放在了网上，但是直到PyTorch 1.0的发行，用户并没有一个统一的界面来获得这些预训练模型。

PyTorch 1.0介绍了PyTorch Hub，这个机制可以让作者以某种PyTorch可以理解的方式，通过Github来将已经预训练好的或者没有预训练的模型发布出来。这使得发布模型变得很方便。torchvision模块包含了一些模型，但只是很少一部分，而PyTorch Hub的发布，使得作者可以随意发布自己的模型供其他人参考使用。

通过Torch Hub机制发布模型，作者仅仅需要在该项目的Github repository的根目录下加一个hubconf.py文件。这个文件有着以下的结构：

```python

# 这个文件有着两个部分

# 第一部分是你的代码所需要依赖的modules
dependencies = ['torch', 'math']

# 第二部分是一个或多个函数，这些是用来给用户了解你的模型和代码用的。这些函数需要有根据输入的参数初始化模型的作用。

def some_entry_fn(*arg, **kwargs):
    model = build_some_model(*args, **kwargs)
    return model
    
def another_entry_fn(*args, **kwargs):
    model = build_another_model(*args, **kwargs)
    return model
```

我们查找感兴趣的预训练模型，我们现在就可以搜索Github repository看看哪个包含hubconf.py文件，只要包含了，我们就可以通过torch.hub模块来调用它。我们来看一个实际中的例子。为了方便起见，还是从torchvision看起，因为它提供了一个如何和Torch Hub互动的清晰的例子。

我们访问torchvision repository[所在的位置](https://github.com/pytorch/vision)发现它含有一个hubconf.py文件。这点check，说明这个项目可以用torch.hub module调用。我们要做的第一件事就是看看hubconf.py文件，找到repository的入口处。在Torchvision这个例子里，hubconf.py文件里有两个函数，resnet18和resnet50。我们已经知道这两个函数的作用：它们分别返回一个18层和一个50层的ResNet model。我们同样看到hubconf.py文件里的入口函数有一个pretrained keyword argument。如果是True，返回的model就是已经在ImageNet上预训练过的。

现在我们知道了repo，知道了入口函数，也知道了所需要了解的入口函数的argument，用torch.hub下载模型所需的全部知识我们就都有了，即使不clone那个repo也可以，PyTorch会帮我们解决：

```python
import torch
from torch import hub

resnet18_model = hub.load('pytorch/vision:master',
                          'resnet18',
                          pretrained=True)

# 'pytorch/vision:master'是Github repo的name和branch，'resnet18'是入口函数，entry-point function，的name，pretrained=True是keyword argument
```

上述代码会下载pytorch/visio repo的master branch的一个镜像，随着模型的参数，一起存到本地（系统默认存在home文件夹的 .torch/hub里），并运行resnet18 entry-point function，从而返回一个实例化的model。取决于具体的运行环境，Python可能会报错module缺失，比如PIL。Torch Hub不会主动下载缺失的module，但是它会给出具体缺失什么module的error。

我们现在就可以用返回的model，给出适当的argument，来运行一个forward pass整个模型了。Torch Hub的好处是所有按照上述机制发布的Github repo都可以被其他人所获取了，而不仅仅是torchvision里一小部分的model。

值得注意的是entry-point function需要返回model；但是严格来说，它们并不是一定要这样。比如在一个hubconf.py文简历，我们可以定义一个entry-point function的作用是将output probability转换为文字描述，而另一个的作用是transform输入。或者可以定义一个entry-point function输出model，另一个entry-point function来对这个model做进一步的处理。这种flexibility为将来PyTorch Hub的发展提供了无限可能。


### 2.5 Summary

* 一个预训练的网络是一个已经在某个数据集上被训练过的model。这样的网络在载入之后可以直接给出有用的结果。
* 通过学习如何使用一个预训练的网络，我们可以将一个预训练好的模型直接整合到项目里，而不需要对它有过多的了解或者对它进行训练。
* AlexNet和ResNet是两个十分著名的computer vision的model。
* GAN有两个部分：generator和discriminator。这两个部分相互促进来使得model生成足以以假乱真的data。
* CycleGAN的结构使得它可以在两个不同类的图片间互相转换。
* NeuralTalk2用了一种混合的模型架构，以图片为输入，生成描述该图片内容的文字。
* Torch Hub是一个从有满足条件的hubconf.py文件的任意Github项目载入model和model的参数的的标准的方法。


## Chapter3 It starts with a tensor

>这章包括的内容
>* 理解PyTorch里最基本的data structure：tensor
>* 在tensor上index和operate
>* tensor和Numpy的multidimensional array相互操作
>* 将tensor上的computation转移到GPU上以加速计算速度

在上一章里，我们看了一些各个领域的deep learning的pretrained model，并加以运行。它们都需要以某种格式的data做输入，比如image或者text，然后输出另外一种格式的data，比如labels，numbers，或者别的images、text。从这个角度来看，deep learning实际上就是将data从某种representation转换成另一种的一个system。这个转换是通过寻找所给的数据example里的共同点来学习到的。比如，这个system可能会注意到狗的一些普遍的特征以及金毛特殊的颜色，将这两个特征结合起来，系统能将输入的金毛图片转换为输出的代表金毛的label。学习后的系统能够广泛的识别相似的输入，并将它们归于同一个label。

上述的过程开始于将输入转换为floating-point numbers。figure1里的第一步，将输入的image pixels转换为numbers，这个内容将会在Chapter4里详细说。但在此之前，在这章里，我们将会介绍在PyTorch里如何用tensor来处理所有的floating-point numbers。


### 3.1 The world as floating-point numbers

既然floating-point numbers是一个network能够理解的处理信息的数据表达方式，我们就需要将各种各样的现实世界里的data encode成networks能够理解的数据格式（floating-point numbers），然后再将输出转换回我们人类能够更好理解的数据格式。

![floating-point number]({{ '/assets/images/DLP-3-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. A deep neural network learns how to transform an input representation to an output representation (Note: The numbers of neurons and outputs are not to scale.).*

一个deep learning network一般学习从一种数据格式到另一种数据格式的transformation，这表明在network中间每层之间的那些中间层数据可以被理解成一系列中间层的数据representations。对于image recognition来说，网络前面层的representations一般是一些edge detection或者某些特定的textures比如fur。而后面更深的层的representations就会表达一些更复杂的结构比如ears，noses等。

总的来说，这样的中间层representations是一系列的floating-point numbers，这些floating-point numbers表示了输入的内容和数据结构的特征，并对network如何描述从输入转换为输出这个过程有十分有用。这样中间层的representation是从输入的众多examples里学习到的。这些floating-point numbers（中间层representations）和对于它们的操作是现代AI领域最重要的内容，在这本书里我们会一直提及。

我们需要知道的是，这些中间层的representations（如figure1里的step2里的内容）是输入和前面层的neurons的weights相结合的产物。每个中间层representation对于每个输入都是特别的。

在学习如何将我们的数据转换为floating-point numbers之前，我们需要首先理解PyTorch如何处理以及存储各个层面的数据，包括输入，中间层的representations，和输出。这章将会详细讲解这方面的内容。

为此，PyTorch引入了一个基础的data structure：tensor。在Chapter2里，当inferece预训练模型的时候，我们已经遇到了tensor了。对于数学里的tensor而言，其和space，transformations等等相关，而数学里的tensor和PyTorch里的tensor并不是一个东西。在deep learning，或者PyTorch的领域，tensor指的是vector，matrix的任意维度的推广，我们可以在figure2里看到。tensor这个概念的另一个名字叫multidimensional array。一个tensor的维度和想要获取这个tensor里某个位置的scalar值所需要的index的长度是相同的（从figure2里也可看出）。

![tensor]({{ '/assets/images/DLP-3-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Tensors are the building blocks for representing data in PyTorch.*

PyTorch并不是支持multidimensional array唯一的库。Numpy是目前最流行的multidimensional array库，它已经成为了data science领域的通用语言。PyTorch可以和Numpy无缝衔接，互相交互使用。从而PyTorch和众多的Python库比如SciPy，Scikit-learn以及Pandas都有交互。

和Numpy arrays相比，PyTorch tensors有一些”超能力“，比如能在GPU上很快的计算，能部署在多个设备上，以及能够跟踪计算图。这些都是现在deep learning库所需要的特点。

我们将首先介绍PyTorch tensor，介绍tensor的基本内容以便之后使用。第一步是学习如何用PyTorch tensor库来操作tensor。这些操作包括如何将数据存在内存里，如何在任意大的tensor上运行一些特定的operation，如何和Numpy相交互，以及如何加速GPU计算。理解tensor的能力以及它的API是学会PyTorch最重要和基础的内容。在下一章，我们将会把这章学习到的知识应用于实践，学习如何将各种形式的输入数据转化为neural network能处理的tensor。


### 3.2 Tensors: Multidimensional arrays

我们已经知道tensor是PyTorch里最基础的data structure。一个tensor就是一个array：一种存有一系列数据的data structure，这些数据里的每一个都可以用一列index来获取。

#### 3.2.1 From Python lists to PyTorch tensors

我们来看看Python里的list是怎么被index的，从而可以对比tensor的index操作。

```python
# In [1]:
a = [1.0, 2.0, 1.0]

# 我们可以access这个list的第一个element：
a[0]
a[2] = 3.0
a

# Out [1]:
1.0
[1.0, 2.0, 3.0]
```

用Python list来存储更高维度的数据也是很常见的，比如存储一个2D直线上点的坐标。但是用更高效的tensor data structure，很多类型的数据，比如image，time series，甚至是sentences，都能被tensor表示出来。我们将会在这章里介绍一些tensor上的operations用来操作tensors，这些operation可以被很高层的语言（Python）调用。

#### 3.2.2 Constructing our first tensors

让我们来构建第一个PyTorch tensor来看看它到底是什么样的。

```python
# In [2]:
import torch        # 导入torch module
a = torch.ones(3)   # 创建一个一维的tensor，size是3，每个element都是1
a

# Out [2]:
tensor([1., 1., 1.])

# In [3]:
a[1]

# Out [3]:
tensor(1.)

# In [4]:
float(a[1])

# Out [4]:
1.0

# In [5]:
a[2] = 2.0
a

# Out [5]:
tensor([1., 1., 2.])
```

在导入torch module之后，我们可以调用一个function来构建一个一维的tensor，其size是3，所有的值都是1.0。我们可以用index来access它的element或者更改它。虽然表面上看这个PyTorch tensor和Python的list没什么区别，但背后的运行机制完全不同。

#### 3.2.3 The essence（本质，要素） of tensors

Python的list或者tuple也可以包含一系列的numbers，它们是一系列Python objects的合集，每个number都是一个object，都分配给了单独的内存，而这些内存可能完全是分散开的，如figure3左边所示。而PyTorch的tensors或者Numpy的arrays，是一些包含着unboxed C numeric type的连续的内存空间的views，也就是说tensor或者Numpy array内的numbers是用unboxed C numeric type表示的，而不是Python的objects，而且这些数据在内存中的地址是连续的，如figure3右侧所示。假设PyTorch tensor或Numpy array的每个element都是一个32-bit（4 bytes）的float，如果要存一个一维的包含着1 000 000个element的tensor，则需要精确的4 000 000 bytes，以及一些额外的空间用来存储metadata（比如dimensions或者numeric type）。

>用Python list来存放数据，每个element都是一个单独的python object，分配了单独的内存，并有一些其他的信息。而且list本身也有这些element的信息从而才能找到他们。boxed说明每个element都是本身的value外面还包裹了本身的信息或供list查找的信息。而PyTorch tensor或者Numpy array是直接把这些element放在内存的一整块连续的位置上，从而不需要包裹，只需要存下来element的值就行，所以是unboxed的。从而这样节省了空间，也加快了运行速度。

![memory]({{ '/assets/images/DLP-3-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. Python object (boxed) numeric values versus tensor (unboxed array) numeric values.*

假设我们有一列坐标数据用来表示一个几何物体，比如说一个三角形的三个顶点的坐标（4，1），（5，3），（2，1）。不用Python的list来表示这个数据，我们可以用一个一维的tensor来存储x坐标在偶数index上，存储y坐标在奇数index上：

```python
# In [6]:
points = torch.zeros(6)  # 利用zeros函数构建一个所需要的size的tensor
points[0] = 4.           # 之后在overwrite我们所需要的值到这些0上。
points[1] = 1.
points[2] = 5.
points[3] = 3.
points[4] = 2.
points[5] = 1.

# 还有一种简便的方法，直接将一个Python list作为torch.tensor的参数：

# In [7]
points = torch.tensor([4., 1., 5., 3., 2., 1.])
points

# Out [7]:
tensor([4., 1., 5., 3., 2., 1.])

# 要获取第一个顶点的坐标：

float(points[0]), float(points[1])   # 输出是(4.0, 1.0)
```

上述方法有效，但是还是复杂了，我们可以使得tensor的每一项都直接就是一个顶点的坐标：

```python
# In [8]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points

# Out [8]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])

# 我们直接把一个Python list传递给了torch.tensor函数。我们可以查看这个tensor的shape：

# In [9]:
points.shape

# Out [9]:
torch.Size([3,2])  # tensor的shape attribute直接返回了这个tensor的形状，但返回值仍然是个torch.Size object。

# 当我们知道了要构造的tensor的shape时，
# 我们也可以直接用torch的zeros或ones method直接实例化一个tensor
# 此时需要给ones或者zeros method提供一个包含shape信息的tuple：

# In [10]:
points = torch.zeros(3, 2)

# points = torch.zeros((3,2))是一个效果，所以说 a = (3,2), points = torch.zeros(a)也是一样的结果。

points

# Out [10]:
tensor([[0., 0.],
        [0., 0.],
        [0., 0.]])

# 我们现在access这个tensor里每个element就需要两个index了：

# In [11]:
points = torch.tensor([[4., 1.], [5., 3.], [2,. 1.]])
points[0, 1]

# Out [11]:
tensor(1.)

# 这个返回的是第1个点的第2个坐标，也就是y坐标。

# In [12]:
points[0]

# Out [12]:
tensor([4., 1.])

# 这个返回的就是第1个点。
```

上面这些index所返回的都仍然是tensor（除非用float函数强行转换掉，如float(points$$\left[0, 1\right])$$），这些tensor是内存里相同位置所存的数据的一个view。上面的points$$\left[0\right]$$是一个新的1D tensor，size是2，参考的是points这个tensor第一行的值。新的这个tensor并不代表新的内存被开辟，然后值被复制后放在新的内存，再将这些值打包返回给新的这个tensor。因为这样做非常的效率低。我们将会在这章的3.7的tensor的view再来详细介绍发生了什么。


### 3.3 Indexing tensors

如果我们想获得一个tensor，只包含之前的tensor的除去第一个point以外的所有的points，可以使用range index notation，这个在Python的list里也经常被使用。

```python
# In [1]:
some_list = list(range(6))
some_list[:]     # 返回这个list所有的值
some_list[1:4]   # 返回这个list从第2个值开始到第5个值结束，包括第2个但不包括第5个
some_list[1:]    # 返回这个list从第2个值开始到最后一个值的所有值，包括第2个
some_list[:4]    # 返回这个list从第1个值开始到第5个值的所有值，不包括第5个
some_list[:-1]   # 返回这个list从第1个值开始到最后一个值，不包括最后一个
some_list[1:4:2] # 返回这个list从第2个值开始到第5个值每隔两个取一个值，不包括第5个

# 上面是Python里的range index的用法，而PyTorch的tensor有着一样的用法。

points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points[1:]       # 返回这个tensor从第2个值到最后一个值，tensor([[5., 3.], [2., 1.]])
points[1:, :]    # 返回这个tensor维度1里第2个值到最后一个，维度2里所有的值，tensor([[5., 3.], [2., 1.]])
# 前一行代码没有提到维度2，所以默认取所有的，这两行代表的是一个意思，结果当然也是一样的。

points[1:, 0]    # 返回这个tensor维度1里第2个值到最后一个，维度2里第一个值，tensor([5., 2.])
points[None]     # 给这个tensor加上一个维度为1的新维度，这个新加的维度是在最外层的，就像unsqueeze function的效果一样，tensor([[[4., 1.], [5., 3.], [2., 1.]]])
```

除了range index以外，PyTorch还给tensor提供了另外的更强力的index方法，叫advanced indexing，将在下一章说。


### 3.4 Named tensors

tensors的各个dimensions（或axes）通常是某些东西的index（索引），比如image的pixel location，或者color channels。这个表明当我们想index一个tensor的时候，我们需要记住每个dimension代表的含义，并有效的按顺序的进行索引。但因为数据在作为tensor输入之后，中间层会将这个tensor经过很多次转换，所以跟踪每个dimension的含义是很容易出错的。

举一个具体的例子来说明上面所说的问题。想象我们有一个3D的tensor，比如2.1.4里的img_t，这是一个RGB的三通道的image（在下面的例子里我们就随机生成一个维度相同的tensor）， 我们想将它转为gray-scale（灰度图）。而生成灰度图所需要的RGB三通道的权重，就选一个很常见的生成亮度值的：

```python
# In [1]:
img_t = torch.rand(3, 5, 5)  # shape [channels, rows, columns]，这就是随机生成的RGB三通道的输入图片
weights = torch.tensor([0.2126, 0.7152, 0.0722])  # 用于RGB三通道的权重
```

我们希望我们的代码能更具有通用性，比如，从表示为标记着height和width的2D tensor的grayscale image变成RGB三通道的3D tensor，或者从一张输入image到一个batch的images。在2.1.4里，我们用torch.unsqueeze function给输入的3D RGB image tensor加了一个维度。类似的，我们也给输入的3D RGB image tensor加上batch这个新的维度，在我们的例子里，batch=2：

```python
# In [2]:
batch_t = torch.rand(2, 3, 5, 5)  # shape [batch, channels, rows, columns]，这就是随机生成的RGB三通道的输入图片，一共有两张
```

有时候我们加了batch这个维度，从而输入的tensor是4维的，那么RGB channels这个维度就在dimension 1，而有时候我们没加batch这个维度，从而输入的tensor是3维的，那么RGB channel这个维度就在dimension 0。但不管怎样，我们从末尾开始数，RGB channel这个维度永远在dimension -3。我们不用weights tensor给RGB三通道加权平均，只是单纯的取个平均数：

```python
# In [3]:
image_gray_naive = img_t.mean(-3)     # 将3D tensor img_t按照RGB channel这个维度取算术平均，得到一个2D tensor灰度图
batch_gray_naive = batch_t.mean(-3)   # 将4D tensor batch_t按照RGB channel这个维度取算术平均，得到一个3D tensor灰度图
img_gray_naive.shape, batch_gray_naive.shape

# Out [3]:
(torch.Size([5, 5]), torch.Size([2, 5, 5]))
```

现在变得更复杂一点，我们现在用weights tensor来给RGB三通道加权平均，而不仅仅像上面的例子那样直接取算术平均数。PyTorch允许我们将有相同shape的两个tensor进行multiply操作，当某个乘数的某一个dimension的size是1时，仍然是可以的。

>这就是为什么在2.1.4里要将img_t加上一个维度变成batch_t，尽管这个维度的size是1，但是ResNet生成的实例model默认输入的tensor的dimension是4，分别是(batch, channels, height, width)。

当PyTorch里做运算的两个operands的维度不一样的时候，PyTorch和Numpy一样提供了处理这种情况的一个简便的机制：broadcasting。意思是，将两个tensor的维度按右对齐对准，只要它们满足在对应的维度上的shape是相同的，或者某一个tensor这个维度上的shape是1，那么即使它们的维度长度不同，短的那个左侧将会被补上1，相当于将tensor的内容复制几份来扩充维度。

在我们这个例子里，我们可以用一个unsqueezed_weights这样一个tensor，它的维度是(3, 1, 1)，用它来乘以batch_t这个维度是(2, 3, 5, 5)的tensor，就可以得到结果为(2, 3, 5, 5)的tensor，其RGB channel的维度被乘以了weights相应的值，并且这个操作在batch的每个image和image的每个pixel上都做了操作：

```python
# In [4]:
unsqueezed_weights = weights.unsqueeze(-1).unsqueeze(-1)
img_weights = img_t * unsqueezed_weights
batch_weights = batch_t * unsqueezed_weights
img_gray_weighted = img_weights.sum(-3)
batch_gray_weighted = batch_weights.sum(-3)
batch_weights.shape, batch_t.shape, unsqueezed_weights.shape, batch_gray_weighted.shape

# Out [4]:
(torch.Size([2, 3, 5, 5]), torch.Size([2, 3, 5, 5]), torch.Size([3, 1, 1]), torch.Size([2, 5, 5]))
```

einsum是一种处理index的function，在PyTorch和Numpy里都有用，此处不再赘述。

通过上面的内容可以看到，这样的代码很容易出错，因为往往人们就忘了每个tensor每个dimension都是代表了什么意思。所以给tensor的dimension加上名字就变得合理了。

PyTorch1.3加了named tensor的特征。Tensor factory functions比如torch.tensor或torch.rand都可以接收一个叫做names的argument，这个argument必须是一个string sequence。

```python
# In [5]:
weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['Channels'])
weights_named

# Out [5]:
tensor([0.2126, 0.7152, 0.0722], names=('Channels',))
```

当我们已经有了一个tensor之后，我们还想给它加上names（但是不改变已经存在的那些），我们就可以调用这个tensor的一个method，refine_names。ellipsis省略号（...）可以允许我们跳过任意的dimensions。

```python
# In [6]:
img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
batch_named = batch_t.refine_names(..., 'channels', 'rows', 'columns')
print("img named:", img_named.shape, img_named.names)
print("batch named:", batch_named.shape, batch_named.names)

# Out [6]:
img named: torch.Size([3, 5, 5]), ('channels', 'rows', 'columns')
batch named: torch.Size([2, 3, 5, 5]), (None, 'channels', 'rows', 'columns')
```

利用rename这个sibling method，我们还可以覆盖或者丢弃原有的names（丢弃的话就传入None）：

```python
# In [7]:
img_new_named = img_named.rename(..., 'height', 'width')
batch_new_named = batch_named.rename(..., None)
image_new_named.names, batch_new_named.names

# Out [7]:
(('channels', 'height', 'width'), (None, 'channels' , 'rows', None))
```

>注意，不管是rename还是refine_names，内部只能最多有一个省略号。

当对两个tensor做operation的时候，PyTorch除了会进行dimension的验证（验证它们对应位置的dimension的shape是否一样或某个为1，如果dimension长度不同是否能够broadcast），它还会对names也进行检查。目前的PyTorch还不会根据names自动对齐dimensions，我们还需要在code里明确指出。align_as method会返回一个tensor，这个tensor会补全所缺少的dimensions而且将已经存在的dimensions按正确的顺序排列，而补上的缺少的dimensions都是shape=1的。

```python
# In [8]:
weights_aligned = weights_named.align_as(img_named)
weights_aligned.shape, weights_aligned.name

# Out [8]:
(torch.Size([3, 1, 1]), ('channels', 'rows', 'columns'))
```

对于那些接收dimension作为argument的function，比如sum，同样可以接受这个dimension的names作为argument：

```python
# In [9]:
gray_named = (img_named * weights_aligned).sum('channels')
batch_gray_named = (batch_named * weights_aligned).sum('channels')
gray_named.shape, gray_named.names, batch_gray_named.shape, batch_gray_named.names

# Out [9]:
(torch.Size([5, 5]), ('rows', 'columns'), torch.Size([2, 5, 5]), (None, 'rows', 'columns'))
```

我们可以注意到，在上面的例子里，weights_aligned的dimension是(3,1,1)，而batch_named的dimension是(2,3,5,5)，但是它们符合broadcasting的规则，且对应dimension的名字是一样的，从而可以运算。我们还可以注意到，结果的tensor是(2,5,5)，'channels'维度被sum函数消去，而第一个维度的名字和shape都继承了batch_named的，因为weight_aligned根本没有这个维度。

如果operation的两个tensor对应位置的names不一样，会直接报错：

```python
# In [10]:
gray_named = (img_named[..., :3] * weight_named).sum('channels')

# Out [10]:
RuntimeError: Error when
 attempting to broadcast dims ['channels', 'rows',
 'columns'] and dims ['channels']: dim 'columns' and dim 'channels'
 are at the same position from the right but do not match.

如果我们希望将已经命名的tensors还原到没有命名的时候，直接使用rename function将它们的名字改为None：

```python
# In [11]:
gray_plain = gray_named.rename(None)
gray_plain.shape, gray_plain.names

# Out [11]:
(torch.Size([5, 5]), (None, None))
```

named tensors仍然处于试用期，要谨慎使用。


### Tensor element types

到目前为止，我们已经讲了tensor是怎么运行的基础知识，但还没有讲什么类型的numeric types我们可以储存在tensor里。正如我们在3.2里所说的，用standard Python numeric types存放在tensor里并不是最优的选择，有以下几个原因：
* Numbers in Python are objects。因为一个floating-point number可能只需要32-bits来存放，Python会将它转换为一个有很多功能的复杂完整的Python object。这样的operation，叫做boxing，在存储小量数据时没有问题，但在数据量十分大的时候效率就变得十分低下了。
* Lists in Python are meant for sequential collections of objects。没有一个高效的算法能计算两个vectors之间的内积，或者计算一个vector内部所有element的和。同样的，Python的lists也没有一个高效的方法来优化如何在内存中存放list，因为Python的list内部存储的是list各个element的位置的指针，而各个element其实也都是Python object，并不一定非要是numbers。而且Python的list是一维的，尽管我们可以创建list of list，但这个效率就更低了。
* The Python interpreter is slow compared to optimized, compiled code。在大量的数据上运行数学计算，利用已经编译好的，底层的C代码要比Python快得多。

因为上述这些原因，data science libraries依赖Numpy或者是PyTorch tensors来提供numerical data structures以及它们上的operations的效率高的low-level的实现，并将它们包装在一个high-level的API里。为了使这个操作能够实现，一个tensor里的objects必须是同一种类型的numbers，而且PyTorch必须明确表示这个number的numeric type。

#### 3.5.1 Specifying the numeric type with dtype

tensor constructors（就是返回一个tensor的functions，比如torch.tensor，torch.ones，torch.zeros等）用dtype argument来指明要构建的tensor里的numbers的numerical data type（简写为dtype，就是从这来的）。dtype argument指明这个tensor内的numbers是intergers还是floating numbers，以及每个number占据多少个bytes，以及每个number是signed还是unsigned。PyTorch的tensor constructor的dtype argument故意设计的和标准的Numpy构建ndarray时的dtype叫同一个名字。

下面是dtype argument所有可能的取值：
* torch.float32 或者 torch.float：32-bit的floating-point
* torch.float64 或者 torch.double：64-bit的双精度的floating-point
* torch.float16 或者 torch.half：16-bit的半精度的floating-point
* torch.int8：有符号的8-bit整数
* torch.uint8：无符号的16-bit整数
* torch.int16 或者 torch.short：有符号的16-bit整数
* torch.int32 或者 torch.int：有符号的32-bit整数
* torch.int64 或者 torch.long：有符号的64-bit整数
* torch.bool：布尔值

默认的tensor的data type是32-bit的floating-point

>注意，floating-point都是带符号的。


#### 3.5.2 A dtype for every occasion

我们在后续的章节可以看到，在neural networks里发生的绝大多数运算用的都是32-bit floating-point的精度。更高的精度，64-bit floating-point不会给精度带来多少提高，但是会大大增加计算量。而更小的精度，16-bit floating-point在现在流行的CPU上不支持，在GPU上支持。用16-bit floating-point来计算可以减小计算量，但它对精度会带来多大的影响也是不可知的。

tensors同样可以用作其它tensors的index，那在这种情况下，PyTorch要求作为index的tensors的dtype是64-bit interger。构建一个tensor，利用integers作为初始值，那么PyTorch将会将dtype默认置为64-int integer，比如torch.tensor($$\left[2, 2\right]$$)。所以说，我们绝大多数时间都是处理dtype为int64或者float32的tensors。

最后，有tensors参与的谓词，比如points > 1.0，会产生bool tensor，来表示tensor的每个element是否满足条件。

以上就是PyTorch里tensor的dtype的一个概括。


#### 3.5.3 Managing a tensor's dtype attribute

为了给一个tensor分配正确的numeric type，我们需要在tensor constructor中给dtype特别赋值。

```python
# In [1]:
double_points = torch.ones(10, 2, dtype=torch.double)
short_points = torch.tensor([[1, 2], [3, 4]], dtype=torch.short)
```

我们直接通过访问一个tensor的dtype attribute来查看它：

```python
# In [2]:
short_points.dtype

# Out [2]:
torch.int16

我们还可以用相应的casting method将一个tensor constructor的输出投射到一个正确的numeric type：

```python
$ In [3]:
double_points = torch.zeros(10, 2).double()
short_points = torch.ones(10, 2).short()

# 或者使用更简单的to method：
double_points = torch.zeros(10, 2).to(torch.double)
short_points = torch.zeros(10, 2).to(dtype=torch.short)
```

在背后的机理里，to method会检查这个转换是否是有必要的（也就是说是不是本来就已经是该data type了），如果有必要再执行。那些dtype-named casting methods比如float是to method的简单表达方式，但是to method还可以有更多的argument，我们在3.9里会进一步讨论。

当运算的两个tensors有着不一样的data type时，较”小“的data type将会自动被转换为较”大“的那种。比如，如果我们希望计算只涉及32-bit的数，那所有的输入的tensor的dtype至多只能是32-bit的。

```python
# In [1]:
points_64 = torch.rand(5, dtype=torch.double)   # torch.rand用来生成0到1之间的随机数，5表示生成的维度是一维，shape=5
points_short = point_64.to(torch.short)         # 转换到torch.int16的情况下，每个值就变成0了
points_64 * points_short                        # integer和floating-point也可以相乘

# Out [1]:
tensor([0., 0., 0., 0., 0.], dtype=torch.float64)
```


### 3.6 The tensor API

现在我们已经知道了PyTorch的tensor是什么，以及它们背后的运行机制是什么样的。现在有必要来了解一下PyTorch提供的tensor operations。

首先，绝大部分一个tensor的operations或者两个tensors之间的operations都在torch module里，而且它们都可以被当作tensor的method来调用。

比如：transpose function可以从torch module中被调用：

```python
# In [1]:
a = torch.ones(3, 2)
a_t = torch.transpose(a, 0, 1)
a.shape, a_t.shape

# Out [1]:
(torch.Size([3, 2]), torch.Size([2, 3]))

# 而transpose function也可以被当成a的一个method被调用：

# In [2]:
a_t = a.transpose(0, 1)
a.shape, a_t.shape

# Out [2]:
(torch.Size([3, 2]), torch.Size([2, 3]))

# 这两种使用方法完全没有区别。
```

所有的tensor operations都可以在这个[online doc](https://pytorch.org/docs)上找到。它组织的很合理，将tensor的operations分为了以下几类：

* Creation ops：用来construct一个tensor的function，比如ones，from_numpy等
* Indexing，slicing，joining，mutating ops：用来改变一个tensor的shape，stride，或者内容的functions，比如transpose等。
* Math ops：通过计算来改变tensor的内容的funtions：
    * Pointwise ops：通过对一个tensor里每个element都作用该function来得到一个新的tensor，比如abs，cos等
    * Reduction ops：通过循环一个tensor里所有的element来得到一个总体的值，比如mean，std，norm等
    * Comparison ops：对tensors做numerical predicates，比如equal，max等
    * Spectral ops：在频域区间内转换和操作的函数，比如stft，hamming_window等
    * Other operations：一些tensor的特殊的function，比如cross等
    * BLAS和LAPACK ops：对scalar，vector-vector，matrix-vector，matrix-matrix的operations适用的BLAS（Basic Linear Algebra Subprogramming） functions。
* Random Sampling：从概率分布里随机取数的functions，比如randn，normal等
* Serialization：加载和存储tensor的functions，比如laod和save
* Parallelism：控制parallel CPU多线程运行的线程数量的functions，比如set_num_threads。


### 3.7 Tensors: Scenic views of storage

现在是时候看看PyTorhc里tensor的实现机理了。tensors里所储存的values被torch.Storage的instance分配给一块连续的内存空间。一个storage就是一个一维的Numerical data的array，也就是一个包含某个给定数据类型（比如float或者int64等）的连续内存空间。一个PyTorch tensor实例是这样一个torch.Storage实例的一个view，并且tensor加上了能够index等的功能。

多个tensors可以index同一个storage，即使形式上它们是不一样的，如figure4所示。实际上，在3.2里，point$$\left[0\right]$$会返回另一个tensor，而这个tensor index的storage和points tensor所指向的storage是同一个（只不过point$$\left[0\right]$$只是指向了一部分数据）。背后的内存实际上只被分配过一次，所以当创建这些新的tensors时，如果还是这个storage的或者其一部分的view，那么就会很快，因为并没有实际上在内存中创建了新的数据，这些都是被tensor.Storage实例来妥善解决的。

![Tensor storage]({{ '/assets/images/DLP-3-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. torch.Tensors are views of a torch.Storage instance.*

#### 3.7.1 Indexing into storage

我们用之前的2D points例子来看看如何在storage里index。一个给定的tensor的storage可以用.storage性质来获取：

```python
# In [1]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points.storage()

# Out [1]:
4.0
1.0
5.0
3.0
2.0
1.0
[torch.FloatStorage of size 6]
```

尽管tensor本身有三行两列，是个2维的tensor，但是其后背的内存是一个size为6的array。所以说，tensor知道如何将一对index对应到storage里具体的value上。

我们可以手动index torch.Storage里的内容：

```python
# In [2]:
points_storage = points.storage()
points_storage[0]

# Out [2]:
4.0

# In [3]:
points.storage()[1]

# Out [3]:
1.0
```

我们并不能用一对index来index一个2维tensor背后的storage。一个storage的布局永远都是一维的，不管view它的tensor是什么维的。

现在，改变一个tensor背后storage的值就会改变这个tensor的值是显而易见的。

```python
# In [4]:
points_storage = points.storage()
points_storage[0] = 2.0
points

# Out [4]:
tensor([[2., 1.],
        [5. ,3.],
        [2., 1.]])
```

#### 3.7.2 Modifying stored values: In-place operations

之前介绍了很多tensor的operations既可以用torch.tensor的methods调用，也可以用以tensor为输入的function来调用，但有一小部分一小部分这样的operations仅仅可以作为torch.tensor的method被使用。它们通过在名字后面加个下划线来分辨，比如zero_，表明这个method是通过改变输入的in place的作用而不是创建饿了一个新的tensor作为输出。比如，zero_ method会将输入tensor的所有值都变成0。所有的名字后面没有下划线的method，都会保持输入的tensor不变，而输出一个新的tensor：

```python
# In [1]:
a = torch.ones(3, 2)
a.zero_()
a

# Out [1]:
tensor([[0., 0.]
        [0., 0.]
        [0., 0.]])
```


### Tensor metadata: Size, offset, and stride

为了能够index一个tensor的storage，除存储的数据以外tensor需要有一些格外的信息，叫做metadata，明确的定义size，offset和stride。figure5表现了这些metadata之间如何互相作用的。size（在Numpy中叫做shape）是一个tuple，表示这个tensor每个维度有多少个element，也就是每个维度的size。offset是存储这个tensor的storage第一个element所在的index（tensor内的内容作为一个一维array存储在storage里，offset指的是这个array第一个element的index）。stride指的是storage这个一维的array每一个element之间所跳过的element的个数，因为每个tensor的数据类型是知道的，所以根据element的个数也能算出具体要跳过多少个bytes，而stride仍然是个长度和tensor维度相同的tuple，它表示在tensor的每个维度下，保持别的维度值不变，从这个维度的一个element到这个维度的下一个element要跳过的element的个数。

![Tensor storage metadata]({{ '/assets/images/DLP-3-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. Relationship between a tensor's offset, size, and stride. Here the tensor is a view of a larger storage, like one that might have been allocated when creating a larger tensor.*

#### 3.8.1 Views of another tensor's storage

```python
# In [1]:
points = torch.tensor([[4. ,1.], [5., 3.], [2., 1.]])
second_point = points[1]
second_point.storage_offset()

# Out [1]:
2

# In [2]:
second_point.size()

# Out [2]:
torch.Size([2])
```

所获得的新的tensor，second_point在storage里的offset是2（因为跳过了points tensor里的2个elements），其size用tensor的size() method所获得，返回的是一个torch.Size类的实例，内部参数是长度为1的list，因为second_point是一维的，具体数值表示这个维度的大小。实际上同样的信息也被存储在tensor的shape property里：

```python
# In [3]:
second_point.shape

# Out [3]:
torch.Size([2])
```

stride是一个tuple，维度和tensor的维度一样，tuple内每个值表示这个值所在位置表示的维度的index增加1在storage里所需要跳过的element的个数。比如points的stride是(2,1):

```python
# In [4]:
points.stride()

# Out [4]:
(2, 1)
```

在一个2维的tensor里获取一个element，index是i,j，实际上是在storage里通过storage_offset + stride$$\left[0\right]$$ * i + stride$$\left[1\right]$$ * j位置的element获得。offset一般都是0，但如果这个tensor是一个更大的tensor的storage的一部分的view，那么offset就会是一个正整数。

torch.Tensor和tensor.Storage之间的这种关系让一些operations变得计算更容易，比如transposing一个tensor或者获取一个子tensor，因为这些operations都不必再在memory里重新分配内存了。取而代之的是，这些operations返回的是一个拥有新offset，size和stride的新的tensor，但并没有新的storage。

在上面的例子里我们已经在points这个tensor里获取了子tensor，second_point，发现second_point的offset增加了。我们再来看看second_point的size和stride是怎样的：

```python
# In [5]:
second_point = points[1]
second_point.size()

# Out [5]:
torch.Size([2])

# In [6]:
second_point.storage_offset

# Out [6]:
2

# In [7]:
second_point.stride()

# Out [7]:
(1, )
```

second_point是points tensor的子tensor，相对于points tensor少了一个维度。second_point和points仍然指向的是同一个storage。这意味着改变second_point这个tensor同样会改变points tensor的值：

```python
# In [8]:
second_point[0] = 10.0
points

# Out [8]:
tensor([[4., 1.],
        [10., 3.],
        [2., 1.]])
```

如果我们希望不要改变原tensor的值（实际上所有的view都会改变它们所指的storage的值），我们clone那个子tensor到一个新的tensor：

```python
# In [1]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
second_point = points[1].clone()
second_point[0] = 10.0
points

# Out [1]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])
```


#### 3.8.2 Transposing without copying

points tensor是一个2维的tensor，每一个row代表一个point的XY坐标，一共三个点，所以shape是(3, 2)。而我们现在希望将其transpose，从而shape变成(2, 3)，每一列代表一个point的XY坐标。我们趁机介绍一个t function，是transpose的简称，专门用来转秩2维tensor：

```python
# In [1]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points

# Out [1]:
tensor([[4., 1.],
        [5., 3.],
        [2., 1.]])

# In [2]:
points_t = points.t()
points_t

# Out [2]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

# 我们可以来验证一下这两个tensor是不是指向同一个storage：

# In [3]:
id(points.storage()) == id(points_t.storage())

# Out [3]:
True

# 实际上这两个tensor，points和points_t仅仅在size和stride上有所不同：

# In [4]:
points.stride()
points_t.stride()

# Out [4]:
((2, 1), (1, 2))
```

所以说在points tensor里的第一个index增加1，比如说从points$$\left[0, 0\right]$$到points$$\left[1, 0\right]$$会在storage里跳过两个element，而第二个index增加1，比如说从points$$\left[0, 0\right]$$到points$$\left[0, 1\right]$$会在storage里跳过一个element。对于二维tensor来说，storage按照每行来存储值，而对于多维tensor来说，storage按照括号从内到外存储。

figure6表示了transpose一个2维tensor的过程。tranpose操作的正规定义为：并没有新的storage被创建，transpose是通过建立一个新的Tensor实例，这个实例的stride ordering与原tensor不同，而获得的。


#### 3.8.3 Transposing in higher dimensions

在PyTorch里，tranpose并不仅限于matrices。我们可以沿着任意两个dimension来transpose一个multidimensional array。

```python
# In [1]:
some_t = torch.ones(3, 4, 5)
transpose_t = some_t.transpose(0, 2)
some_t.shape

# Out [1]:
torch.Size([3, 4, 5])

# In [2]:
transpose_t.shape

# Out [2]:
torch.Size([5, 4, 3])

# In [3]:
some_t.stride()

# Out [3]:
(20, 5, 1)

# In [4]:
transpose_t.stride()

# Out [4]:
(1, 5, 20)
```

一个tensor，如果它所指向的storage所存储的数据是按照这个tensor内部括号从内到外存储的，比如之前的points，some_t等，那么称这个tensor是contiguous的，因为获取它的值只要按照括号从内到外的顺序，就不需要在storage里跳过任何element。经过transpose的tensor不是contiguous的。


#### 3.8.4 Contiguous tensors

某些PyTorch里的tensor operations仅仅作用于contiguous tensors，比如view，我们在下个chapter会遇到。tensor的contiguous method可以将一个non-contigous的tensor变成contigous的，而本来就是contiguous的tensor不会受到影响。

points是contiguous的，而points_t不是：

```python
# In [1]:
points.is_contiguous()

# Out [1]:
True

# In [2]:
points_t.is_contiguous()

# Out [2]:
False
```

我们可以通过contigous method来从non-contigous tensor获取一个新的contiguous tensor。新的tensor的内容是一样的，但是stride和storage都会变：

```python
# In [1]:
points = torch.tensor([[4., 1.], [5., 3.], [2., 1.]])
points_t = points.t()
points_t

# Out [2]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

# In [3]:
points_t.storage()

# Out [3]:
4.0
1.0
5.0
3.0
2.0
1.0
[torch.FloatStorage of size 6]

# In [4]:
points_t.stride()

# Out [4]:
(1, 2)

# In [5]:
points_t_cont = points_t.contigous()
points_t_cont

# In [6]:
tensor([[4., 5., 2.],
        [1., 3., 1.]])

# In [7]:
points_t_cont.stride()

# Out [7]:
(3, 1)

# In [8]:
points_t_cont.storage()

# Out [8]:
4.0
5.0
2.0
1.0
3.0
1.0
[torch.FloatStorage of size 6]
```

我们注意到，在上面的代码里，points_t.contiguous()重新创建了一个新的tensor，points_t_cont，使用contiguous methos使得points_t_cont变成了contiguous的，这个时候给points_t_cont tensor实际上分配了一个新的内存地址，而此时的storage就按照points_t_cont是个contiguous tensor的方式来存储，而且stride也变成了contiguous的。


### 3.9 Moving tensors to the GPU

到目前为止，当我们讨论storage，我们都讨论的是CPU上的memory。PyTorch tensor可以被存储在另一种processor上：graphics processing unit(GPU)。每个PyTorch tensor都可以被转移到某个GPU上来进行大规模快速并行计算。每个tensor的operation都会被PyTorch的GPU routines来执行。

>PyTorch Support for various GPUs
>在2019中旬之前的PyTorch版本只支持有CUDA的GPU。PyTorch也可以在AMD的ROCm上运行，但是需要用户自己编译。对Google的tensor processing units(TPUs)的支持仍在建设中，而别的类型的GPU，比如OpenCL,目前还不支持。

#### 3.9.1 Managing a tensor's device attribute

一个PyTorch的tensor，除了有dtype这样一个argument以外，还有一个device argument，指明这个tensor的数据放在computer的哪个设备上。我们通过指明这个argument来构造一个在GPU上的tensor：

```python
# In [1]:
points_gpu = torch.tensor([[4., 1.], [5., 3.], [2., 1.]], device = 'cuda')

# 我们也可以用to method将一个创建在CPU上的tensor转移到GPU上

# In [2]:
points_gpu = points.to(device = 'cuda')
```

通过上述操作，可以创建一个新的tensor，和原来的tensor有着一样numeracal data，但是存储在GPU的RAM里而不是默认系统的RAM里。现在tensor已经存在GPU里了，我们来看看它到底是如何加速计算的。在绝大多数情况下，CPU和GPU上的tensors使用相同的API，使得代码移植和编写变得简单。

如果设备上有多于一片GPU，我们可以直接指定用哪个GPU（从0开始）：

```python
# In [3]:
points_gpu = points.to(device='cuda:0')
```

现在这个tensor上的operations都会在GPU上进行了：

```python
# In [4]:

# 做一个操作，使得tensor里每个值乘以2

# 这个multiplication在CPU上计算
points = 2 * points

# 这个multiplication在GPU上计算
points_gpu = 2 * points.to(device='cuda')
```

上述计算结果points_gpu并不会被传回到CPU上，上面的代码过程如下：
* points tensor被复制到GPU上
* 一个新的GPU上的tensor被创建出来，用来存储multiplication的结果
* 这个GPU上的tensor的一个handle被返回

所以，如果我们给points_gpu加上一个常数，这个加法仍然在GPU上计算，而且并不会有数据流向CPU（除非我们要print它们或者index access它们）。

```python
# In [5]:
# 在GPU上进行
points_gpu = points_gpu + 4
```

如果我们想要将tensor转移回CPU，我们需要给to method提供一个cpu argument：

```python
# In [6]:
points_cpu = points_gpu.to(device='cpu')
```

我们可以直接使用cpu和cuda methods而不用to method来实现同样的操作：

```python
# In [7]:
points_gpu = points.cuda()   # 默认在GPU 0上
points_gpu = points.cuda(0)
points_cpu = points_gpu.cpu()
```

但是通过to method我们可以同时改变数据的部署设备和数据类型，因为to method可以传入device和dtype两个arguments。


### 3.10 Numpy interoperability

PyTorch tensors和Numpy arrays可以非常高效的互相转换。通过将PyTorch tensor转换为Numpy array，我们可以借用Numpy的广泛的功能。

```python
# In [1]:
points = torch.ones(3, 4)
points_np = points.numpy()
points_np

# Out [1]
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]], dtype=float32)
```

上述代码从points这个tensor里获得了points_np这个Numpy multidimensional array，其拥有着正确的size，shape，以及numerical type。有趣的是，返回的这个Numpy array和tensor storage共享同一个buffer。这表明只要tensor是在CPU RAM里的，这个tensor对应的Numpy array的Numpy method可以很高效的被执行。这同样表明改变Numpy array的值同样会改变原tensor的值。如果这个tensor是GPU上的，那PyTorch会复制一份这个tensor的内容给CPU上的Numpy array。

相反的，我们也可以从一个Numpy multidimensional array来获得一个PyTorch tensor：

```python
# In [2]:
points = torch.from_numpy(points_np)
```

这样生成的PyTorch tensor和原Numpy array关于buffer的关系和上面所述的一样。

>注意到，PyTorch默认的数据类型是32-bit floating-point，而Numpy默认的数据类型是64-bit floating-point。但正如我们之前所说的，我们希望使用32-bit floating-point，所以在转换后，我们要确认dtype=torch.float。


### 3.11 Generalized tensors are tensors, too

### 3.12 Serializing tensors

当数据很重要的时候，我们希望能将tensor存储下来，并且以后加载使用。毕竟我们不想每次都重新训练我们的模型。PyTorch使用pickle来serialize一个tensor，同时伴有精心设计的storage的serialization code。这是如何将points tensor存储在outpoints.t file的方法：

```python
# In [1]:
torch.save(points, '../data/p1ch3/ourpoint.t')

# 我们也可以用一个file descriptor来替代filename：

# In [2]:
with open('../data/p1ch3/outpoints.t', 'wb') as f:
    torch.save(points, f)
```

加载points回来也是很简单的：

```python
# In [3]:
points = torch.load('../data/p1ch3/outpoints.t')

# 或者下面的代码也可以：

# In [4]:
with open('../data/p1ch3/outpoints.t') as f:
    points = torch.load(f)
```

虽然说上述的方法存储和加载tensor都很方便，但是我们只能通过PyTorch来存储和加载它，这个文件其它的软件无法识别，not interoperative。所以更有用的是掌握如何将tensors存储为其他格式文件的方法。


#### 3.12.1 Serializing to HDF5 with h5py

HDF5格式是一种很常见的文件存储格式。HDF5是一种可移动的，广泛支持的用来表示serialized multidimensional arrays的文件格式，其组织成一种nested key-value字典的形式。Python通过h5py库来支持HDF5，这个库接收以及返回Numpy的arrays。

```python
# In [1]
conda install h5py  # 安装h5py库
```

从而我们可以通过先将points tensor转换为Numpy array（如之前所提，很方便），之后再将其用create-dataset function来存储：

```python
# In [1]:
import h5py

f = h5py.File('../data/p1ch3/ourpoints.hdf5', 'w')
dset = f.create_dataset('coords', data=points.numpy())
# coords是文件的key

f.close()
```

HDF5文件的key还可以是nested种类的。HDF5一个有意思的事情是我们可以在打开数据之后，数据仍然还在硬盘里的时候，index我们感兴趣的部分，然后只加载这部分的内容。

```python
# In [2]:
f = h5py.File('../data/p1ch3/outpoints.hdf5', 'r')
dset = f['coords']
last_points = dset[-2:]  # 我们只需要后两个点的内容被加载加来
```

上述代码里，当file打开的时候，数据没有被加载，仍然存在硬盘里。直到最后一行代码被执行之前，数据都还在硬盘里，没有加载到内存里。在最后一行执行之后，h5py获取了dset这个数据在硬盘上最后两个column的内容，并返回了那个区域的数据，以Numpy array的形式。

因为加载的数据是Numpy array的形式，所以我们可以使用torch.from_numpy funcion直接将返回的array转换为PyTorch的tensor。

```python
# In [3]:
last_points = torch.from_numpy(dset[-2:])
f.close()
```

当我们结束加载数据后，我们用f.close()关闭这个文件。关闭这个HDF5文件会使得dset失效，从而访问它会报错。我们之后只需要用last_points这个tensor就可以。


### 3.13 Conclusion

现在我们已经具备了将任意输入转换为floating-point number的先前知识。我们将会介绍tensor的其他知识，比如创建tensor的view，用tensor来index tensor，以及broadcasting，随着内容的深入而逐步介绍。

在chapter4里，我们将会学会如何在PyTorch里表示现实世界里的数据。我们从最简单的tabular data开始，之后逐渐到更复杂的数据。在这个过程中，我们会更深入的了解tensors。


### 3.14 Summary
* neural networks将floating-point representations转换为其它的floating-point representations。起始和结束的representations是人类所能够理解的，但是中间的那些representations比较难以理解。
* 这些floating-point representations存储在tensor里。
* tensor是multidimensional array，它们是PyTorch里最基本的数据结构。
* PyTorch对于tensor creation，manipulation和mathematical operations有着详尽的标准库。
* tensors可以被serialized到硬盘里，并加载回来。
* 所有的PyTorch里的tensor operations可以在CPU和GPU上被操作，而不需要改变代码。
* PyTorch使用带有一个下划线的函数来表示这个函数是作用在tensor上的in-place函数（也就是说它改变原tensor的值，不返回新的tensor），比如Tensor.sqrt_等。

```python
# In [1]:
a = torch.tensor(list(range(9)))
a = a.to(dtype=torch.float)       # 必须将a的数据类型转换，否则cos_函数将无法将原始值替换掉，因为原始的数据类型是int。
a.cos_()                          # 带有一个下划线结尾的函数会直接作用在原tensor上，并替换它的值，in-place操作
a

# Out [1]:
tensor([ 1.0000,  0.5403, -0.4161, -0.9900, -0.6536,  0.2837,  0.9602,  0.7539, -0.1455])
```




















---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

