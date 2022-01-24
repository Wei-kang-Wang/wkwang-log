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
*Fig 5. A CycleGAN trained to the point that it can fool both discriminator networks.*







---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

