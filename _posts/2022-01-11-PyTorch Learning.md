---
layout: post
comments: false
title: "PyTorch Learning"
date: 2022-01-11 01:09:00

---

> This post is about PyTorch learning.


<!--more-->

---

## A Talk About PyTorch Internals by Ezyang

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

假设我们想要获得tensor$$\left[0,1\right]$$位置的值，我们该如何将这个逻辑上的index $$\left[0, 1\right]$$ 转换到memory的物理地址呢？stride告诉我们如何实现：为了找到tensor每个位置的元素的物理地址，我们将逻辑上的index分别乘以该维度上的stride，并加在一起。在上图里，第一个维度涂了蓝色，第二个维度涂了红色，从而可以清晰的看到计算过程。在上例中，tensor $$\left[0, 1\right]$$ 所得到的sum是2，而且该tensor的部署方式是连续的，所以tensor$$\left[0,1\right]$$应该在从头数两个bytes之后，位于第三个byte的物理地址上。

在这篇talk之后，我还会介绍TensorAccessor，是一个类，用来解决index计算。当你看到了TensorAccessor而不是原始指针，那index的计算就已经被算好了。

stride是PyTorch实现为用户提供操作tensor的一个重要基础。比如，我们想要看看上述tensor第二行所表示的tensor长什么样，

![9]({{ '/assets/images/slide9.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

用上层高级index代码，我可以直接用tensor$$\left[1,:\right]$$来获取这一行。当我写这个代码的时候，我并没有真的创建了一个新的tensor，而是返回了一个原有tensor部分数据的一个新的view。这表明如果我在这个view里编辑了数据，我将会改变原有tensor该位置的值。在这个例子里，tensor $$\left[1, 0\right]$$ 和tensor $$\left[1, 1\right]$$ 位于连续的物理地址上，而且我们知道该行物理地址开始于初始地址两个bytes之后，所以我们只需要记录下每个元素的offset即可。（每个tensor都记录了一个offset，但是大多数时候都是0，所以上图例子中省略了）。

*Question from the talk: If I take a view on a tensor, how do I free the memory of the underlying tensor?

Answer: You have to make a copy of the view, thus disconnecting it from the original physical memory. There's really not much else you can do. By the way, if you have written Java in the old days, taking substrings of strings has a similar problem, because by default no copy is made, so the substring retains the (possibly very large string). Apparently, they fixed this.*

一个更有意思的例子是如果我们想要取第一列的数据，

![10]({{ '/assets/images/slide10.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

当我们看物理内存的时候，我们可以看出列元素并非连续存储的：每个列元素之间相隔一个元素。从而stride派上了用场：stride被指定为2，而不是1，表示每两个连续元素间，需要跳过2个slots（正就是为什么这个metadata叫做stride的原因：如果我们将index想作在物理内存上walk的位置，stride告诉了我们每一步前进该跳过几个位置）。

stride可以让你表示tensor各种有意思的view。

我们考虑内部该如何实现stride的功能。如果我们可以取tensor的view，说明我们需要在内部设计时，将tensor的定义（用户层所涉及的内容）与实际存储tensor数据的物理数据（storage）分离开：

![11]({{ '/assets/images/slide11.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

可能有多个tensors共享同一个storage。storage定义了tensor的dtype和physical size，而每个tensor记录了每个维度的size，stride，offset，定义了物理内存的高层逻辑理解。

一件需要注意的事是永远都会有个tensor-storage对，即使是你实际上不需要storage的很简单的情况（比如，分配了一个连续的tensor，*torch.zeros(2, 2)*)。

#### Tensor Operations

我们已经说了一些tensor的数据分布内容。接下来我们简要介绍一下tensor的operation是怎么实现的。在最高层，最抽象的层面，当你写*torch.mm*的时候，发生了两个dispatch：

![12]({{ '/assets/images/slide12.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

第一个dispatch基于tensor的device类型和布局：比如是CPU还是CUDA tensor，再比如是strided tensor或者是sparse tensor。这是个dynamic dispatch：这是个virtual function调用（这个virtual function具体在哪个位置被调用实际上是这篇talk第二部分所要讨论的内容）。这里需要一个dispatch是合理的：CPU矩阵乘法的实现和CUDA上的实现是很不一样的。这是个dynamic dispatch是因为这些kernels在不同的libraries里（比如，*libcaffe2.so* versus *libcaffe2_gpu.so*)，所以别无选择：如果你想用一个library里的函数却没有对这个库直接的依赖，那么就得用dynamic dispatch。

第二个dispatch是因为dtype。这个dispatch仅仅是一个简单的switch表达式，为了让kernels去选择支持各种各种的dtypes。这里需要dispatch也是合理的：实现乘法的CPU代码（或者CUDA代码）对于float和int数据类型是不一样的。所以需要对不同的dtype用不同的kernels。

#### Tensor Extensions

![13]({{ '/assets/images/slide13.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

有很多有意思的tensor拓展类，比如XLA tensor，quantized tensor， MKL-DNN tensor，等。一个需要考虑的重要的事情，就是我们该如何适配这些拓展类。

![14]({{ '/assets/images/slide14.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

我们现有的模型为拓展类提供四个拓展的点。首先，决定tensor的三个参数是：

* device，描述tensor的物理内存存放在哪的描述，比如，CPU，或NVIDIA CPU（CUDA），或者AMD GPU（hip）或TPU（xla）。每个设备都有他自己的allocator，和其他设备不兼容，所以需要为每种设备独立定义tensor。
* layout，描述如何高层逻辑上解释物理内存中存放的数据。最常见的layout就是strided tensor，但是sparse tensor有一个不同的layout，包含一对tensors，一个存储index，另一个存储data。MKL-DNN tensor拥有更复杂的layout，不能仅仅被stride所表示。
* dytpe，描述tensor里每个元素到底存放了什么数据。可以是floats或者integers，或者quantized integers等。

如果你想为PyTorch的tensor加上拓展，你需要考虑拓展哪个上述参数。上述三个参数每个都对应了一个向量，存放了各种可能的值，比如device=$(CPU, CUDA, hip,...)$。从而这三个向量的Cartesian积定义了所有你可能使用的tensor类型。但是现在并不是所有的类型都被实现了，即并不是所有的kernels都有（比如，sparse quantized tensors on FPGA就没有定义相对应的tensor类），但是原则上这些组合都是合理的。

还剩最后一种可以拓展tensor方法的手段，就是写一个wrapper类，包裹住PyTorch tensor类，实现你所需要的object type。利用wrapper实现tensor类方法的扩展更加清晰，使得代码具有更好的可读性和将来的可扩展性，前面的三种参数的修改，均可以通过wrapper来修改。

所以什么时候用tensor wrapper，而什么时候拓展PyTorch tensor本身呢？关键在于你是否需要这个tensor extension在autograd backward的时候被考虑，PyTorch tensor本身的拓展是会被考虑的，而tensor wrapper是不会的。比如，在优化输出embedding的network时，我们希望embedding的梯度是稀疏的，从而需要tensor的一个sparse拓展，而这个梯度是需要参与autograd backward计算的，从而不能定义一个tensor wrapper，因为它仅仅是一个Python object，而需要在原PyTorch tensor的基础上拓展，使得拓展后的sparse tensor仍然是一个PyTorch tensor类。

![15]({{ '/assets/images/slide15.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


### Autograd














