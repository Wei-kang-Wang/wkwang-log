---
layout: post
comments: false
title: "A Talk About PyTorch Internals by Ezyang"
date: 2022-01-11 01:09:00

---

> This is A Talk About PyTorch Internals by Ezyang.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

这篇文章是Ezyang的一篇[talk](http://blog.ezyang.com/2019/05/pytorch-internals/)的翻译并加上自己的理解。此talk主要讲了PyTorch的内部机理。

这篇文章是为了那些已经用过PyTorch并且希望能够给PyTorch代码库做出贡献却被PyTorch背后的C++代码唬住的人。实话说，有时候PyTorch的代码确实比较难懂。这篇文章的目的是给你一份有关系PyTorch的总体指南，告诉你PyTorch作为一个tensor library that supports automatic differentiation的基本概念框架，并且教会你如何找到自己的方式来学习代码库。这篇文章假设你已经写过一些PyTorch代码，但是还没有完全明白机器学习代码库是怎么写的。

这篇文章主要分为两部分（Concepts和Mechanics）。在第一部分里，我将会介绍tensor library的基本知识。从所熟知的tensor数据类型开始，并且仔细说明这个数据类型的属性到底为我们提供了哪些功能和内容。这可以让我们对于代码背后到底发生了什么有更好的理解。我们还会讨论extension points的三个方面：layout，device以及dtype，这可以让我们更深的理解tensor class的扩展。最后我还会讨论一些autograd的内容。在第二部分里，我们将会讨论PyTorch代码里复杂难搞的细节。我会告诉你在大片的autograd代码里哪一部分是核心的，哪一部分是legacy，并且介绍PyTorch提供的写扩展代码的工具。

## 第一部分：Concepts

### 1. Tensor

Tensor是PyTorch里最核心的数据结构。你可能对一个tensor长什么样有着良好的直觉：一个包含着某种标量类型的n维的数据结构，标量可以是floats，ints等。我们可以认为一个tensor是一个含有一些数据，同时也含有一些metadata用来描述这个tensor的大小，所含的数据的类型（dtype），这个tensor分布在什么设备上（CPU memory或者是CUDA memory）等的综合体。

![6]({{ '/assets/images/slide6.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

有些metadata你可能不是很了解：stride。stride实际上是PyTorch一个很独特的特征，所以值得单独讨论一下。

![7]({{ '/assets/images/slide7.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

一个tensor是一个数学概念。但是为了在计算机上表达它，我们需要定义某种physical representation。最常见的representation是将tensor的每个值连续的部署在memory里，将每一行都写在memory里，正如上图所示的那样。在上述例子中，我规定了tensor包含32bit的整数类型，所以你可以看到每个整数部署在一个物理地址里，每一个物理地址都相隔4bytes。为了记住tensor的实际维度，我们还需要将每个维度的size记录下来作为metadata。（部署在memory的时候，是将高维的tensor按某种规则展开成了一维）

所以说，stride起到了什么作用呢？

![8]({{ '/assets/images/slide8.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

假设我们想要获得tensor$$\left[0,1\right]$$位置的值，我们该如何将这个逻辑表达式里的index $$\left[0, 1\right]$$ 转换到memory的物理地址呢？stride告诉我们如何实现：为了找到tensor每个位置的元素的物理地址，我们将逻辑表达式里的index分别乘以该维度上的stride，并加在一起。在上图里，第一个维度涂了蓝色，第二个维度涂了红色，从而可以清晰的看到计算过程。在上例中，tensor $$\left[0, 1\right]$$ 所得到的sum是2，而且该tensor的部署方式是连续的，所以tensor$$\left[0,1\right]$$应该在从头数两个bytes之后，位于第三个byte的物理地址上。

在这篇talk之后，我还会介绍TensorAccessor，是一个类，用来解决index计算。当你使用TensorAccessor，那index对应的物理地址的结果会被直接提供给你。

stride是PyTorch实现为用户提供tensor操作的一个重要基础。比如，我们想要看看上述tensor第二行所表示的tensor长什么样，

![9]({{ '/assets/images/slide9.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

用上层高级index代码，我可以直接用tensor$$\left[1,:\right]$$来获取这一行。当我写这个代码的时候，我并没有真的创建了一个新的tensor，而是返回了一个原有tensor部分数据的一个新的view。这表明如果我在这个view里编辑了数据，我将会改变原有tensor该位置的值。在这个例子里，tensor $$\left[1, 0\right]$$ 和tensor $$\left[1, 1\right]$$ 位于连续的物理地址上，而且我们知道该行物理地址开始于初始地址两个bytes之后，所以我们只需要记录下每个元素的offset即可。（每个tensor都记录了一个offset，但是大多数时候都是0，所以上图例子中省略了）。

>*Question from the talk: 如果我们建立了一个tensor的view，那我们如何在不影响这个view的情况下，将原tensor的内存解放呢（也就是删除原tensor）?
>
>Answer: 你需要构建一个这个view的copy，从而从原tensor对应的physical memory里解放出来。实际上，除了这个方法似乎也没有什么别的方法可行。而且，如果你以前写过Java，那你就会发现，在旧版的Java里，取一个string的substring和这里的情况差不多，你其实并没有真的在physical memory里分配新的数据，所以你需要一直保证这个string的内容不被修改，才能正确的使用substring（这个问题现在已经被Java修复了）.*


一个更有意思的例子是如果我们想要取第一列的数据，

![10]({{ '/assets/images/slide10.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

当我们看物理内存的时候，我们可以看出列元素并非连续存储的：每个列元素之间相隔一个元素。从而stride派上了用场：stride被指定为2，而不是1，表示每两个连续元素间，需要跳过2个slots（正就是为什么这个metadata叫做stride的原因：如果我们将index想作在物理内存上walk的位置，stride告诉了我们每一步前进该跳过几个位置）。

stride可以让你表示tensor各种有意思的view。

我们在看过了stride的几个例子之后，来考虑内部该如何实现stride的功能。如果我们可以取tensor的view，说明我们需要在内部设计时，将tensor的定义（用户层所涉及的内容）与实际存储tensor数据的物理数据（storage）分离开：

![11]({{ '/assets/images/slide11.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

可能有多个tensors共享同一个storage。storage定义了tensor的dtype和physical size，而每个tensor记录了每个维度的size，stride，offset，定义了物理内存的高层逻辑。

一件需要注意的事是永远都会有个tensor-storage对，即使是你实际上不需要storage的很简单的情况（比如，分配了一个连续的tensor，*torch.zeros(2, 2)*)。

#### 1.1 Tensor Operations

我们已经说了一些tensor的数据分布内容。接下来我们简要介绍一下tensor的operation是怎么实现的。在最高层，最抽象的层面，当你写*torch.mm*的时候，发生了两个dispatch：

![12]({{ '/assets/images/slide12.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

第一个dispatch基于tensor的device类型和layout：比如是CPU还是CUDA tensor，再比如是strided tensor或者是sparse tensor。这是个dynamic dispatch：这是个virtual function调用（这个virtual function具体在哪个位置被调用是这篇talk第二部分所要讨论的内容）。这里需要使用一个dispatch是合理的：CPU矩阵乘法的实现和CUDA上的实现是很不一样的。这是个dynamic dispatch是因为这些kernels在不同的libraries里（比如，*libcaffe2.so* versus *libcaffe2_gpu.so*)，所以别无选择：如果你想用一个library里的函数却没有对这个library直接的依赖，那么就得用dynamic dispatch。

第二个dispatch是因为dtype。这个dispatch仅仅是一个简单的switch表达式，让kernels去选择支持各种各种的dtypes。这里需要dispatch也是合理的：对于乘法操作，CPU代码（或者CUDA代码）对于float和int数据类型是不一样的。所以需要对不同的dtype用不同的kernels。

上述过程可能是你理解PyTorch里tensor上的operations最重要的部分。


#### 1.2 Tensor Extensions

![13]({{ '/assets/images/slide13.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

既然我们说到了PyTorch里的Tensor，我们就不得不提各种各样的tensor extensions。毕竟，dense，CPU上的float tensors只是Tensor家族里很小的一部分。PyTorch里有很多有意思的tensor extensions，比如XLA tensor，quantized tensor， MKL-DNN tensor，等。而我们需要考虑的事情，是我们该如何扩展出这些tensor类型。

![14]({{ '/assets/images/slide14.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

我们现有的模型为拓展类提供四个拓展的点。首先，决定tensor的三个参数是：

* device，描述tensor的物理内存存放在哪的描述，比如，CPU，或NVIDIA CPU（CUDA），或者AMD GPU（hip）或TPU（xla）。每个设备都有他自己的allocator，和其他设备不兼容，所以需要为每种设备独立定义tensor。
* layout，描述如何高层逻辑上解释物理内存中存放的数据。最常见的layout就是strided tensor，但是sparse tensor有一个不同的layout，包含一对tensors，一个存储index，另一个存储data。MKL-DNN tensor拥有更复杂的layout，不能仅仅被stride所表示。
* dytpe，描述tensor里每个元素到底存放了什么数据。可以是floats或者integers，或者quantized integers等。

如果你想为PyTorch的tensor加上拓展，你需要考虑拓展哪个上述参数。上述三个参数每个都对应了一个向量，存放了各种可能的值，比如device=$(CPU, CUDA, hip,...)$。从而这三个向量的Cartesian积定义了所有你可能拓展的tensor类型。但是现在并不是所有的拓展都被实现了（比如，sparse quantized tensors on FPGA就没有定义相对应的tensor类），但是原则上这些组合都是合理的。

还剩最后一种可以拓展tensor方法的手段，就是写一个wrapper类，包裹住PyTorch tensor类，实现你所需要的object type。利用wrapper实现tensor类方法的扩展更加清晰，使得代码具有更好的可读性和将来的可扩展性，前面的三种参数的修改，均可以通过wrapper来修改。

所以什么时候用tensor wrapper，而什么时候拓展PyTorch tensor本身呢？关键在于你是否需要这个tensor extension在autograd backward的时候被考虑，PyTorch tensor本身的拓展是会被考虑的，而tensor wrapper是不会的。比如，在优化输出embedding的network时，我们希望embedding的梯度是稀疏的，从而需要将tensor的扩展为一个sparsed tensor，而这个梯度是需要参与autograd backward计算的，从而不能定义一个tensor wrapper，因为它仅仅是一个Python object，而需要在原PyTorch tensor的基础上拓展，使得拓展后的sparse tensor仍然是一个PyTorch tensor类。如果要是定义了一个tensor wrapper，那他就是一个有着记录index的tensor和记录value的tensor的Python object，这样的object在通过autograd backward之后计算的梯度，将不会是sparsed tensor对应的正确的梯度。

![15]({{ '/assets/images/slide15.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

我们关于extension如何设计的原则同样也影响了tensor本身的layout。我们希望tensor有一个固定的layout：因为我们不希望很基础的一些operations比如这个tensor的size是多少还需要virtual dispatch。所以当我们来看一个Tensor的layout时，首先是一个所有的tensor-like object都会有的prefix，然后是strided tensors所需要的fields（因为strided tensors过于常见），再之后是用户自己定义的fields（比如sparse tensor的index和values都在这个suffix里）。


### 2. Autograd

我们已经讲了PyTorch里Tensor的全部内容，但如果这是PyTorch仅仅能够提供的内容，那PyTorch也就只是NumPy的一个克隆罢了。PyTorch最具有辨识度的特征就是它为tensors提供了automatic differentiation！

automatic differentiation是做什么的？它是负责自动计算一个neural network的loss关于这个neural networks内所有需要计算梯度的parameters的梯度：

![16]({{ '/assets/images/slide16.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

![17]({{ '/assets/images/slide17.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

上述过程有很多需要细讲的内容：

* 首先，来看上面图片所示的代码里红色和蓝色的变量。PyTorch实现的是reverse-mode automatic differentiation，也就是说它是沿着forward计算反向来计算gradients的。我们看到，在红色变量的最下面，我们计算出了loss；然后我们首先计算出的是grad_loss。loss是由next_h2计算所得，所以再然后我们计算grad_next_h2。严格意义上，上述这些grad_开头的变量都不是真正的gradients，它们实际上是左乘了一个vector的Jacobian，但是在PyTorch里我们就这么叫了，大家也都知道是什么就行了。
* forward计算的每一行，在backward计算里都会被替换为另一种计算，用来表示forward里该计算的derivative。比如说，tanh被换成了tanh_backward（在图里用灰色的线连起来了）。forward和backward过程是相反的：如果一个forward计算输出next_h2，那其对应的backward里的计算会用grad_next_h2作为输入。

autograd最关键的点在于上述图里的计算是被执行的，但从不需要具体的写出这些步骤。

为了实现这个效果，我们需要在tensors的operation上存储更多的metadata。让我们来改进一下tensor data structure：除了仅仅是指向physical address的一个tensor，我们现在有了一个wrap这个tensor的variable，这个variable同时也存储了更多的信息（AutogradMeta）用于在loss.backward()被调用的时候来实现autograd。

![18]({{ '/assets/images/slide18.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

我们同时也需要更新我们关于dispatch的图示：

![19]({{ '/assets/images/slide19.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

在选择CPU还是GPU这个dispatch之前，还有另一个有关variables的dispatch，是用来unwrapping variables用的，这个dispatch调用某种implementations（蓝色的部分），然后将结果重新wrap到variables里，并且记录下backward所需要的autograd metadata。

某些implementations并不会unwrap，它们会调用其它variables的implementations。

接下来的几张PPT还没有写解释，尽情期待。


#### 2.1 AutogradMeta

![20]({{ '/assets/images/slide20.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

![21]({{ '/assets/images/slide21.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

![22]({{ '/assets/images/slide22.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

![23]({{ '/assets/images/slide23.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


#### 2.2 Autograd Engine

![24]({{ '/assets/images/slide24.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

![25]({{ '/assets/images/slide25.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


#### 2.3 Parallelization across devices

![26]({{ '/assets/images/slide26.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}



## 第二部分：Mechanics

关于concepts我们了解得够多了，现在我们来看看code。

### 1. Find your way around

PyTorch code有很多的folders，在CONTRIBUTING document里也详细说明了每个folder的作用，但实际上，只有4个folders是你真正需要留心注意的：

![27]({{ '/assets/images/slide27.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

* 首先，torch/包含了你最熟悉的内容：在实际使用过程中import和使用的Python modules。这一部分是用Python写的，比较容易看懂它们的源码。
* torch/csrc/，C++的代码，实现了PyTorch的前端。更详细的说，它实现了Python和C++代码之间的连接，这部分也是如同autograd engine和JIT compiler一样的PyTorch里很重要的内容。它包含了C++前端的代码。
* aten/，是”A Tensor Library"的简写，是一个实现Tensor上的operations的C++ library。如果你想寻找某个kernel的代码在哪里，那它可能就在aten/里。aten/本身分为两个部分：一部分是由C++写的operators，一部分是由C写的operators，C的那一部分是老的代码，C++是顺应发展潮流更新的，所以尽量别去使用以及不需要花很多时间在C的那部分代码上。
* c10/，包含PyTorch的核心，包括Tensor和storage数据结构的实现。

当我们调用一个function，比如说torch.add，实际上发生了什么？如果还记得之前关于dispatch的讨论，你应该在脑中已经有了一个基本的流程：

* 首先从Python code转换为C++ code（python argument parsing）
* 然后处理variable dispatch（这和用什么programming language没有关系，但是是个不可或缺的dispatch，因为对于不同的variable类型，kernels和library都不一样）（VariableType）
* 处理device type/layout这个dispatch（Type）
* 现在我们获取到了实际的kernel，要不然是一个modern native function（C++写的），要不然是一个legacy TH function（C写的）

上述code的划分用图示表示：

![28]({{ '/assets/images/slide28.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

上述每一步都对应着明确的某些code，让我们来仔细看看。

![29]({{ '/assets/images/slide29.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

将Python code转换为C++ code首先会到达它的C implementation，其展现给Python部分的内容类似于torch.\_C.VariableFunctions.add。THPVariable_add是这个C implementation的implementation。

关于上述code所需要了解的一个内容是，它是自动生成的。如果你在Github里查找这部分内容是没有的，因为你需要实际上来构造PyTorch来看到它。另一个重要的事情是，你并不需要深刻了解这部分代码的细节；你应该跳过他们，而只需要明白他们做了什么就可以了。在上面的代码里，我将某些重要的部分标了蓝色：你可以看到PythonArgParser这个class的使用，从Python args和kwargs里将C++ objects提取了出来；之后我们调用了一个dispatch_add function（用红色线标记）。之后，我们重新wrap了Tensor，并将其返回成了一个PyObject。

![30]({{ '/assets/images/slide30.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

当你调用Tensor class的add method时，并没有virtual dispatch会发生。取而代之的是，有一个inline method来对"Type" object调用一个virtual method。这个method是一个实际上的virtual method（所以说转换为Type实际上就是让你能用virtual dispatch的一个小技巧）。在这个例子里，这个dispatch会应用一个叫做TypeDefault的定义在class上的add。之所以会这样是因为对于不同的device（CPU或者CUDA），add的implementation是一样的；如果有着不同的implementation，那么我们可能就会应用某个叫CPUFloatType::add的东西。正是这个virtual method的implementation最后带我们到了真正的kernel code。

在我们将kernel之前，我们需要再强调一遍，上面的所有的code都是在构建了PyTorch代码之后自动生成的。

![31]({{ '/assets/images/slide31.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

上面的内容过于复杂繁琐，所以建议只要有一些基本的认识就可以了，主要注意力还要集中在kernels和Python实现上。


### Writing kernels

PyTorch为kernel writers提供了很多的有用的工具。在这节里，我们将会介绍一部分。但是首先，要写kernel，我们需要什么？

![32]({{ '/assets/images/slide32.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

总的来说，我们认为一个PyTorch kernel包含以下几部分：

* 首先，有一些关于kernel的metadata，可以用于促进code generation以及链接Python code，而不需要额外再写代码来实现这些功能
* 一旦到达kernel，那就已经过了device type/layout dispatch的部分。所以说第一件事就是error checking，要确认input tensors有着正确的dimensions
* 之后，分配result tensor，我们要将输出写入这个tensor
* 终于到了kernel的时间。在此时，你需要做第二次dtype dispatch，从而跳转到这个dtype所对应的kernel
* 大多数kernels都有一些parallelization的操作，从而你可以借用Multi-CPU的优势（CUDA kernels本身就是parallelized，因为CUDA本身就是parallelization的编程语言）
* 最终，你获取了数据，并可以开始做任何你想做的计算了。

在接下来的slides里，我们将会介绍可以帮助你实现上述这些steps的PyTorch的tools。

![33]({{ '/assets/images/slide33.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}










