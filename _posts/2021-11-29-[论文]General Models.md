---
layout: post
comments: false
title: "[论文]Convolutional Neural Networks & Multi-Layer Perceptrons."
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of general machine learning models.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## Convolutional neural network architectures

### 1. [AlexNet: ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
*ALex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton* 

*NIPS 2012*

AlexNet是深度学习浪潮的奠基作之一。

这篇文章里的模型赢得了ILSVRC 2012年的classification任务的冠军。

#### 1. title
标题很简洁明了，用什么方法、模型（deep convolutional neural network）解决了什么问题（ImageNet classification）。

#### 2. authors
Hinton大佬的文章，那肯定是值得看的。

#### 3. abstract

我们设计了一个很大、很深的cnn来解决ImageNet LSVRC-2010的120万张图片分为1000个类的任务。在测试数据上，top1和top5的错误率都远远比其它的方法要好。

>这种写法其实是很少见的，第一句话说我做了什么，第二句话就直接展示效果并且说十分好。

这篇文章里的神经网络是很大的，有6千万个参数以及650000个神经元，而且有五个卷积层，三个全连接层，最后还有个长为1000的softmax。为了使得训练更快，我们设计了方法，使得神经元是位于non-saturated区域的，并且用了高效的GPU设计方法来训练。我们还用了dropout来使得全连接层不至于过拟合。我们最后将训练后的模型再应用到ILSVRC-2012的比赛里，达到了top-5比第二名低很多的错误率。

>这个摘要主要就说了我们用了什么方法做了什么事情，介绍了一下方法、模型具体的设计细节，之后就说了在实验上比其他方法远远要好。其实不太像是摘要，反而像是技术报告（technic report）的写作方式。但毕竟是大佬的文章，结果就是要比其他方法好的非常多。

#### 4. Introduction
>一篇文章的第一段一般都是在讲一个故事，我们在做一个什么问题，这个问题为什么重要等。

我们要做object detection这个问题。为了提升它的性能，我们需要收集更大的数据集，学习更好的模型，以及使用技术来使得模型不要过拟合。

>用一些正则的方式使得模型不要过拟合在当时是个很重要的方向，但实际上现在有很多研究表明设计合理的网络结构要比使用正则方法要更加重要，这仍然是一个未解决的研究方向。

再之后就是介绍了各种数据集，因为文章的标题是解决ImageNet的数据集上的任务，所以最后引出ImageNet，吹一波ImageNet很好。

为了能够在上百万张照片里学习1000个种类，我们需要一个具有很大的learning capacity的模型。之后就引入了我们要用CNN来解决这个问题，但是CNN设计的很大并不容易，因为会overfitting而且可能会训练不动。

>这种写法是有问题的，因为当年的主流并不是CNN，而是其它的方法。所以这里一点都不提到其它的方法，而直接说解决这个问题就用CNN，其实是一个不客观，且比较窄的视角。

再之后说明CNN比较难训练，但对于这样比较大的数据集，确实可以训练一个比较大的CNN。而且有了GPU的帮助，可以使得训练高效很多。

接下来说了这篇论文的贡献。这篇论文在ImageNet上训练了一个最大的神经网络，并在ILSVRC-2010和ILSVRC-2012这两个任务上都取得了很好的结果。我们还实现了在GPU上高效的部署。而且我们提出的新的模型能够产生一些新的比较特殊的feature，能够使得效果很好而且减少训练的时间。而且我们做了一些防止过拟合的方法。最后说明网络的结构，有五个卷积层，三个全连接层，而且深度很重要：移除掉任何一层都不行。

>因为这篇文章提出了新的结构，有了新的不常见的feature来解决classification的问题，所以才让这篇文章有了很大的价值，成为了奠基作。而不是仅仅通过很多技术上的操作使得模型取得了第一名，但实际上并没有思想或者结构上的创新，只能让其他人觉得复杂，自叹不如，或者没有能力做如此复杂的工作，但并不具有启发性。而这篇文章提出了新的内容，相当于为后人挖了大坑。

最后介绍了一下用了GTX 580 GPU来训练。

#### 5. Dataset
>因为这篇文章是在ImageNet这个数据集上解决其classification的任务，所以得介绍一下这个数据集。

数据集介绍的部分就不说了，主要介绍了数据集的内容，以及ILSVRC这个竞赛的内容。

（但这部分最后一段很重要）ImageNet实际上每张图片的分辨率是不一样的，也就是说图片是没有裁剪好的。这篇文章里的做法简单粗暴，直接将短边rescale到256，而长边等比例缩小之后，直接在rescale的图片中心区域裁剪出256$$\times$$256大小的部分。除了再将每个通道的平均值减掉这个操作以外，并没有任何别的pre-processing操作，整个模型是在raw-RGB图片上操作的。

>当时大多数的方法都需要从原始输入图片里抽取特征再用模型进行处理，比如说sift特征等等。但是这篇文章不需要抽取任何特征，实际上这是很大的一个进步，但这篇文章并没有把其作为一个卖点。这种方法实际上是end-to-end，也就是说原始数据直接输入模型，没有任何特征提取，结果直接输出。文章没有重点提到这个实际上是历史的局限性，之后的研究表明end-to-end是深度神经网络一个很大的卖点，应该要予以强调。

#### 6. Architecture

##### 6.1 ReLU nonlinearity

运用到神经网络里，饱和的那些非线性函数$$tanh(x)$$、$$sigmoid(x)$$等实际上要比非饱和的那些非线性函数比如说$$ReLU(x)$$使得训练慢很多。从fig 1就能看出，虚线是使用tanh的，实线是使用ReLU的，横轴是epoch数，纵轴是training error，可以看出来效果差别挺大。

![RELU]({{ '/assets/images/ALEXNET-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 我们这里用CIFAR-10数据集，对于一个4层的CNN进行了测试，用的是tanh和ReLU两种不同的非线性函数。而对于每种情况，我们设置的learning rate不一样，learning rate设置成为在每种情况下，都选择使该种情况下训练最快的值。我们可以看到，ReLU使得训练快了很多。*

##### 6.2 Training on Multiple GPUs

介绍的如何用多个GPU训练的，主要都是技术、工程上的细节。和ML、CV关系不大。

##### 6.3 Local Response Normalization

实际上这个东西也不重要，后续也没有什么人用。

>提到了一点，ReLU不是饱和型的非线性函数，所以在输入ReLU之前实际上并不需要对输入做什么操作（比如说将输入集中到非饱和区域等）。

##### 6.4 Overlapping Pooling

将传统的不overlap的pooling改成了overlap的pooling，会使得效果好很多。

##### 6.5 Overall Architecture

用了五个卷积层，三个全连接层，再加上一个1000-way的softmax，如fig 2所示：

![Model Structure]({{ '/assets/images/ALEXNET-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. 模型的结构*

>输入是一个很扁的很宽的图片$$224 \times 224 \times 3$$，然后通过网络将高宽不断变小，而深度不断增加。随着网络的增加，慢慢将空间信息压缩，即高宽逐渐减小，到了最后一个卷积层，已经变成了$$13 \times 13$$，也就是说认为这里一个像素已经可以代表前面很大一块的像素。而随着深度的增加，通道数也在逐渐的增加，可以认为每个通道是在表示一种特定的特征，比如说192个通道，我可以认为它能够识别输入的图片的192种不同的特征，比如说某个通道用来识别边，某个通道用来识别圆等等。所以说，空间信息在逐渐压缩，而语义信息在逐渐增加（通道数就是语义信息）。而最后，到了全连接层。所以说，输入的$$224 \times 224 \times 3$$的图片，最后通过网络变成了一个$$4096$$长度的向量，最后再用一个线性分类器做分类。实际上，这个$$4096$$的向量具有很好的语义信息，所以效果很好，也可以从后面的fig 3看出。

>整个机器学习，都可以看成知识压缩的过程，原始的数据不管是图片、文字还是视频，最后都通过一个向量来表示，而这个向量所含有的语义信息，可以让机器进行识别。从而可以在上面做各种各样的事情。这也是神经网络精髓之所在。


#### 7. Reducing Overfitting

##### 7.1 Data Augmentation

输入是$$256 \times 256 \times 3$$大小的图片，而文章会随机裁剪$$224 \times 224 \times 3$$大小的部分作为输入，引入了随机性，增强了模型的generalization的能力。

还对RGB通道做了PCA的操作。

##### 7.2 Dropout

将很多个模型融合在一起找到效果最好的那个，这个方法是很常见的。但是对于deep neural networks来说，本来训练就很贵了，这么做是不现实的。这篇文章用了dropout的操作，也就是说随机的将模型里某些输出直接设置为0。这个操作可以使得我们每次都会得到不一样的模型，但这些模型的权重实际上是共享的，从而最后等价于很多个模型做融合。

>但后来的工作表明dropout实际上并不是在做模型融合，更像是正则项。之后的工作里说明dropout实际上等价于一个L2的正则项。

#### 8. Details of Training

介绍了用SGD方法来训练模型。使用了weight decay，也使用了momentum。

还介绍了一下模型参数的初始化操作。

最后介绍了一下learning rate的设置方法。

#### 9. Results

介绍了在ImageNet上的classification任务和其他方法的对比，有ILSVRC-2010和ILSVRC-2012两个比赛。

##### 9.1 Qualitative Evaluations

我们从fig 3可以看到，模型的效果很不错。而从fig 3的右侧发现，feature向量相近的那些输入图片，内容确实是相近的，这其实是模型之所以效果好的很重要的点，这说明模型确实学习到了有效的语义信息，对于语义信息相似的图像，能生成距离相近的feature向量，从而在feature空间中能够很容易的分类。

![test]({{ '/assets/images/ALEXNET-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. 左图显示了在Iamgenet数据集里找的一些图片，以及它们通过模型输出的top-5的预测结果。而右图的第一列是随机选取的图片，对于每一行来说，第一列的图片输入模型后可以得到最后的4096长的feature向量，而我们寻找到那些图片，其feature向量与此向量的距离最近。*

>神经网络的可解释性一直是一个研究热点，而此处也提及到了一部分。


#### 10. conclusion
>这篇文章是没有结论的，只有discussion。指明了未来可能还需要做些什么事情。但一篇文章的结论conclusion通常来说是和abstract的对应，所以说没有conclusion是很少见的，也是不推荐的。

我们这篇文章的结果表明深的、大的CNN对于很难的任务是效果很好的。如果我们的模型去掉一层，那么结果会变差，这说明深度也是有必要的。

>其实这里论证不太严谨，因为去掉一层结果会变差也可能是其它的参数没设置好，实际上去掉一层，再改改参数，还是能达到一样的效果的。但是歪打正着，这个结论是正确的，深度确实是很必要的。

我们这篇文章所使用的方法还没有用到任何的unsupervised pre-training的方法，并没有预先transform数据，而且我们所解决的是一个supervised learning的问题。

>实际上机器学习在长时间的范围内研究的都是unsupervised learning，Hinton还有LeCun等人一直都认为unsupervised learning才是主流方向。但这篇文章的出现，使得supervised learning火了起来，一直到最近Bert还有GAN的出现，才让unsupervised learning重新回到人们关注的视野里。实际上在这篇文章之前，neural network一直做的是unsupervised learning的工作，之所以不做supervised learning是因为比不过SVM，所以只能做一些SVM做不了的工作。但这篇文章的出现说明深的、大的neural network可以在supervised learning打赢其它方法了，所以引起了轰动和热潮。



### 2. [VGGNet: Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
*Karen Simonyan, Andrew Zisserman*

*ICLR 2015*

#### 1. Title
和AlexNet、ResNet的标题一样的风格，都是直接说明什么模型（方法）解决什么问题。

#### 2. Authors
Andrew Zisserman是Oxford VGG实验室的leader，很强的组。

#### 3. Abstract
这篇工作我们探究了CNN的深度对于大规模image recognition任务精度的影响。我们的主要贡献是十分详尽的研究了网络深度的作用，而我们的网络使用的是很小的convolutional filters（$$3 \times 3$$），我们的结果显示深度到16-19层的时候，任务的精度能有很大的提升。这些结果是我们参加ImageNet Challenge 2014的模型的基础，我们的模型赢得了localisation的第一名，以及classification的第二名。我们同时也说明我们的方法对于其它的数据集仍然有很好的效果。我们将两个效果最好的CNN模型公布了，希望能够帮助之后在CVi领域使用深度视觉特征的研究。

#### 4. Introduction
CNN最近在大规模的image和video recognition任务中获得了巨大的胜利，也是因为大规模的公开的数据集比如说ImageNet，以及高性能的计算资源比如说GPU，才使得这种成功成为可能。在ILSVRC-2014中，deep visual recognition architecture已经展露了头角，在前几年的比赛里，ILSVRC-2011的冠军使用的是high-dimensional shallow feature encodings，ILSVRC-2012年的冠军使用的是deep CNN（也就是AlexNet）。

随着CNN在CV领域逐渐变得常见，有很多人尝试改进AlexNet以获得更高的精度。比如说，ILSVRC-2013年的冠军就是将AlexNet改进为第一层使用更小的convolutional filter以及更小的stride。另一条改进AlexNet效果的路则是利用不同scale的输入image来训练。在我们这篇文章里，我们解决CNN结构的另一个重要的问题：深度。为了达到这个目的，我们将架构的其它参数固定，然后通过增加更多的convolutional layer来使得网络加深，这在训练上和计算上都是可行的，因为我们使用了$$ 3 \times 3$$的filters（$$ 3 \times 3$$的filters配合padding和stide=1可以使得输出的feature map和输入的feature map长宽不变，从而可以无限的加深下去）。

结果是，我们获得了一个精度更高的CNN模型，不仅在ILSVRC classification和localisation任务上取得了sota的效果，还可以用在其它的image recognition数据集上，仍然可以获得很好的效果，即使是很简单的设计（比如说将deep features直接利用SVM分类，不需要任何的fine-tuning）。我们公开了两个效果最好的模型为了将来的研究。

这篇文章剩下的部分结构如下。在section5里，我们描述了我们设计的CNN模型。image classification的训练和测试细节再section6里。我们的框架和其它人的模型的对比再section7里。section8总结整篇文章。


#### 5. ConvNet Configurations
为了在一个公平的设定下研究增加CNN的深度带来的影响，我们所有的CNN层都使用同一个结构。在这一个section里，我们先描述我们的CNN模型的整体结构，然后再描述一些细节的设置。我们的设计细节最后再和之前的工作进行对比。


##### 5.1 Architecture
在我们的训练过程中，我们的CNN输入是一个固定大小的$$224 \times 224$$的RGB image。我们做的唯一的pre=processing就是在训练集里，将所有的图片都减去它们的平均值（element-wise）。这个image之后通过一系列的convolutional layer，在这些convolutional layer里，我们都使用的是$$3 \times 3$$的filters（这是能够获取一个pixel的上下左右信息的最小的filter size）。stride一直设定为1，而padding的设置使得每层convolutional layer的输入和输出的长宽是一样的，也就是说对于$$3 \times 3$$的filter，padding是1。而max-pooling在某些convolutional layer后会出现，整个CNN一共有五个max-pooling，都是使用的$$2 \times 2$$的大小，stride是2。

在Convolutional layers之后，接了三个fully connected layers：前两个都是有4096个通道，而第三个是1000个通道，因为ILSVRC的分类是1000类，所以用这个作为输出，正好每个通道表示每个类。最后一层是一个soft-max layer，将第三个fully connected layer的输出转换为表示为每个类的概率。

对于所有的隐藏层，我们都用ReLU作为activation function。我们所有网络都没有用到local response normalisation（LRN）的技术。

##### 5.2 Configurations
我们这篇文章里所使用的CNN的结构配置总结在Table1里，每列表示一个网络。再之后我们就使用A-E来表示这些网络。所有的网络设计都按照5.1里描述的那样，它们仅仅在深度上有区别：从11层（8层CNN，3层fc）到19层（16层CNN，3层fc）。convolutional layer的宽度也不大（也就是每个convolutional layer输出的channel数），从64开始，每次遇到max-pooling之后就增加一倍，直到到达512为止。

![TABLE1]({{ '/assets/images/VGG-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 1*

Table2总结了每个网络的参数的数量。尽管我们的网络深度很大，但我们网络的参数的数量并不比那些有着更大convolutional filters的更浅的网络要多。

![TABLE2]({{ '/assets/images/VGG-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 2*

##### 5.3 Discussion
我们的CNN结构和在ILSVRC-2012和ILSVRC-2013竞赛里表现最好的那些模型都很不一样。他们的网络的第一层用stride=4的$$11 \times 11$$的filter，或者用stride=2的$$7 \times 7$$的filter，然而我们这篇文章里用的是很小的stride=1的$$3 \times 3$$的filter。很容易看出来，两个$$3 \times 3$$的filter堆叠起来，就等价于一个$$5 \times 5$$的filter。三个$$3 \times 3$$的filter堆叠起来，就等价于一个$$7 \times 7$$的filter的感受野。所以说，我们通过堆叠三层$$3 \times 3$$的filter而不是一层$$7 \times 7$$的filter，得到了什么？首先，我们的层与层之间还有非线性层，而$$7 \times 7$$的只有一个非线性层，这能使得我们的决策函数更加复杂。其次，我们减少了参数的数量：假设我们的$$3 \times 3$$的层的输入和输出的channel都是$$C$$，从而对于一共是27$$C^2$$个参数；而同时，一个$$7 \times 7$$的filter的参数是49$$C^2$$。

注意到我们还使用了$$1 \times 1$$大小的convolutional layer，这样的层可以在不影响其他层卷积感受野的情况下，增加模型的非线性程度，也可以改变通道数，因为这个层之后也是有activation function的。

之前也有人使用过小的convolutional filters，但是它们的网络并不深，而且没有在ILSVRC数据集上进行测试。Goodfellow关于识别街景图片里的数字的文章使用了深的CNN（11层）并且说明深的网络效果更好。GoogLeNet在我们这篇文章的同时也研究了小的convolutional filters和深的CNN。



### 3. [ResNet: Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*

*CVPR 2016*

**AlexNet(2012) $$\rightarrow$$ ResNet(2016)**

#### 1. title
deep residual learning是文章提出的方法，而image recognition则是要解决的问题，同样也是简洁明了的标题。

#### 2. Authors
四个人都是大佬，这是在微软研究院所做的工作。

#### 3. Abstract
训练深的神经网络非常不容易。我们使用深的残差网络使得训练比之前容易很多。我们将结构设计为，让每一层去学习相对于这一层输入的残差（因为结构设计为$$output = x + f(x)$$，从而$$f(x) = output - x$$即是残差）。我们通过大量的实验来证明我们提出的residual networks很容易训练，对于增加的深度来说也是一样。在ImageNet数据集上我们使用了152层深度的网络，比VGG深了8倍。最后我们在ILSVRC 2015的竞赛里达到了3.57%的test error而获得了冠军。我们还在CIFAR-10数据集上演示了如何训练100和1000层的网络。

对于很多视觉的任务来说，深度是很重要的。我们仅仅是将分类器用到的feature换成deep residual network所学习到的feature，就使得COCO object detection的准确率提升了很多。我们还赢得了ImageNet detectino， ImageNet localization，COCO detection和COCO segmentation比赛的冠军。

#### 4. Introduction

深度CNN在image classification领域获得了很多突破。深度networks通过综合low/mid/high level的features，以end-to-end的方式实现。而这些不同level的features可以通过加深层数来使得它们非常的丰富。很多最近的工作表明网络的深度是很重要的，并且在ImageNet数据集上的任务做得好的网络都用了很深的结构，从16到30层不等。很多其他的视觉任务也从深度网络种获利。

但随之而来的出现了一个问题：学习一个好的深度网络就是简单的将层堆叠在一起就可以了吗？一个最显著的问题就是，在网络很深的时候，梯度容易爆炸或者消失，这就使得网络收敛变得很困难。而现有的方法就是在网络初始化和中间层normalization上做工作，这样能使得训练几十层的网络变得略微可行（比如说Batch normalization等），网络可以训练了，能够收敛了。

但虽然现在深的网络能够训练了，实际上网络的性能随着深度的增加而变差了，如fig 1所示。而这样一个现象并不是因为overfitting，这是因为此时训练误差也变大了（overfitting指的是训练误差小，而测试误差大）。

![Hard to Train]({{ '/assets/images/RESNET-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. 左侧是20层和56层普通的网络在CIFAR-10数据集上的training error，而右侧是test error。可以看到，深的网络的training error和test error都更大。*

>所以说用了一些方法之后，深的网络是收敛了，但是并没有训练到很好的结果，也就是属于欠拟合的状态。

上述的现象说明深的网络并不是那么好训练的。我们考虑一个浅的网络，以及它的深的改造（也就是多加一些层进去）。如果我们所添加的这些层所学习到的都是identity map，其它的层和浅的网络是一样的，那这个深的网络的效果是不会比浅的网络差的。但理论上是可以的，现实中却很难让深度网络学到这样的结果，是很难做到的。

>这篇文章提出的方法就是显式的构造一个结构，使得浅的网络的深的改造不会比浅层网络更差。

在这篇文章里，我们通过介绍deep residual learning framework来解决这个问题。我们并不是让堆叠的layers构成的网络来学习一个要学的mapping，而是显式的构造结构从而让每层学习residual mapping。对于我们要添加的层来说，输入为$$x$$，而我们希望能够学习到的mapping是$$H(x)$$，我们并不直接让网络来学习$$H(x)$$，而且让他来学习$$F(x)=H(x)-x$$，从而这样的话，我们希望学习到的mapping $$H(x)=F(x)+x$$。我们假设优化redisual mapping要比优化初始的mapping要容易。在最极端的情况，如果浅层网络就已经足够了，那实际上所添加的网络就是identity mapping，那这个时候residual mapping就是0。

>这里就是residual learning的核心的思想。见fig 2。

![Residual Block]({{ '/assets/images/RESNET-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. residual learning：一个模块.*

在模型设计里实现上述的residual结构，即$$F(x)+x$$可以通过shortcut来实现，如fig 2所示。

>但实际上shortcut的结构在很久以前就提出过了。

shortcut的结构很简单的就实现了我们的residual learning的结构，而且其就是简单的identity mapping，所以不增加参数也不增加计算复杂度，而且也不影响反向传播算法的使用。

我们在ImageNet上进行了实验证明了之前所说的过深的网络可能会造成的问题，并且也验证了我们提出的方法的有效性。1) 我们用了residual结构的很深的网络也能够很容易的被训练，而没有用residual结构的网络在网络加深的过程中训练error会增加。2) 我们的residual neural network可以享受由深度网络带来的性能上的提升，从而在ILSVRC比赛里打败了之前的那些网络。

相同的现象在CIFAR-10这个数据集上也得到了验证，从而说明我们的方法并不是仅仅局限于ImageNet这一个数据集的。

我们设计了一个152层的residual neural network在ILSVRC-2015的比赛里赢得了第一名。而且我们还将residual neural network用于ImageNet detection, ImageNet localization,
COCO detection, and COCO segmentation in ILSVRC & COCO 2015比赛均获得了第一名。这些证据足以说明我们的residual neural network这个框架是十分有效的。


>introductino是abstract的扩充，也是对文章整个过程的一个描述。这篇文章的introduction写的很标准。读者看完之后就能了解文章最核心的内容是什么。


#### 5. Related Work

##### 5.1 Residual Representations

##### 5.2 Shortcut Connections


#### 6. Deep Residual Learning

##### 6.1 Residual Learning

假设$$H(x)$$是我们想要通过几层堆叠的layers要拟合的mapping，$$x$$是这些堆叠的layers的输入。如果假设多层的layers的堆叠能够拟合任意复杂的mapping，那么它也可以拟合residual mapping，也就是$$H(x)-x$$（假设输入和输出的维度是一样的）。所以说，我们不让堆叠的layers去拟合$$H(x)$$，而是显式的让它们去拟合residual mapping，$$F(x)=H(x)-x$$。从而我们所想要拟合的原mapping就可以表示为$$H(x)=F(x)+x$$。虽然说堆叠的layers都有能力去学习$$H(x)$$和$$F(x)$$，但是对于neural networks来说，学习的难易程度是不一样的。

上述的过程是由我们在introduction里的内容而启发的。如果我们所添加的layers在学习之后就是identity mapping，那么添加了这些layers至少不应该使得training error变大。但实际上，很深的neural networks的效果是会变差的，也就是说training error是会变大的。发生这种现象表明想要深的网络训练好，是很困难的。而使用residual的结构，则会使得训练简单很多。


##### 6.2 Identity Mapping by Shortcuts

我们每过几层就采用residual learning的结构。而每个residual block就如fig 2所示。公式化来说，我们的residual block是这样的：

$$y = F(x, \{W_i\}) + x$$

$$x,y$$分别是这个block的输入和输出，而$$F$$代表residual mapping，$$W_i$$是参数。

上述公式里的shortcut既没有增加新的参数，也没有增加计算复杂度。这不仅在实践中会高效很多，也使得我们在实验中进行对比的时候，能控制其他变量不变，只是将shortcut的部分去掉，从而对比很公平。

在上述公式里我们可以看到，最后的结果$$y$$是residual mapping $$F$$和$$x$$的element-wise的addition，从而$$x$$和$$F$$的维度得是一样的。如果它们的维度不一样，比如说input和output的channels发生了变化，我们就引入linear projection $$W_s$$来解决问题：

$$y = F(x, \{W_i\}) + W_sx$$

residual mapping $$F$$的设计是多样的。我们这篇文章里$$F$$的设计用了2层或者3层 layers，而更多的层也是可以的。但是如果只有一层，那么其实residual block就变成了linear layer：$$y = Wx + x$$，这样是不行的。

而且上述的这些论述不仅仅是对于layers是MLP时候，layers是CNN的时候，同样也是适用的。


##### 6.3 Network Architectures

我们测试了多种不同的plain/residual neural networks，对于我们的结论并没有什么不一样。用于ImageNet的plain和residual network结构分别如下：

* Plain Network
我们的plain network的结构如fig 3中间那幅图所示。

* Residual Network
我们在plain network的基础上加入shortcut connections，从而将网络改成了residual neural networks。对于输入和输出维度相同的residual blocks，shortcut connections就直接是elementwise addition。而对于不同的情况，比如说fig 3里的虚线，可以用zero padding的方法，也可以用$$ 1 \times 1$$的convolution来使得feature map的通道数相符。

![network]({{ '/assets/images/RESNET-3.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 3. ImageNet的示例结构。左侧：VGG-19模型。中间：34层的plain network。右边：34层的residual network。*

##### 6.4 Implementation

输入的图片的短边随机从256到480的范围内采样，长边则是按比例计算。之后再从已经scale的图片里随机裁剪$$224 \times 224$$大小的图片，之后再随机的选择是否需要水平翻转。再之后将图片的平均值减去（elementwise）。而且在每个convolution之后，activation之前还采用了batch normalization。使用SGD进行优化，batch_size选取为256。learning rate从0.1开始并且在error到达平台期后手动减小至1/10。使用了weight decay-0.0001和momentum=0.9。并没有使用dropout。

>dropout常见于MLP或者fully connected layer里，convolution layer一般不用。

在测试的时候，我们从测试图片里随即裁剪10张进行测试。而且还对scale不同的image进行了测试再将结果取平均。

>这些都是刷榜的技巧，使得结果更好。


#### 7 Experiments

#### 7.1 ImageNet Classification

我们在ImageNet 2012 classification dataset上验证我们的方法，这个数据集有1000个类。模型在128万张图片上进行训练，并在50000张图片上进行验证。最后在100000张测试图片上测试最后的结果并报告。我们验证了top-1和top-5的error rate。

* Plain networks
对于plain networks，我们验证了18层和34层的效果。34层的plain network在fig 3中间进行了描述。而18层的plain network是相似的。从Table 1能看到模型的细节。

![table]({{ '/assets/images/RESNET-4.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 1. ImageNet上所使用的plain networks的结构。conv3_1，conv4_1，conv5_1通过stride=2实施了降采样。*

fig 4显示了18层和34层的plain network和residual network在training error和validation error上的区别。我们再一次发现，34层的plain network还不如18层的好，这说明34层的plain network欠拟合。

![plain]({{ '/assets/images/RESNET-5.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 4. 左边：plain network。右边：residual network。*

我们认为上述的这种plain network欠拟合的问题，并不是因为梯度消失导致的。因为我们的网络里都加入了batch normalization。我们猜测可能是因为深的neural networks有着指数降低的收敛率，从而使得training error的降低很困难。但这个问题的原因还需要未来的工作来解决。

* Residual Networks
从fig 4我们可以看到residual结构很好地解决了深度neural network欠拟合的问题，因为对于34层的ResNet，其training error和validation error都要比18层的ResNet低，说明他确实学习到了更优越的性能并做到了generalization。

* Identity vs. Projection Shortcuts
有不同的shortcut的方法。(1) 只用zero-padding的方法来对于维度不同的情况处理 (2) 只使用projection，也就是$$1 \times 1$$的convolution来对维度不同的情况处理 (3) 只使用projection来处理，但对于维度相同的情况仍然也加上。

>注意，在输入和输出维度不同的residual block里，虽然使用projection能够使得通道数很容易的匹配，但是实际上输出的高宽也是输入的一半，所以可以使用stride=2的$$1 \times 1$$的convolution来实现。


* Deeper Bottleneck Architectures
我们对于更加深的网络，比如说ResNet-50，101，152都采用了一种bottleneck的设计，这样可以减少计算复杂度，使得训练更快一些。我们可以在fig 5看到设计的结构。对于高通道数的输入，先用$$1 \times 1$$的convolution将通道数降低，再做普通的convolution，最后再用$$1 \times 1$$的convolution将通道数升回去。


![bottleneck]({{ '/assets/images/RESNET-6.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 5. 左边：普通的residual block。右边：加了bottleneck结构的residual block。*


##### 7.2 CIFAR-10 and Analysis


##### 7.3 Object Detection on PASCAL and MS COCO


>为什么ResNet训练的效果很好？实际上可能是因为它解决了梯度消失的问题。对于plain network来说：

>$$ \frac{df(g(x))}{dx} = \frac{df(g(x))}{dg(x)} \frac{dg(x)}{dx} $$

>梯度都是很小的，所以说这两个乘积项本身都很小，乘在一起就更小了。而网络很深的时候，误差反向传播到很前面的网络的时候，就容易发生梯度消失的问题。

>对于residual network来说，情况就不一样了：

>$$ \frac{df(g(x)) + g(x)}{dx} = \frac{df(g(x))}{dg(x)} \frac{dg(x)}{dx} + \frac{dg(x)}{dx} $$

>两项梯度相加，就很有可能使得梯度不是那么小了，从而就可以训练的动了。

#### 8. Conclusion

这篇文章是没有结论的。因为文章内所要说的结果太多了，超出了会议规定的文章最大页数。这种写法是不建议的，最好还是要有conclusion，使得文章具有完整性。


## Transformer-related architectures

### MAE: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)

*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick*

*Arxiv 2021*

**Transformer $$\rightarrow$$ Bert(using transformer framework, but with self-supervised learning), Transformer $$\rightarrow$$ ViT(using transformer idea in CV), ViT $$\rightarrow$$ MAE (just like Bert' relation to Transformer)**

**Transformer $$\rightarrow$$ Bert $$\rightarrow$$ MAE $$\leftarrow$$ ViT $$\leftarrow$$ Transformer**

#### 1. Title

"Scalable" are usually used when your model is very big, and "efficient" are usually used when your model is quite quick. These are two pupolar words in a title. "Vision learners" is a more broad word rather than classifier or other specific models. The "auto" in "Autoencoders" means that the input $$x$$ and label $$y$$ comes from the same data. "Auto" models are a kind of general models. In NLP, most works are actually "auto", so they usually just neglect this word. But in CV, these kind of works are not very much, because the label for images seldom comes from images themselves, thus "Autoencoders" in this title is essential. It means that in this work, the labels for images come from images themselves, distinguished from other works.

The whole sentance, "a" are "b" "c", where "a" and "c" are nouns and "b" are adjectives, are a popular naming pattern recently. This is a very useful sentance, since it contains the conclusion in the title, and it's objective.

#### 2. Abstract

The idea is to mask random patches of an image and let the model to reconstruct the masked pixels. This idea is quite similar to Bert, since in Bert, they let the model to learn masked words according to other words. There are two core ideas of this model: firstly the whole model is a asymmetric encoder-decoder architecture, since the masked patches will not be encoded and the decoder will reconstruct the whole input image from non-masked representations and masked tokens. Secondly, masking high portion of the image (e.g., 75%) will yield a nontrivial sefl-supervised learning task. Because only masking few pixels, we can just using interpolation to archive pretty good result. These two ideas make training a large model efficiently becomes possible: because large portion will be masked and not encoded, and this problem is hard enough to implement large models to learn. Since this work is a counterpart to Bert, it actually used as some pretrained technic for transfer leaning.


#### 3. Figures

Have a look at figure1. Figure1 is usually at the upper right part of a CV paper and should be the most important figure of this paper to generally explain the idea of this paper. Figure1 describes the whole process of the MAE model. The input image will be cut into patches, and patches are randomly masked. Only non-masked patches are encoded by the encoder to get the representations of each non-masked patches. Then the representations are streched, and arranged according to the positions of there non-masked patches, and masked patches are also included, using only the position information. And finally the decoder will reconstruct the pixel information based on this streched representations. Note that in the figure, the box of encoder is bigger than the one of decoder, indicating that the encoder is more complex and is the key structure in this model.

![MAE]({{ '/assets/images/MAE-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. MAE architecture.*

Now turn to figure2 and figure3. Figure2 shows the reconstruction ability of MAE model on the ImageNet validation set, while figure3 shows the same results on COCO dataset. 

Figure4 shows that for the same image, masking different portion of the original image will result in different reconstructions.

These three figures (2, 3, 4) have very good results on reconstruction based on very large portion of original images been masked. I think one reason could be that the model have learned the common information of one category, and once it encountered an image of this category with large portion occluded, it can use the information of other images of the same category to assist the reconstruction of this image.

#### 4. Discussion and Conclusion

The authors think this model is simple and can be scaled well. Being simple is because it's based on Vision Transformer, and being scaled-well is because it does not encode masked patches and saves lots of calculation for encoding.

The authors also argue that supervised learning still dominates the CV community, different than the NLP groups. And this model has an unsupervised setting, and could be an important future direction for CV.

They also mention that we need to note the differences between CV and NLP tasks. Images and languages are signals of different nature and this intrinsic difference must be addressed carefully. A word is itself a semantic part, and different words form sentences. But for images, a pixel or a patch is not a semantic part of an image. Even having these differences, MAE (based on transformer architecture) could learn some good hidden representations. This part needs future works.


#### 5. Introduction

1. Tell a story which is that deep learning has made great progress in CV, but still requires plenty of labeled data. And in NLP, self-supervised techniques are very popular, e.g., BERT, GPT. etc. Masked autoencoders are also not a novel thing in CV, for example, denoising autoencoders (decades before) involve adding noise to the image and then denoising the noise, and by doing this can let the model learn the useful representations of the images. Some works recently have used BERT structure into CV, but still not having very good results. So the question is: **what makes masked autoencoding different between vision and languages?** The authors give three answers to this question. 1). The architecture is different. For NLP, popular architecture is Transformer. And once you mask a word, you can always know the position information of the masked word easily. But for CV, the popular architecture is CNN. CNN uses convolution calculation sliding over the whole feature maps. Thus you mask one area of the image, and you can hardly track the positions of the masked area after several layers of convolution. But this problem has been issued by the introduction of Vision Transformer (ViT). But actually, why transformer needs position information is that attention mechanism in transformer does not have position information. But CNN naturaly contains position information because the convolution window slides over the feature maps, and record the results of each area into specific location of the resulting next layer feature maps. 2). Information density is different. For language, a word is a semantic identity, and removing words, even few, can make one sentences vague. But there are much information redundacy within an image. Only removing a patch of an image, you can reconstruct this part by interpolating using neighborhood pixels. Because of this reason, masked autoencoding is not very useful in CV, because it does not learn useful semantic representations. In this paper, the authors just simply mask large portion of the images to make the task challenging and thus the leaRned representations in this way is information rich enough, not just interpolating but learning global information instead. 3). The autoencoder's decoder. The decoder for language models usually output a word, which is a very high-level semantic information. So the decoder in NLP tasks are quite simple, just MLP could be enough. But for CV tasks, the output of the decoder of the autoencoder of the CV task are normly pixels, i.e., the reconstruction of the original image, which is a very low-level semantic representation. Thus if the CV task is difficult, for example, the semantic segmentation, the decoder should be more complex rather than the NLP models. Note that for simple CV tasks, such as classification, decoders are also very simple, such as in ResNet, the decoder is just several layers of MLP.

2. Based on the ideas above, they propose MAE. MAE uses a asymmetric architecture of encoder and decoder to reconstruct images with large portion been masked. Being asymmetric means that the input information of the encoder and the decoder is different, in their setting, encoder only encodes non-masked patches, while decoder also has the position information of masked patches. This setting could make the computation of encoder be much less, and more layers can be added and model can be easily scaled. 

3. They finally show that MAE trained on only ImageNet-1K dataset can have very good performance as pre-trained model for downstream tasks such as object detection, instance segmentation, semantic segmentation, etc. So MAE in CV serves as some similar role as the BERT in NLP.


#### 6. Related works

1. Masked language models, including BERT, GPT, etc.

2. Autoencoding models in CV community. 

3. Masked image encoding in CV community. iGPT, BEiT are very closed papers, but they do not explan them clearly. Recommend that for very closely-related papers, you need to explain them and show the differences between your paper and their's.

4. Self-supervised learning in CV community.


#### 7. Approach




#### The writing style of this paper

1. The introduction part is a little long, partly because this paper uses lots of figures. Fruitful figures in CV paper is a good thing. The writing style of the introduction is not just an extension of the abstract (like GAN paper), but raising the topic into a more high level, and raise questions. This paper aims to solve this kind of question. It's a very good writing style. This can explain the necessity of this paper and give the basic idea of how the authors understand this problem. This could make the paper insightful, rather than just a technic report explaining the model details (like AlexNet paper).


### Transformer: [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*

*NIPS 2017*


---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

