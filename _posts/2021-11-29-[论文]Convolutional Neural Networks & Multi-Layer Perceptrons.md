---
layout: post
comments: false
title: "[论文]Convolutional Neural Networks & Multi-Layer Perceptrons."
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of convolutional neural networks and multi-layer perceptrons.


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


### 3. [GoogleNet: Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)
*Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich*

*CVPR 2015*


### 4. [ResNet: Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
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

![Hard to Train]({{ '/assets/images/ResNet-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. 左侧是20层和56层普通的网络在CIFAR-10数据集上的training error，而右侧是test error。可以看到，深的网络的training error和test error都更大。*

>所以说用了一些方法之后，深的网络是收敛了，但是并没有训练到很好的结果，也就是属于欠拟合的状态。

上述的现象说明深的网络并不是那么好训练的。我们考虑一个浅的网络，以及它的深的改造（也就是多加一些层进去）。如果我们所添加的这些层所学习到的都是identity map，其它的层和浅的网络是一样的，那这个深的网络的效果是不会比浅的网络差的。但理论上是可以的，现实中却很难让深度网络学到这样的结果，是很难做到的。

>这篇文章提出的方法就是显式的构造一个结构，使得浅的网络的深的改造不会比浅层网络更差。

在这篇文章里，我们通过介绍deep residual learning framework来解决这个问题。






#### 7. Conclusion

这篇文章是没有结论的。因为文章内所要说的结果太多了，超出了会议规定的文章最大页数。这种写法是不建议的，最好还是要有conclusion，使得文章具有完整性。



The biggest contribution of this paper is to offer a model strucuture that makes the training of very deep neural networks possible. The model in this paper won the ILSVRC 2015 competition classification task.

For naive convolutional neural networks, using very deep architecture will not cause over-fitting easily, but cause under-fitting, i.e., the model is very hard to train. Because not only the testing error of deep models are higher, the training error is also higher. The phenomenon is shown in the below figures. 


![Hard to Train]({{ '/assets/images/ResNet-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena shows on ImageNet.*

![Main result]({{ '/assets/images/ResNet-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. Training on ImageNet. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layers. Right: ResNets of 18 and 34 layers. In this plot, the residual networks have no extra parameter compared to their plain counterparts.*

When the network becomes deep, the problem of gradients vanishing/exploring will be remarkable. Former methods to alleviate this problem include setting good initiation parameters, using batch normalization. These methods makes training deep networks become possible, but actually the performance becomes worse. In principle, this should not be the case. Because if we have a shallow network, and then we add several more layers to create its deep counterpart. The deep network should be at least good as the shalow one, because it can let the added layers be just identity mapping. But these kind of parameters are very hard for deep networks to learn, thus the above phenomenon exists.

In this paper, having the above ideas in mind, the authors create a model that explictly having structures to represent this **identity mapping**. In this paper, they use a shortcut directly add $$x$$ from the input to the output of two layers. If the groundtruth is $$H(x)$$, they actually want the two layers to learn the "residual", i.e., $$f(x)=H(x)-x$$, where $$H(x)$$ is the desired output and $$x$$ is the input, with respect to this two layer structure. The shorcut here resemble the identity mapping.
zero-padding or $$1 \times 1$$ convolution are used to solve the problem that $$f(x)$$ and $$x$$ has different width $$&$$ length and channels.

![Residual Block]({{ '/assets/images/ResNet-3.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 3. Residual learning: a building block.*

Some implementation details are different from the AlexNet, for example, the shorter side of input image is firstly randomly scaled to a number in \[256,480] and then a $$224 \times 224$$ patch is sampled from the original input. There are no fully-connected layers except for the last softmax classifier, thus drop out is also not implemented.

Another structure design in this paper is the bottleneck design. This design helps not increase parameter numbers too much while using very deep architectures (bigger than 50). Use the below figure as an example. The input has size $$width \times length \times 256$$. Firstly the input is compressed into $$width \times length \times 64$$ by using $$1 \times 1$$ convolution, then normal convolutional layers are deployed. In the end, $$1 \times 1$$ convolution is used again to raise the output channel to 256. This process will make the computation complexity of this structure be similar to the left one, but the model depth will be much deeper. Also, due to the existence of the shortcut connection, the information loss in this process only happens in the residual computation, and the information in the input is not influenced.

![Bottleneck]({{ '/assets/images/ResNet-4.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 4. Left: normal residual block. Right: Bottleneck residual block.*

Suppose $$g(x)$$ is a neural network. Adding more layers to this net will make the model function become $$f(g(x))$$. $$\frac{d f(g(x))}{d x} = \frac{d f(g(x))}{d g(x)} \frac{d g(x)}{d x}$$ is the gradients of the new model and $$\frac{d g(x)}{d x}$$ is the original one's. Gradients are always be quite small, thus the multplication will make the gradients of the new model be much smaller than the original one's, thus the training of deep model is very hard.

But if we use the ResNet structure in this paper, then our new deep counterpart of $$g(x)$$ becomes $$f(g(x))+g(x)$$, and the gradient becomes $$\frac{d f(g(x))}{d g(x)} \frac{d g(x)}{d x} + \frac{d g(x)}{d x}$$, which is comparable to the original one's.

The key idea is: **Always make the gradients be large enough, and then your model can be trained well!**


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


## Natural Language Processing

### Transformer: [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*

*NIPS 2017*



## Generative Models

### GAN: [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*

*NIPS 2014*

#### 1. Abstract
1. The abstract of this paper is very standard and unique. This abstract generally introduces the main concept of this paper, i.e., GAN, rather than introducing the problem and existing methods. This abstract is kind of like the Wikipedia introduction of some concept. If you are very confident that your paper is very novel and proposes a very useful idea, model or concept that can be recorded in this area, you can use this kind of writing style of abstract. This is very clear to the community, but not very clear to the people that do not know any about this area.

#### 2. Introduction
2. This paper thinks that besides deep learning architectures, deep learning will learn the representations of data probability distributions over all kinds of data, including natural images, audio waveforms containing speech, symbols in natural language processing, etc. This idea is always held by Yoshua Bengio and his group.

3. Discriminative models in deep learning have already invovled into various kinds and solved many kinds of problems. But generative models, instead, still have lots of part remained mysterious. This due to the fact that in order to learn data distributions underlying generative models, you need many approximation methods to approximate the distribution in order to make calculation work. But this process will make the distribution unaccurate and even not working. So in this paper, they do not try to approximate the distributions, they use deep learning models to do this job.

4. Generative adversarial models will contain two parts: the generative model and the discriminator model. The generative model aims to generate data that the discriminator model can not distinguish apart from true training data, and the discriminator model aims to distinguish between them. The final goal is to let the generative model to generative plausible data that the distriminator model can not distinguish.

5. In this paper, they use MLP as models for the generative and discriminator models, and in this case, this generative adversarial model is called generative adversarial net (GAN). The input of the generative model is random noise (usually a Guassion noise), and they want the generative model to map this random noise distribution to any distribution we want to fitting, i.e., the training data distribution.


#### 3. Related work

1. There are two kinds of former works on deep generative models. The first ones concentrated on building a probability distribution function, and these models are trained on maximizing the log likelihood, such as the deep Boltzmann machine. The biggest difficulty is that the calculation of sampling this distribution is hard, especially when the dimension is high. The other ones are called generative machines. They do not explicity construct the distribution, and they learn a model to approximate this distribution. There are intrinsic differences between these two kinds of methods. The first ones acutally learn the distribution, though using some kind of approximation method to make the calculation feasible, but in the end, you actually get a distribution, you can calculate the mean, variance and all kind of properties of this distirbution. But the other ones do not construct the distribution, and only learn a model to approximate this distribution. So in the end, we do not know what this distribution looks like. 

2. Variational Autoencoder (VAE) actually has similar ideas to this paper. And using a distriminator model to assist the generative model is also not novel, such as Noise-contrastive Estimation (NCE). 


#### 4. Adversarial nets
1. Generator wants to learn the distribution of the input training data $$x$$, $$p_g$$. We give an example of GAN. Suppose there is a video game and it can generate images of the game, and now we want to learn a generator to generate the images of the game. Suppose that our display resolution is 4K, then each time we need to generate an image vector of length 4k. Each pixel could be considered as a random variable, thus this vector can be considered as a multi-dimensional ramdon variable of length 4k. We know that this vector is controled by the underlying game code, and this code is actually the underlying $$p_g$$ for this vector. Now how to let the generator to generate data? We define a prior on the input noise variable $$p_z(z)$$, this $$z$$ could be a 100 dimensional Guassion distribution with mean 0 and variable matrix I. The generator aims to map $$z$$ onto $$x$$, the generator model can be formed as $$G(z, \theta_g)$$. Return to our game example. In order to generate game images, one way is that we can conversly compile the game code, and know the underlying game code. In this way, we can acutally know how the game images are generated. This method can be considered similar to the methods described in the related work that aim to construct the underlying distribution. Another way is that we neglect the underlying code, we guess that this game is not very complicated, thus maybe there are only 100 variables underlying are used to control the generation of images. Thus we contructed a known distribution of dimension 100 $$z$$, and due to the fact that MLP is able to approximate any functions, we let this MLP to map the input $$z$$ into the image space $$x$$. 

2. The discriminator $$D(x, \theta_d)$$ is also an MLP, and its output is a scalar value, for distinguishing between the true data and generated data. Thus actually $$D$$ is a two-label classifier. We know where our input data is from (true or generated), thus we can give them labels. 

3. The training process involves training D and G simultaneously:

$$\min_{G}\max_{D} V(D,G) = E_{x \sim p_{data}}\left[log D(x)\right] + E_{z \sim p_z(z)}\left[log(1-D(G(z)))\right]$$

This is a two-player minimax game. When the G and D reach a stable state, they are actually arrive at a Nash equilibrium.

4. Look at figure1. This example is simple. The input noise distribution of $$z$$ is a uniform distribution (the lower line of $$z$$ has equal intervals), and our $$x$$ is a Guassian distribution (the black dotted line). The arrows between the line of $$x$$ and $$z$$ is the mapping, i.e., generator. From plot (a) in figure1, we see that, at first, the mapping maps $$z$$ to the behind part of $$x$$, so the output distribution of this mapping is the green line in the plot, and the blue line is the discriminator. The next step, we update the discriminator, we can see that, the discriminator choose the margin in between of the mean of the true distribution and the generator output distribution, as shown in plot (b). Then we update the generator, we can see that the generator output distribution will move closer to the true distribution, in plot(c). And then we update discriminator, update generator, ..., and finally ,we get plot(d), in that the generaotr output will be the same to the true distribution and the discriminator will show a horizontal line indicating that it can not distinguish between true and generated data. 

![GAN]({{ '/assets/images/GAN-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. An example of training process of a GAN.*

5. Each training iteration of the training algorithm of GAN involves two steps. In the first step, there is a for loop, loops over $$m$$ times. And in each loop, we get $$m$$ true data and $$m$$ generated data from the generator, and then calculate the gradient of the minimax loss defined above to update the discriminator. In the second step, we sample another $$m$$ samples from the generator to calculate the gradient of the minimax loss with respect to the generator for updating it. The for loop iteration time $$k$$ in the first step, is a hyperparameter of this algorithm. In each training iteration, we need the generator and distriminator be at the same levle, i.e., the performance of one should be be much better than the other. Only in this case, we can make the training trackable. The decision of whether a GAN is trained well is also a difficult question, i.e., the iteration time of the training algorithm. This still remains a hot area and unsolved.

6. One training tip is that, since when the discriminator trains well, the $$log(1-D(G(z)))$$ will be close to 0, thus the gradient will not be applicable, instead of minimizing $$log(1-D(G(z)))$$, we maxmizing $$log(D(G(z)))$$.


#### 4. Theoretical Results

1. There is a global optimum for the generator, $$p_g = p_{data}$$. Firstly, we see a lemma firstly. **Lemma**: if the generator is fixed, then the optimal discriminator will be

$$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$

i.e., the error probability (the training criterion of discriminator) of the distriminator will be the smallest. **Theorem**: Setting $$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$ as in Lemma in the equation $$\min_{G}\max_{D} V(D,G) = E_{x \sim p_{data}}\left[log D(x)\right] + E_{z \sim p_z(z)}\left[log(1-D(G(z)))\right]$$, we can get $$C(G) = E_{x \sim p_{data}}\left[log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + E_{x \sim p_g}\left[log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right] $$. Then $$C(G)$$ get its minimum when $$p_g = p_{data}$$.

2. The algorithm described above is able to train the discriminator and the generator, i.e., the training algorithm is convergent. 


#### 5. Experiments
The experiments in this paper is not good enough and quite simple.


#### 6. Conclusion

GAN is actually an unsupervised learning setting, but it leverages supervised learning framework by using the label of true or generated data for training. This idea insights the future self-supervised learning frameworks.


#### The conclusion of writing style of this paper

This paper proposes a very novel idea and model, thus it elaborates the details of design and ideas behind GAN very clearly. The authors are very ambitious and are confident that this work will be recorded in this area. So the whole writing style is kind of like Wikipedia, i.e., the very detailed description of the proposed model, with a little mention of existing works and comparison with other works. Also, experiments are few.

If you are confident that your work is novel and very important, you can use this kind of writing style. Otherwise, you need to describe clearly the existing works and your contribution to this problem.









---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

