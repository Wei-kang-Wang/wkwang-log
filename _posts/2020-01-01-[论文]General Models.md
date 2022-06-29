---
layout: post
comments: false
title: "[论文]General Models."
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

**1. title**

标题很简洁明了，用什么方法、模型（deep convolutional neural network）解决了什么问题（ImageNet classification）。

**2. authors**

Hinton大佬的文章，那肯定是值得看的。

**3. abstract**

我们设计了一个很大、很深的cnn来解决ImageNet LSVRC-2010的120万张图片分为1000个类的任务。在测试数据上，top1和top5的错误率都远远比其它的方法要好。

>这种写法其实是很少见的，第一句话说我做了什么，第二句话就直接展示效果并且说十分好。

这篇文章里的神经网络是很大的，有6千万个参数以及650000个神经元，而且有五个卷积层，三个全连接层，最后还有个长为1000的softmax。为了使得训练更快，我们设计了方法，使得神经元是位于non-saturated区域的，并且用了高效的GPU设计方法来训练。我们还用了dropout来使得全连接层不至于过拟合。我们最后将训练后的模型再应用到ILSVRC-2012的比赛里，达到了top-5比第二名低很多的错误率。

>这个摘要主要就说了我们用了什么方法做了什么事情，介绍了一下方法、模型具体的设计细节，之后就说了在实验上比其他方法远远要好。其实不太像是摘要，反而像是技术报告（technic report）的写作方式。但毕竟是大佬的文章，结果就是要比其他方法好的非常多。

**4. Introduction**

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

**5. Dataset**

>因为这篇文章是在ImageNet这个数据集上解决其classification的任务，所以得介绍一下这个数据集。

数据集介绍的部分就不说了，主要介绍了数据集的内容，以及ILSVRC这个竞赛的内容。

（但这部分最后一段很重要）ImageNet实际上每张图片的分辨率是不一样的，也就是说图片是没有裁剪好的。这篇文章里的做法简单粗暴，直接将短边rescale到256，而长边等比例缩小之后，直接在rescale的图片中心区域裁剪出256$$\times$$256大小的部分。除了再将每个通道的平均值减掉这个操作以外，并没有任何别的pre-processing操作，整个模型是在raw-RGB图片上操作的。

>当时大多数的方法都需要从原始输入图片里抽取特征再用模型进行处理，比如说sift特征等等。但是这篇文章不需要抽取任何特征，实际上这是很大的一个进步，但这篇文章并没有把其作为一个卖点。这种方法实际上是end-to-end，也就是说原始数据直接输入模型，没有任何特征提取，结果直接输出。文章没有重点提到这个实际上是历史的局限性，之后的研究表明end-to-end是深度神经网络一个很大的卖点，应该要予以强调。

**6. Architecture**

**6.1 ReLU nonlinearity**

运用到神经网络里，饱和的那些非线性函数$$tanh(x)$$、$$sigmoid(x)$$等实际上要比非饱和的那些非线性函数比如说$$ReLU(x)$$使得训练慢很多。从fig 1就能看出，虚线是使用tanh的，实
是使用ReLU的，横轴是epoch数，纵轴是training error，可以看出来效果差别挺大。
  
![RELU]({{ '/assets/images/ALEXNET-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. 我们这里用CIFAR-10数据集，对于一个4层的CNN进行了测试，用的是tanh和ReLU两种不同的非线性函数。而对于每种情况，我们设置的learning rate不一样，learning rate设置成为
每种情况下，都选择使该种情况下训练最快的值。我们可以看到，ReLU使得训练快了很多。*

**6.2 Training on Multiple GPUs**

介绍的如何用多个GPU训练的，主要都是技术、工程上的细节。和ML、CV关系不大。

**6.3 Local Response Normalization**

实际上这个东西也不重要，后续也没有什么人用。
  
>提到了一点，ReLU不是饱和型的非线性函数，所以在输入ReLU之前实际上并不需要对输入做什么操作（比如说将输入集中到非饱和区域等）。

**6.4 Overlapping Pooling**

将传统的不overlap的pooling改成了overlap的pooling，会使得效果好很多。

**6.5 Overall Architecture**

用了五个卷积层，三个全连接层，再加上一个1000-way的softmax，如fig 2所示：
![Model Structure]({{ '/assets/images/ALEXNET-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. 模型的结构*
  
>输入是一个很扁的很宽的图片$$224 \times 224 \times 3$$，然后通过网络将高宽不断变小，而深度不断增加。随着网络的增加，慢慢将空间信息压缩，即高宽逐渐减小，到了最后一个卷积层
已经变成了$$13 \times 13$$，也就是说认为这里一个像素已经可以代表前面很大一块的像素。而随着深度的增加，通道数也在逐渐的增加，可以认为每个通道是在表示一种特定的特征，比如说19个通道，我可以认为它能够识别输入的图片的192种不同的特征，比如说某个通道用来识别边，某个通道用来识别圆等等。所以说，空间信息在逐渐压缩，而语义信息在逐渐增加（通道数就是语义信息）。而最后，到了全连接层。所以说，输入的$$224 \times 224 \times 3$$的图片，最后通过网络变成了一个$$4096$$长度的向量，最后再用一个线性分类器做分类。实际上，这个$$4096$$的向量具有很好的语义信息，所以效果很好，也可以从后面的fig 3看出。
>整个机器学习，都可以看成知识压缩的过程，原始的数据不管是图片、文字还是视频，最后都通过一个向量来表示，而这个向量所含有的语义信息，可以让机器进行识别。从而可以在上面做各种各样的事情。这也是神经网络精髓之所在。


**7. Reducing Overfitting**

**7.1 Data Augmentation**

输入是$$256 \times 256 \times 3$$大小的图片，而文章会随机裁剪$$224 \times 224 \times 3$$大小的部分作为输入，引入了随机性，增强了模型的generalization的能力。还对RGB通道做了PCA的操作。

**7.2 Dropout**

将很多个模型融合在一起找到效果最好的那个，这个方法是很常见的。但是对于deep neural networks来说，本来训练就很贵了，这么做是不现实的。这篇文章用了dropout的操作，也就是说随机的将模型里某些输出直接设置为0。这个操作可以使得我们每次都会得到不一样的模型，但这些模型的权重实际上是共享的，从而最后等价于很多个模型做融合。
  
>但后来的工作表明dropout实际上并不是在做模型融合，更像是正则项。之后的工作里说明dropout实际上等价于一个L2的正则项。

**8. Details of Training**

介绍了用SGD方法来训练模型。使用了weight decay，也使用了momentum。

还介绍了一下模型参数的初始化操作。

最后介绍了一下learning rate的设置方法。

**9. Results**

介绍了在ImageNet上的classification任务和其他方法的对比，有ILSVRC-2010和ILSVRC-2012两个比赛。

**9.1 Qualitative Evaluations**

我们从fig 3可以看到，模型的效果很不错。而从fig 3的右侧发现，feature向量相近的那些输入图片，内容确实是相近的，这其实是模型之所以效果好的很重要的点，这说明模型确实学习到了有效的语义信息，对于语义信息相似的图像，能生成距离相近的feature向量，从而在feature空间中能够很容易的分类。
![test]({{ '/assets/images/ALEXNET-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. 左图显示了在Iamgenet数据集里找的一些图片，以及它们通过模型输出的top-5的预测结果。而右图的第一列是随机选取的图片，对于每一行来说，第一列的图片输入模型后可以得到最后的4096长的feature向量，而我们寻找到那些图片，其feature向量与此向量的距离最近。*
  
>神经网络的可解释性一直是一个研究热点，而此处也提及到了一部分。

**10. conclusion**

>这篇文章是没有结论的，只有discussion。指明了未来可能还需要做些什么事情。但一篇文章的结论conclusion通常来说是和abstract的对应，所以说没有conclusion是很少见的，也是不推荐的。

我们这篇文章的结果表明深的、大的CNN对于很难的任务是效果很好的。如果我们的模型去掉一层，那么结果会变差，这说明深度也是有必要的。

>其实这里论证不太严谨，因为去掉一层结果会变差也可能是其它的参数没设置好，实际上去掉一层，再改改参数，还是能达到一样的效果的。但是歪打正着，这个结论是正确的，深度确实是很必要的。

我们这篇文章所使用的方法还没有用到任何的unsupervised pre-training的方法，并没有预先transform数据，而且我们所解决的是一个supervised learning的问题。

>实际上机器学习在长时间的范围内研究的都是unsupervised learning，Hinton还有LeCun等人一直都认为unsupervised learning才是主流方向。但这篇文章的出现，使得supervised learning火了起来，一直到最近Bert还有GAN的出现，才让unsupervised learning重新回到人们关注的视野里。实际上在这篇文章之前，neural network一直做的是unsupervised learning的工作，之所以不做supervised learning是因为比不过SVM，所以只能做一些SVM做不了的工作。但这篇文章的出现说明深的、大的neural network可以在supervised learning打赢其它方法了，所以引起了轰动和热潮。



### 2. [VGGNet: Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
*Karen Simonyan, Andrew Zisserman*

*ICLR 2015*

**1. Title**

和AlexNet、ResNet的标题一样的风格，都是直接说明什么模型（方法）解决什么问题。

**2. Authors**

Andrew Zisserman是Oxford VGG实验室的leader，很强的组。

**3. Abstract**

这篇工作我们探究了CNN的深度对于大规模image recognition任务精度的影响。我们的主要贡献是十分详尽的研究了网络深度的作用，而我们的网络使用的是很小的convolutional filters（$$3 \times 3$$），我们的结果显示深度到16-19层的时候，任务的精度能有很大的提升。这些结果是我们参加ImageNet Challenge 2014的模型的基础，我们的模型赢得了localisation的第一名，以及classification的第二名。我们同时也说明我们的方法对于其它的数据集仍然有很好的效果。我们将两个效果最好的CNN模型公布了，希望能够帮助之后在CVi领域使用深度视觉特征的研究。

**4. Introduction**

CNN最近在大规模的image和video recognition任务中获得了巨大的胜利，也是因为大规模的公开的数据集比如说ImageNet，以及高性能的计算资源比如说GPU，才使得这种成功成为可能。在ILSVRC-2014中，deep visual recognition architecture已经展露了头角，在前几年的比赛里，ILSVRC-2011的冠军使用的是high-dimensional shallow feature encodings，ILSVRC-2012年的冠军使用的是deep CNN（也就是AlexNet）。

随着CNN在CV领域逐渐变得常见，有很多人尝试改进AlexNet以获得更高的精度。比如说，ILSVRC-2013年的冠军就是将AlexNet改进为第一层使用更小的convolutional filter以及更小的stride。另一条改进AlexNet效果的路则是利用不同scale的输入image来训练。在我们这篇文章里，我们解决CNN结构的另一个重要的问题：深度。为了达到这个目的，我们将架构的其它参数固定，然后通过增加更多的convolutional layer来使得网络加深，这在训练上和计算上都是可行的，因为我们使用了$$ 3 \times 3$$的filters（$$ 3 \times 3$$的filters配合padding和stide=1可以使得输出的feature map和输入的feature map长宽不变，从而可以无限的加深下去）。

结果是，我们获得了一个精度更高的CNN模型，不仅在ILSVRC classification和localisation任务上取得了sota的效果，还可以用在其它的image recognition数据集上，仍然可以获得很好的效果，即使是很简单的设计（比如说将deep features直接利用SVM分类，不需要任何的fine-tuning）。我们公开了两个效果最好的模型为了将来的研究。

这篇文章剩下的部分结构如下。在section5里，我们描述了我们设计的CNN模型。image classification的训练和测试细节再section6里。我们的框架和其它人的模型的对比再section7里。section8总结整篇文章。


**5. ConvNet Configurations**

为了在一个公平的设定下研究增加CNN的深度带来的影响，我们所有的CNN层都使用同一个结构。在这一个section里，我们先描述我们的CNN模型的整体结构，然后再描述一些细节的设置。我们的设计细节最后再和之前的工作进行对比。


**5.1 Architecture**

在我们的训练过程中，我们的CNN输入是一个固定大小的$$224 \times 224$$的RGB image。我们做的唯一的pre=processing就是在训练集里，将所有的图片都减去它们的平均值（element
-wise）。这个image之后通过一系列的convolutional layer，在这些convolutional layer里，我们都使用的是$$3 \times 3$$的filters（这是能够获取一个pixel的上下左右信息的最小
filter size）。stride一直设定为1，而padding的设置使得每层convolutional layer的输入和输出的长宽是一样的，也就是说对于$$3 \times 3$$的filter，padding是1。而max-pooling在
某些convolutional layer后会出现，整个CNN一共有五个max-pooling，都是使用的$$2 \times 2$$的大小，stride是2。
  
在Convolutional layers之后，接了三个fully connected layers：前两个都是有4096个通道，而第三个是1000个通道，因为ILSVRC的分类是1000类，所以用这个作为输出，正好每个通道表示每个类。最后一层是一个soft-max layer，将第三个fully connected layer的输出转换为表示为每个类的概率。
  
对于所有的隐藏层，我们都用ReLU作为activation function。我们所有网络都没有用到local response normalisation（LRN）的技术。

**5.2 Configurations**

我们这篇文章里所使用的CNN的结构配置总结在Table1里，每列表示一个网络。再之后我们就使用A-E来表示这些网络。所有的网络设计都按照5.1里描述的那样，它们仅仅在深度上有区别：从11层（8层CNN，3层fc）到19层（16层CNN，3层fc）。convolutional layer的宽度也不大（也就是每个convolutional layer输出的channel数），从64开始，每次遇到max-pooling之后就增加一倍，直到到达512为止。
![TABLE1]({{ '/assets/images/VGG-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 1*
  
Table2总结了每个网络的参数的数量。尽管我们的网络深度很大，但我们网络的参数的数量并不比那些有着更大convolutional filters的更浅的网络要多。
![TABLE2]({{ '/assets/images/VGG-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 2*

**5.3 Discussion**

我们的CNN结构和在ILSVRC-2012和ILSVRC-2013竞赛里表现最好的那些模型都很不一样。他们的网络的第一层用stride=4的$$11 \times 11$$的filter，或者用stride=2的$$7 \times 7$$的filter，然而我们这篇文章里用的是很小的stride=1的$$3 \times 3$$的filter。很容易看出来，两个$$3 \times 3$$的filter堆叠起来，就等价于一个$$5 \times 5$$的filter。三个$$3 \times 3$$的filter堆叠起来，就等价于一个$$7 \times 7$$的filter的感受野。所以说，我们通过堆叠三层$$3 \times 3$$的filter而不是一层$$7 \times 7$$的filter，得到了什么？首先，我们的层与层之间还有非线性层，而$$7 \times 7$$的只有一个非线性层，这能使得我们的决策函数更加复杂。其次，我们减少了参数的数量：假设我们的$$3 \times 3$$的层的输入和输出的channel都是$$C$$，从而对于一共是27$$C^2$$个参数；而同时，一个$$7 \times 7$$的filter的参数是49$$C^2$$。

注意到我们还使用了$$1 \times 1$$大小的convolutional layer，这样的层可以在不影响其他层卷积感受野的情况下，增加模型的非线性程度，也可以改变通道数，因为这个层之后也是有activation function的。

之前也有人使用过小的convolutional filters，但是它们的网络并不深，而且没有在ILSVRC数据集上进行测试。Goodfellow关于识别街景图片里的数字的文章使用了深的CNN（11层）并且说明深的网络效果更好。GoogLeNet在我们这篇文章的同时也研究了小的convolutional filters和深的CNN。



### 3. [ResNet: Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*

*CVPR 2016*

**AlexNet(2012) $$\rightarrow$$ ResNet(2016)**

**1. title**

deep residual learning是文章提出的方法，而image recognition则是要解决的问题，同样也是简洁明了的标题。

**2. Authors**

四个人都是大佬，这是在微软研究院所做的工作。

**3. Abstract**

训练深的神经网络非常不容易。我们使用深的残差网络使得训练比之前容易很多。我们将结构设计为，让每一层去学习相对于这一层输入的残差（因为结构设计为$$output = x + f(x)$$，从而$$f(x) = output - x$$即是残差）。我们通过大量的实验来证明我们提出的residual networks很容易训练，对于增加的深度来说也是一样。在ImageNet数据集上我们使用了152层深度的网络，比VGG深了8倍。最后我们在ILSVRC 2015的竞赛里达到了3.57%的test error而获得了冠军。我们还在CIFAR-10数据集上演示了如何训练100和1000层的网络。

对于很多视觉的任务来说，深度是很重要的。我们仅仅是将分类器用到的feature换成deep residual network所学习到的feature，就使得COCO object detection的准确率提升了很多。我们还赢得了ImageNet detectino， ImageNet localization，COCO detection和COCO segmentation比赛的冠军。

**4. Introduction**

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

**5. Related Work**

**5.1 Residual Representations**

**5.2 Shortcut Connections**


**6. Deep Residual Learning**

**6.1 Residual Learning**

假设$$H(x)$$是我们想要通过几层堆叠的layers要拟合的mapping，$$x$$是这些堆叠的layers的输入。如果假设多层的layers的堆叠能够拟合任意复杂的mapping，那么它也可以拟合residual mapping，也就是$$H(x)-x$$（假设输入和输出的维度是一样的）。所以说，我们不让堆叠的layers去拟合$$H(x)$$，而是显式的让它们去拟合residual mapping，$$F(x)=H(x)-x$$。从而我们所想要拟合的原mapping就可以表示为$$H(x)=F(x)+x$$。虽然说堆叠的layers都有能力去学习$$H(x)$$和$$F(x)$$，但是对于neural networks来说，学习的难易程度是不一样的。

上述的过程是由我们在introduction里的内容而启发的。如果我们所添加的layers在学习之后就是identity mapping，那么添加了这些layers至少不应该使得training error变大。但实际上，很深的neural networks的效果是会变差的，也就是说training error是会变大的。发生这种现象表明想要深的网络训练好，是很困难的。而使用residual的结构，则会使得训练简单很多。


**6.2 Identity Mapping by Shortcuts**

我们每过几层就采用residual learning的结构。而每个residual block就如fig 2所示。公式化来说，我们的residual block是这样的：
  
$$y = F(x, \{W_i\}) + x$$
  
$$x,y$$分别是这个block的输入和输出，而$$F$$代表residual mapping，$$W_i$$是参数。
  
上述公式里的shortcut既没有增加新的参数，也没有增加计算复杂度。这不仅在实践中会高效很多，也使得我们在实验中进行对比的时候，能控制其他变量不变，只是将shortcut的部分去掉，从而对比很公平。
  
在上述公式里我们可以看到，最后的结果$$y$$是residual mapping $$F$$和$$x$$的element-wise的addition，从而$$x$$和$$F$$的维度得是一样的。如果它们的维度不一样，比如说input和output的channels发生了变化，我们就引入linear projection $$W_s$$来解决问题：
  
$$y = F(x, \{W_i\}) + W_sx$$
  
residual mapping $$F$$的设计是多样的。我们这篇文章里$$F$$的设计用了2层或者3层 layers，而更多的层也是可以的。但是如果只有一层，那么其实residual block就变成了linear layer：$$y = Wx + x$$，这样是不行的。
  
而且上述的这些论述不仅仅是对于layers是MLP时候，layers是CNN的时候，同样也是适用的。


**6.3 Network Architectures**

我们测试了多种不同的plain/residual neural networks，对于我们的结论并没有什么不一样。用于ImageNet的plain和residual network结构分别如下：

**Plain Network**

我们的plain network的结构如fig 3中间那幅图所示。

**Residual Network**

我们在plain network的基础上加入shortcut connections，从而将网络改成了residual neural networks。对于输入和输出维度相同的residual blocks，shortcut connections就直接是elementwise addition。而对于不同的情况，比如说fig 3里的虚线，可以用zero padding的方法，也可以用$$ 1 \times 1$$的convolution来使得feature map的通道数相符。

![network]({{ '/assets/images/RESNET-3.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 3. ImageNet的示例结构。左侧：VGG-19模型。中间：34层的plain network。右边：34层的residual network。*

**6.4 Implementation**

输入的图片的短边随机从256到480的范围内采样，长边则是按比例计算。之后再从已经scale的图片里随机裁剪$$224 \times 224$$大小的图片，之后再随机的选择是否需要水平翻转。再之后将图片的平均值减去（elementwise）。而且在每个convolution之后，activation之前还采用了batch normalization。使用SGD进行优化，batch_size选取为256。learning rate从0.1开始并且在error到达平台期后手动减小至1/10。使用了weight decay-0.0001和momentum=0.9。并没有使用dropout。
  
>dropout常见于MLP或者fully connected layer里，convolution layer一般不用。
  
在测试的时候，我们从测试图片里随即裁剪10张进行测试。而且还对scale不同的image进行了测试再将结果取平均。
  
>这些都是刷榜的技巧，使得结果更好。


**7 Experiments**

**7.1 ImageNet Classification**

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


**7.2 CIFAR-10 and Analysis**


**7.3 Object Detection on PASCAL and MS COCO**


>为什么ResNet训练的效果很好？实际上可能是因为它解决了梯度消失的问题。对于plain network来说：

>$$ \frac{df(g(x))}{dx} = \frac{df(g(x))}{dg(x)} \frac{dg(x)}{dx} $$

>梯度都是很小的，所以说这两个乘积项本身都很小，乘在一起就更小了。而网络很深的时候，误差反向传播到很前面的网络的时候，就容易发生梯度消失的问题。

>对于residual network来说，情况就不一样了：

>$$ \frac{df(g(x)) + g(x)}{dx} = \frac{df(g(x))}{dg(x)} \frac{dg(x)}{dx} + \frac{dg(x)}{dx} $$

>两项梯度相加，就很有可能使得梯度不是那么小了，从而就可以训练的动了。

**8. Conclusion**

这篇文章是没有结论的。因为文章内所要说的结果太多了，超出了会议规定的文章最大页数。这种写法是不建议的，最好还是要有conclusion，使得文章具有完整性。


## Transformer-related architectures

### 1. [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*

*NIPS 2017*

**1. Title**

这个标题很有意思，开创了一个潮流，之后很多论文都用类似的标题。

**2. Authors**

这篇论文有8个作者，且对文章的贡献都是均等的，而且还有额外的解释说明了每个作者的贡献。这是很少见的，但是这是Google的论文，他们经常做这种骚操作。

**3. Abstract**
主流的序列转录模型（输入和输出都是序列，比如说机器翻译，输入是一种语言序列，输出是另一种语言序列）主要都是用复杂的recurrent或者convolutional的neural network，而且都是有encoder和decoder的结构。而且表现很好的那些模型，在encoder和decoder之间是有attention机制进行连接的。我们这篇文章提出了一个简单的架构，只用attention机制，而没有任何recurrent或者convolutional的结构，叫做Transformer。我们在两个机器翻译任务上测试了Transformer，发现它效果更好，而且并行度更好所以训练的时间也更短。我们的模型在WMT 2014 English-to-German的翻译任务上达到了28.4 BLEU。在WMT-2014 English-to-French的翻译任务上，通过8个GPU训练了3.5天达到了41.8的BLEU。


>这篇文章一开始是针对机器翻译而写的，所以整篇文章、摘要都只提到了机器翻译的实验。之后因为Bert等文章使得Transformer这篇文章火了，才受到大量人的关注。但这篇文章一开始就是机器翻译领域的文章。

**3. Introduction**
RNN，包括LSTM，gated RNN等，在序列转录任务里包括Language modeling，machine translation等具有重要的地位，基于RNN的模型也在这些任务里获得了最好的结果。recurrent language models和encoder-decoder architecture是最主流的两个模型。

RNN计算一个序列的时候，是从左往右挨个词来看。对于第t个词，会计算一个输出$$h_t$$，也叫做它的隐藏状态，而$$h_t$$是由前面一个词的隐藏状态$$h_{t-1}$$和当前的词来决定的。这样就可以将历史信息传递过去，这也就是RNN为什么能够处理时序信息的关键。这个时序的特征使得并行计算是不可行的，这对于很长的序列是很致命的。另一个缺点是时序信息是按串联的方法依次传递下去的，所以对于很早的历史信息就可能会丢失掉。解决这个问题的办法就是使用更大的$$h_t$$，但是这会导致计算量和内存量的需求变大。

attention机制已经在很多任务的模型里被应用了，对于input和output序列来说，attention机制可以表示它们之间或者内部的依赖关系，而不需要顾及依赖关系双方的距离有多远。但attention机制一直是与RNN一起使用的。

>在RNN里用的attention机制主要是将encoder的信息有效的传递给decoder。

在这篇文章里，我们提出了Transformer，不再使用recurrence结构而是只使用attention机制来构建input和output之间的依赖关系。我们的Transformer可以进行并行计算，使得训练时间大大减短，而且效果更好。

>这个introduction写的比较短，可以认为是abstract的扩充。这么写是因为后面的主题内容过多，而NIPS是一个篇幅较短的会议，所以只能压缩其他部分的内容。


**4. Background**

为了减少序列计算的成本，很多基于CNN的架构被提了出来，它们都是使用并行计算来构建input和output之间的依赖关系。但是因为CNN的卷积核是有大小的，CNN的卷积操作是在数据上按照卷积核的大小以一定步长滑动的，所以说如果两个相距很远的词想要被学习到之间的关系，就需要很多的卷积操作才可以（因为CNN是卷积层的叠加，后面的卷积操作基于前面的卷积操作）。从而CNN的这种特性使得学习距离很远的词之间的依赖关系变的更加的复杂。使用Transformer结构的话，模型一层就可以看到所有的词，就不会有CNN那样的问题。但CNN的一个优势是有很多的通道，而每个通道都可以代表不同的模式，而我们也想在Transformer里实现这样的效果，所以我们使用Multi-head attention的结构，希望能够模仿这种多通道的效果。

self-attention，有时候也叫做intra-attention，是一种attention机制，它作用在一个单一序列的不同的位置上，从而计算出这个序列有效的representation。self-attention机制也并不是这篇文章提出来的，很早以前就在很多地方被使用过了。

Transformer是第一个只依赖于self-attention机制而不需要用到任何recurrent或者convolutional结构来计算input和output的representations的模型。


**5. Model Architecture**

目前比较好的序列转录模型都有一个encoder-decoder的结构。encoder将输入的序列$$(x_1,...,x_n)$$映射到$$z = (z_1, ..., z_n)$$，$$z$$的长度也是$$n$$，而且每个$$z_i$$是一个向量，是$$x_i$$的feature。而decoder则是将$$z$$作为输入，而输出一个$$(y_1,...,y_m)$$，$$m$$和$$n$$是不一定一样的。而且$$(y_1,...,y_m)$$里的$$y_i$$是一个一个依次生成的。这样的一个生成的方式叫做auto-regressive，也就是说在每一步都需要前面已经生成的$$y$$作为额外的输入，配合$$z$$来生成新的$$y$$。

而Transformer也是采用了这样一种encoder-decoder的结构，Transformer将self-attention和point-wise fully connected layers堆叠在一起实现的encoder和decoder。如fig 1左右两部分所示。

![architecture]({{ '/assets/images/TRANSFORMER-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. Transformer：模型结构。*
 
**5.1 Encoder and Decoder stacks**
 
**Encoder**: encoder是由$$N=6$$个完全一样的层堆叠构成的。而每个层都有两个子层。第一个子层是multi-head self-attention机制，第二个子层实际上就是个MLP。对于每个子层我们还有残差连接，而且每个子层还有个layer normalization。也就是说，每个子层的输出为： LayerNorm(x + Sublayer(x))，其中Sublayer(x)是子层里的function（也就是前面说的multi-head self-attention或者是MLP）。为了简便残差连接，这个模型里所有的子层，以及初始的input embedding，输出的大小都是$$d_{model} = 512$$。

>这种简单的设计使得模型调参很简单，因为encoder部分就两个超参数，$$N$$和$$d_{model}$$。

>通过和BatchNorm对比一下解释LayerNorm是什么，以及为什么我们在这种输入是变长的模型里不使用BatchNorm。假设我们的输入是2维的，row表示batch，column表示feature，也就是每一行是个样本，每一列是个特征。BatchNorm是对于每个batch，将每列特征的均值变成0，方差变成1，这是通过计算整个batch对于每一列的均值和方差，之后再减去均值，除以方差的根号来实现的。最后还有可学习的参数，$$\lambda$$和$$\beta$$，来对normalization的值做一个线性转换，使得其表达性更强。而LayerNorm和BatchNorm很像，也是用来做归一化的，但区别在于LayerNorm是将这个2维输入的每一行进行归一化，也就是对于每个输入，其内部进行归一化。如fig 2所示。

![LN]({{ '/assets/images/TRANSFORMER-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. BatchNorm和LayerNorm的2维例子，row是batch，column是feature。*

>但是对于Transformer或者正常的RNN而言，输入是3维的，因为输入的是一个batch的序列，而每个序列里有很多个元素，每个元素又可以被表示为一个embedding向量，从而输入是个$$batch \times sequence \times features$$的三维数据。对于BatchNorm来说，我们仍然对于每个feature都做归一化，如fig 3蓝色方框所示，也就是对于每个feature而言，对于所有的batch和所有的sequence的元素求归一化。而对于LayerNorm来说，我们对每个batch做归一化，也就是说对于每个batch来说，我们对所有的feature和所有的sequence内的元素求归一化，如fig 3黄色所示。

![EX]({{ '/assets/images/TRANSFORMER-3.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 3. BatchNorm和LayerNorm的3维例子。*

>所以说为什么要用LayerNorm而不是BatchNorm呢？因为很多时候，我们一个batch里面的sequence里包含的元素的个数是不等的，如fig 3所示。那么如果用BatchNorm，如fig 4蓝色部分所示，在计算方差和均值的时候，有很多部分是0，是没有用的。那么在各个batch之间，这种sequence内元素长度的变化，会导致方差和均值计算的不稳定，很波动。而且还有个问题就是如果在测试的时候遇到了一个sequence，里面包含的元素的数量比所有的训练时候所看到的sequence内的元素数量都要多，如fig 4上面那条蓝色的长条所示，那么这个时候用之前所计算的均值和方差来处理这个数据就是不合理的。而对于LayerNorm来说就不存在这个问题，因为LayerNorm计算均值和方差是在每个sequence内部计算的，所以不存在其它的sequence还能影响它，从而长度也不会影响。

![E4]({{ '/assets/images/TRANSFORMER-4.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 4. BatchNorm和LayerNorm生成的结果的样子。*


**Decoder**: decoder和encoder是很像的。decoder同样是由$$N=6$$个相同的层堆叠而成的。每个层有和encoder一样的两个子层，而且decoder还多了个子层，同样也是multi-head attention，但这回不是self-attention，而是利用了encoder的输出。decoder里的每个子层也用了残差连接的结构，并且跟上了layernorm。因为我们采用的是自回归的方式，所以decoder里的multi-head self-attention的结构也需要更改一下，将还没有看到的那些输出遮住，也就是我们所说的masked multi-head attention。

接下来我们来看每个子层是什么样的。

**5.2 Attention**

一个attention function可以被描述为将一个query和一个key-value对的集合映射到一个output，其中query，key，value和output都是向量。output是values的一个加权和，而每个value的权重则是通过用compatibility function来衡量query和key获得的。而不同的attention的结构里，compatibility function用的不一样，也就是衡量相似度的方式不一样。

**5.2.1 Scaled Dot-Product Attention**

我们这篇文章里用的attention叫做scaled dot-product attention，如fig 5左边所示。

![att]({{ '/assets/images/TRANSFORMER-5.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 5. 左：scaled dot-product attention。右：multi-head attention（有多个并行计算的attention layers）。*

我们attention的输入里的query和key的维度都是$$d_k$$，而value的维度是$$d_v$$（从而output的维度也是$$d_v$$）。我们的做法是对于query和每个key做dot product，再除以$$\sqrt(d_k)$$，然后再用一个softmax function来获取每个value的权重。

在具体操作中，我们针对一系列的query同时进行计算。将这些query表示为Q，key和value也同样表示为K和V，Q, K, V都是矩阵。从而，output的矩阵就是：

$$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt(d_k)})V$$

两种最常见的attention functions是additive attention和dot product attention。我们这里用的就是dot product attention，除了还额外除以了$$\sqrt(d_k)$$。而addtive attention是通过一个只有一个隐藏层的MLP实现的， 它可以处理query和key不等长的情况。这两种attention function在理论上复杂度差不多，但在实践上因为矩阵乘法有高效算法，所以后者会快很多。

当$$d_k$$不大的时候，上述两种attention function的效果差不多，但$$d_k$$大的时候，如果不除以$$\sqrt(d_k)$$，additive attention的效果要更好。我们猜测是因为$$d_k$$很大的时候，计算出来的dot product有的会很大，从而到了softmax function的饱和区，会在反向传播的时候产生很小的gradient，影响训练效果。所以我们除以了$$\sqrt(d_k)$$。
（因为softmax的作用本来就是达到一个效果，在大的地方输出接近1，小的地方输出接近0，如果已经达到了这种效果了，那么softmax就难以再训练了）

而对于mask而言，上述计算依然进行，只不过在进入softmax之前，对于那些还不应该被看到的output的value对应的权重，将其都置成非常大的负数，比如说-1e10，这样在进入softmax之后，其权重就几乎是0，从而被排除在外。

**5.2.2 Multi-Head Attention**

与其我们只用一个key，value和query的维度都为d_{model}的attention function，我们发现将query，key和value用不同的可学习的linear projection映射到$$d_k$$，$$d_k$$和$$d_v$$维度$$h$$次，从而学习$$h$$个output的效果更好。这些$$h$$个attention function还可以并行操作，从而计算的更快。而这$$h$$个output连接在一起，维度是$$d_k \times h$$，再经过一次linear projection，从而得到$$d_v$$维度的output。如fig 5右侧所示。

>为什么要用multi-head的结构呢？因为我们可以发现attention function其实没有什么可以学习的参数，相似度就是内积。但我们有时候希望能学习到更多的模式，而这些模式可能需要不一样的计算相似度的方法。如果用的是additive attention的结构，其实那里面还有一些可以学习的参数，但这里没有。从而这里的做法是给$$h$$次低维的投影，希望其能够找到有效的投影模式，从而学到有效的内容。而投影是有参数可以学习的，从而增加了模型的学习能力。有点像CNN里的多个通道的意思。

multi-head attention使得模型能够从不同的subspace里学到不同的信息。

$$MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O$$

其中$$head_i = Attention(QW_I^Q, KW_i^K, VW_i^V)$$，而这些linear projections都是可学习的矩阵$$W_i^Q \in R^{d_{model} \times d_k}$$，$$W_i^K \in R^{d_{model} \times d_k}$$，$$W_i^V \in R^{d_{model} \times d_v}$$，$$W^O \in R^{hd_v \times d_{model}}$$。

在这篇文章里，我们使用$$h=8$$。使用$$d_k=d_v=d_{model}/h=64$$。因为我们做multi-head的时候降维度了，所以其实multi-head的计算量和single-head的差不多。

**5.2.3 Applications of Attention in our Model**

整个Transformer模型用三种不同的方式使用multi-head attention。

* 在encoder-decoder的attention子层里，query是从前面一个decoder的子层来的，而key和value则是由encoder的输出提供。这种设计可以使得decoder里的每个位置都能获取输入的sequence所有位置的信息。这模仿了在sequence-to-sequence模型里的典型的encoder-decoder attention的机制。
* encoder本身包含了self-attention子层。在一个self-attention子层里，key，value和query都是一样的，也就是前一个子层的输出。从而encoder的每一个位置都可以获取encoder其它位置的信息。
* 相似的，decoder里也有self-attention子层，它允许decoder的每个位置都能获取该位置之前的所有位置的信息。因为这是个auto-regressive的模式，所以说每个位置后面的信息它是获取不到的。我们通过在softmax之前将对应的权重设置为很大的负数来做到mask out。


**5.3 Position-wise Feed-Forward Networks**

encoder和decoder的每个层，除了有attention子层以外，也有还有一个fully connected network，其对于输入的每个词单独作用（也就是说有每个子层只有一个MLP，其作用在每个词上，也就是每个position上，而不是一个整体的大的MLP，这就是position wise名字的来由）。而这个MLP有两个线性层，并且其中还有ReLU：

$$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$$

虽然说在同一个子层里对于不同的position（词）我们使用的是同一个MLP，也就是同一个linear transformation（with ReLU），但不同的层其内部的MLP的参数还是不一样的。另一种描述这个MLP的方式就是kernel size是1的convolution。对于这个FFN来说，input和output的维度都是$$d_{model}=512$$，内部隐藏层的维度是$$d_{ff}=2048$$。

>attention的作用其实就是将序列里的信息抓取出来，做一次汇聚。因为在汇聚之后每个position其实就已经有序列整体的信息了，所以说不同position的MLP可以是同一个。

>RNN和Transformer的区别：见fig 6。Transformer和RNN一样，都是用一个MLP来做语义空间的转换，而且RNN也是用了同一个MLP来处理不同position（实际上RNN就只有一个MLP在不断更新），但不一样的是如何传递序列信息。RNN是把上一个时刻的输出作为下一个时刻的一个输入，而Transformer是通过attention结构来获得全局的序列信息，再用MLP进行转换。它们的关注点都在于如何有效使用序列信息，但实现的方式不一样。


**5.4 Embeddings and Softmax**

和别的sequence tranduction模型类似，我们使用需要学习的embedding来将encoder和decoder的输入的tokens都映射到$$d_{model}$$维的向量。我们同样使用需要学习的linear transformation和softmax function将decoder的输出转换为预测下一个token的概率。在我们的模型里，encoder和decoder的tokens的embeddings层的权重和decoder之后的linear transformation的权重是共享的。在embedding层，我们将这个层的参数乘以$$\sqrt(d_{model})$$。

>因为embedding最后容易将每个输入的norm都学成l2norm接近于1，所以乘上一个系数，以便它的值的范围和positional encoding的范围差不多，不至于过小。


**5.5 Positional Encoding**

>attention其实并没有任何时许信息，那么即使打乱一个句子的词的顺序，也会是一样的效果，这是不应该出现的，所以我们需要加入时序信息。RNN是本身就有时序结构，而Transformer的做法是直接在输入里加入时序信息。

因为我们的模型并没有recurrence或者convolution的结构，所以为了我们的模型能够利用sequence的时序信息，我们需要加入一些sequence里的tokens的相对或者绝对的位置信息。为了达到这个目标，我们在encoder和decoder的输入的embeddings上加上了positional encoding。positional encoding和embedding一样也是$$d_{model}$$维的，所以两者可以相加。positional encoding有很多种。

在我们这篇文章里，我们使用不同频率的sine和cosine function来表示positional encodings：

$$PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})$$

其中pos是position，i是dimension。也就是说，每一个dimension，都对应着一个三角函数。


**6. Why Self-Attention**

在这一节里我们将会比较self-attention层和recurrent以及convolution层之间各个不同的方面。

首先是每个层的计算复杂度。另一个是可以被并行的计算的总和。第三个是长距离的两个position之间需要信息交互需要的计算量。学习长距离的依赖关系对于很多sequence transduction任务来说都是很具有挑战性的。影响模型学习长距离依赖关系的一个重要的点就是距离远的两个position之间的信息传递最少要经过多少计算。任意的两个position之间的信息交换所需的计算量越小，模型学习这种长距离的依赖关系的能力就越强。结果如table 1所示。

![tab]({{ '/assets/images/TRANSFORMER-7.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Table 1. *

**Experiment**和**Result**就不讲了，因为都是NLP领域的实验，和我的方向没什么关系。


**7. Conclusion**
我们介绍了Transformer，这是第一个仅仅依赖于attention机制的序列转录模型，将所有的recurrent layers都换成了multi-headed self-attention。

在机器翻译这个任务上，Transformer要比那些recurrent或者convolutional layers结构的模型训练起来要快很多。在WMT-2014 English-to-German和English-to-French的任务上确实效果很好。

我们对于这种只利用了attention机制的模型十分有信心，认为它可以被应用在别的任务之上。我们打算将Transformer应用到text以外的数据上，包括images，audio，video等。而使得生成不是那么的时许化也是另一个未来的研究方向。



### 2. [BERT: Pre-training of Deep Bidirectional Transformers for Language understanding](https://arxiv.org/pdf/1810.04805.pdf)

[CODE](https://github.com/google-research/bert)

*Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova*

*Arxiv 2018*

**Transformer $$\rightarrow$$ BERT**

>在CV领域，很早之前就可以使用CNN在一个很大的数据集，比如说ImageNet上学习一个模型，而这个模型可以为很多的CV任务提升他们的性能，也就是说这是一个pre-trained的模型。但是在NLP领域，在BERT之前，一直都没有一个训练好的NLP网络，使得其再训练好之后能够帮助后面一批的NLP任务。从而NLP任务都是针对每个任务单独构造网络然后单独训练。BERT的出现使得NLP领域有了这样以一个在很大的数据集上训练好的pre-trained模型，其能够应用于后续很多的NLP任务当中，既简化了这些NLP任务的训练，又提升了它们的性能。所以说BERT以及后续的改进使得NLP领域在过去几年内有了个质的飞跃。

**Title**

BERT是起好了一个名字，这样方便之后的人引用或者提及。pre-training意味着我们这篇文章的目的是训练一个pre-trained的模型，用于后续其它一批任务上。deep bidirectional transformers解释了所用的主要的网络结构，而language understanding则解释了这个pre-trained模型之后可以应用在哪些任务上。

所以说这个标题告诉我们，BERT这个模型是一个深的双向的transformer，是用来做预训练的，针对的是一般的语言理解型的任务。

**Authors**

是Google AI语言组的人，一作从有想法到跑第一次代码成功只花了几周的时间，之后写完论文也就花了几个月的时间。


**Abstract**

我们提出了一个新的language representation模型，叫做BERT，是Bidirectional Encoder Representations from Transformers的简称。和最近的language representation模型不一样，BERT通过在每一层里都联合左右两侧的上下文来在没有标签的文本上学习预训练的bidirectional representation。结果是，预训练的BERT模型再加上一个附加的output层就可以对一系列任务都有sota的效果，比如说question answering，language inference等，而不需要对具体任务来调整模型的结构。

BERT在概念上很简单，在实验上效果很好。其在11个NLP任务上达到了sota的效果。


**1. Introduction**

语言模型预训练对于提升很多NLP任务性能都有显著的效果。这些NLP任务包括sentence-level的任务比如说natural language inference，paraphrasing等，它们通过整体分析句子来预测sentences之间的关系，也包括token-level的任务比如说named entity recognition，question answering等，它们在token level生成更加细粒度的输出。

现在有两种将预训练的语言特征应用到下游任务的方法：feature-based和fine-tuning。feature-based方法，比如说ELMo，使用task-specific架构，将预训练的特征作为补充特征来使用。fine-tuning方法，比如说GPT，尽量对于各种任务都使用同样的模型结构，在各个不同任务的数据集上来微调已经预训练好的模型参数。这两种方法在预训练时是一样的，使用的是一样的目标函数，也就是使用unidirectional language models来学习语言特征。

我们认为现有的这些方法限制了预训练语言特征的效果，特别是对于fine-tuning方法。最主要的一个限制点是标准的语言模型是单向的（unidirectional），这限制了可以被用来做预训练的模型种类。比如，在GPT里，作者使用了一个从左到右的模型结构，其中每个token在Transformer结构的网络里只能获取它之前的tokens的信息。这样的限制对于sentence-level的任务来说是sub-optimal的，而对于使用fine-tuning方法的token-level任务来说也是很不好的，比如说question answering，因为双向信息会格外的重要。

在这篇文章里，我们改进了fine-tuning based方法，提出了BERT：Bidirectional Encoder Representations from Transformers。BERT通过使用了一个masked language model (MLM)减轻了之前提到的单方向的限制。masked language model随机的在输入里遮住一些tokens，而目标则是根据上下文来预测被遮住的tokens。和从左到右的预训练模型不一样，MLM的目标函数使得所学习到的representations能够融合左右两侧的信息，也就允许我们预训练一个bidirectional Transformer。除了MLM，我们还使用了一个next sentence prediction任务来联合预训练text-pair representations。我们这篇文章的贡献总结如下：

* 我们阐述了双向预训练模型对于语言representations的重要性。和GPT使用单向语言模型学习representation不同，BERT使用MLM来学习双向语言representations。这和ELMo也不一样，ELMo是仅仅将从左到右和从右到左分别学习到的语言representations简单的连了起来。
* 我们说明这样的预训练representations使得很多针对具体任务设计具体模型结构的努力变得不再需要了。BERT是第一个在很多sentence-level和token-level的下游任务上都获得了sota效果的预训练模型，效果超过了很多根据具体任务设计具体模型结构的方法。
* BERT在11个NLP任务上达到了sota的效果。


**2. Related Work**

预训练的语言representations有着很长的历史，所以只能简要的介绍最著名的那些。

**2.1 Unsupervised Feature-based Approaches**

学习words的能够广泛使用的representations已经是一个研究了几十年的方向，包括non-neural和neural方法。预训练好的word embeddings是现在NLP系统里不可或缺的一部分，比之前hand-crafted的embeddings效果要好很多。为了预训练word embedding向量，很多方法应用了从左到右的语言模型目标函数。

这些方法之后被扩展到更粗粒度的情况，比如说sentence embeddings，或者paragraph embeddings。为了训练sentence representations，以前的工作有的使用目标函数来排列下一个sentence的候选，有些工作基于之前的sentences的representations生成下个sentence，还有些工作使用denoising autoencoder。

ELMo以及其后续跟进工作将传统的word embedding拓展到一个新的高度。ELMo利用一个从左到右和一个从右到左的语言模型来获取上下文相关的features。每个token的上下文相关的representations就是将从左到右的模型的特征和从右到左的模型的特征连起来而获得。和目前已有的task-specific模型相结合起来，ELMo在多个NLP任务上取得了sota的效果，包括question answering，named entity recognition，sentiment analysis等。还有研究工作提出使用LSTM来从左侧和右侧上下文预测单个词。和ELMo类似，它们的模型是feature-based的而且并没有深度双向。

**2.2 Unsupervised Fine-tuning Approaches**

正如feature-based方法，fine-tuning based方法也是从无标签的数据上预训练word embeddings。

最近，生成上下文相关的token representations的sentence或者document encoders也是从非标签的文本上预训练或者从监督的下游任务上训练的。GPT在很多sentence-level的任务上获得了sota的效果。从左到右的语言模型以及auto-encoder目标函数被用来预训练这样的模型。


**3. BERT**

在这一节里我们介绍BERT和它详尽的细节。我们的框架有两个步骤：pre-training和fine-tuning。在pre-training过程中，模型在不同的pre-training任务上使用非标注的数据进行训练。在fine-tuning过程中，BERT模型在预训练后的参数被下游的任务的监督数据进行进一步的微调。每一个下游的任务都会微调出不同的模型。fig1使用question answering作为例子来解释了这个过程。

![1]({{ '/assets/images/BERT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. BERT的overall pre-training和fine-tuning过程。除了输出层，pre-training和fine-tuning的网络结构是一样的。对于不同的下游任务，也是使用同样的预训练好的网络参数。在fine-turning过程中，所有的参数都被微调了。$$\left[CLS \right]$$是加在所有输入的前面的特殊的符号，$$\left[ SEP \right]$$是一个特殊的separator token（比如，分开questions和answers）。*

BERT的一个特殊的地方是它在不同的任务里的模型结构都是一样的。在预训练模型框架和最后的下游任务模型框架之间存在很小的差别。


*Model Architecture*

BERT的模型结构是一个多层的双向Transformer encoder，在TensorFlow的tensor2tensor库里收录。因为Transformer太火了，而我们的模型基本上和原Transformer一模一样，就不再赘述。

在这篇文章里，我们将层数（也就是Transformer的blocks数）定义为$$L$$，隐层大小为$$H$$，self-attention的head个数为$$A$$。我们在两个模型上测试结果：$$BERT_{BASE}$$，$$L=12, H=768, A=12$$，总参数为1.1亿；$$BERT_{LARGE}$$，$$L=24, H=1024, A=16$$，总参数为3.4亿。

$$BERT_{BASE}$$的模型大小（参数个数）设计为和GPT差不多，为了能够更合理的比较。


*Input/Output Representations*

为了使得BERT能够处理一系列下游任务，BERT的input representation可以用一个token sequence既能表示单个sentence，又能表示一对sentences构成的pair（比如说，$$\langle Question, Answer \rangle$$）。一个sentence可以是任意长度的连续文本，而并不是我们平常意义里的一个sentence。一个sequence表示的是BERT的token sequence输入，可能是一个sentence也可能是两个打包在一起的sentence pair。

我们使用有30000个token vocabulary的Wordpiece embeddings（[Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/pdf/1609.08144.pdf%20(7.pdf)）。每个sequence的第一个token都被设定为一个特殊的classification token，记为$$\left[ CLS \right]$$。这个token最终对应的模型的hidden state被认为是aggregated sequence representation被用来做classification任务。sentences pairs会被pack到一起成为一个sequence。我们使用下述方法来区分一个sequence里的两个sentences。首先，我们用一个特殊的token，$$\left[ SEP \right]$$来区分它们。其次，我们给每个token都加上一个可学习的embedding来表示这个token是属于sentence A还是sentence B。正如fig1所示，我们将输入的embedding记为$$E$$，特殊token，$$\left[ CLS \right]$$的最终的hidden vector记为$$C \in R^{H}$$，第$$i$$个输入token的最终的hidden vector记为$$T_i \in R^{H}$$。

对于某个token，它的输入representation通过将token的embedding，segment和position encoding加起来而获得。这个构造的可视化见fig2。

![2]({{ '/assets/images/BERT-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. BERT输入representation。每个输入的token的embedding是token embedding，segmentation embedding和positional embedding加起来获得的。*

**3.1 Pre-training BERT**

跟ELMo和GPT不一样，我们并不使用传统的从左到右或者从右到左的语言模型来预训练BERT。我们使用两个非监督的任务来与训练BERT。fig1的左边展示了pre-training的过程。

*Task 1: masked LM*

直觉上，认为一个双向的模型肯定比一个单向的从左到右的模型或者只是简单的将从左到右和从右到左模型的输出连起来的模型要强大是很正常的。但实际上，标准的conditional language model只能是从左到右或者从右到左训练，因为双向会使得每个词非直接的看到自己，从而模型可以在多层结构的上下文环境中很容易的预测target word。

为了能够训练双向的representations，我们只是将一些输入的tokens遮起来，然后让模型来预测这些遮起来的词。我们将这个过程成为masked language model（MLM）。之后，模型对应于被遮住的tokens的最后的hidden vectors被为给一个output softmax over the vocabulary，正如标准的LM里所做的那样。在我们的所有的实验里，我们遮住每个sequence里随机的15%的tokens。和denoising auto-encoders不同，我们仅仅是预测被遮住的tokens，而并不是要重构整个原输入。

尽管这样做让我们可以获得一个双向预训练模型，一个缺点是在预训练和fine-turning的时候会不一样，因为fine-tuning的时候并没有$$\left[ MASK \right]$$ token。为了缓解这个问题，我们在预训练的时候再加入一些随机性。当第$$i$$个token被选中作为mask token时，我们80%的概率将其作为mask token，10%的几率将其作为一个随机token，10%的几率不改动。然后$$T_i$$再和最初的token利用cross-entropy loss计算差别。

*Task 2: Next Sentence Prediction (NSP)*

很多重要的下游任务比如说question answering（QA）以及natural language inference（NLI）基于的是两个sentences之间的关系，这并不能直接被language model所获得。为了训练一个能够理解sentence关系的模型，我们为一个binarized next sentence prediction任务来作预训练。当选择sentence A和sentence B作为预训练例子的时候，50%的时候，B是A的下一个sentence（标注为IsNext），50%的情况是随机选的（标注为NotNext）。正如我们在fig1中所示，$$C$$就被用于next sentence prediction (NSP)。尽管结构很简单，实验证明这样做对于后续的QA和NLI任务效果提升都很有作用。


**3.2 Fine-tuning BERT**

fine-tuning是十分自然地，因为Transformer里的self-attention机制使得BERT能够建模很多下游的任务（不管输入是单个文本或者文本对），fine-turning的时候将输入输出换掉就行。对于那些使用文本对的任务，一个常见的方式是在使用bidirectional cross attention之前先独立的encode文本对。BERT并不这样做。BERT使用self-attention机制来将这两个stages统一到了一起，直接使用self-attention机制来对pack到一个sequence的两个sentences构成的pair进行处理，等价于在两个sentences之间进行bidirectional cross attention。

对于每个任务，我们很简单的将该任务的输入和输出放到BERT里，来微调BERT的系数。对于输入来说，预训练时候的sentence A和sentence B就类似于（1）paraphrasing任务里的sentence pair；（2）entailment任务里的hypothesis-premise pairs；（3）question answering里的question-passage pairs；和（4）text classification或者sequence tagging里的一个degenerate text pair。对于输出来说，对于token-level的任务（比如说sequence tagging或者question answering），token representation被喂给一个output layer，而对于classification-level的任务（比如说entailment或者sentiment analysis），$$\left[ CLS \right]$$ representation被喂给一个output layer。


**4. Conclusion**

最近由于language models的transfer learning所带来的实验上效果的提升已经证明了种族的非监督的预训练已经是很多语言模型的不可或缺的一部分了。特别是，这些预训练模型使得某些难以获得数据的任务也能够从很深的网络中获利。我们主要的贡献是进一步将上述发现拓展到bidirectional的模型架构，允许一个预训练模型在不改变结构的情况下处理一批下游任务。



### 3. [iGPT: Generative Pretraining from Pixels](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)



### 4. [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf)



### 5. [An Image is worth 16 $$\times$$ 16 workds: Transformers for image recognition at scale](https://openreview.net/forum?id=YicbFdNTTy)

*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenbron, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby*

*ICLR 2021 Oral*

**Abastract**

当Transformer架构已经成为NLP任务的标准架构时，Transformer在CV领域的应用仍然没有太大的发展。在视觉领域，attention要么和CNN一起被使用，要么就修改CNN结构里的某些内容，但是CNN的整体结构是没有改变的。我们将会表明完全依赖CNN是没有必要的，一个应用在图像patches上的Transformer在image classification任务上就可以表现得很好。当在大量的数据上与训练之后再转移到一些中等数据集大小的recognition benchmarks上（比如IamgeNet，CIFAR-100，VTAB等）时，Vision Transformer（ViT）具有和CNN sota相同的效果。


**1. Introduction**

self-attention架构，特别是Transformers，在NLP领域现在是最主流的。现在最主要的方法是在一个大的text corpus上进行预训练，然后再在特定任务的数据集上fine-tune。多亏了Transformer架构计算上的高效性和scalability，我们可以训练很大的Transformer架构的模型（超过1000亿个参数）。随着模型和数据集的增长，目前还没有看到饱和的发生。

在CV领域，卷积架构仍然占据着主导地位，比如AlexNet，LeNet，ResNet等。受到NLP领域成功的启发，一些工作开始尝试将CNN架构和self-attention机制结合起来，[Non-local Neural Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Non-Local_Neural_Networks_CVPR_2018_paper.pdf)，[End-to-end object detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)。有些工作尝试将CNN里的卷积计算换掉，[Stand-alone self-attention in vision models](https://arxiv.org/pdf/1906.05909.pdf)，[Stand-alone axial-attention for panoptic segmentation](https://arxiv.org/pdf/2003.07853.pdf)。这些将CNN里的卷积替换掉的模型，虽然理论上很高效，但是还并不能被高效用在目前硬件设备上。因此，在大规模的image recognition任务上，经典的类ResNet架构还是sota。

被Transformer在NLP领域的scalability的成功所启发，我们尝试直接将一个标准的Transformer模型应用在image上，做尽可能少的改动。为了做到这点，我们将一张图片分为patches，然后将这些patches的linear embeddings排成sequence作为Transformer的输入。image patches被当作NLP里的tokens（words）。我们以监督学习的方式来在image classification任务上训练这个模型。

当没有使用很强的regularization，在中等大小的数据集上训练时，比如说ImageNet，Transformer的结果比ResNet的结果要第几个百分点。这似乎说明Transformer在image上这样使用的效果并不好：Transformers并没有CNN固有的一些inductive biases，比如translation equivariance和locality，因此在有限的数据集上训练的模型泛化效果并不好。

但是，当我们在更大的数据集上训练的时候（1400万-3亿张图片），情况就有所变化了。Transformer在images上的效果要比很多任务的sota效果要好。


**2. Related Work**

Transformer是用来做machine translation而被提出的，现在已经在很多NLP任务里是sota的架构了。大型的Transformer-based模型通常在大的corpora上预训练，然后在具体任务上微调。BERT使用了一种denoising的self-supervised预训练任务，而GPT line of work使用language modeling作为其预训练任务。

naive的直接将self-attention机制用到images上就会变成让每个pixel来注意其它的pixels。这会造成pixel数量平方级别的计算，这scale到实际任务是不现实的。因此，为了将Transformer用到image领域，很多研究者尝试了一些近似的办法。[]()对于每个pixel，只将self-attention应用在其邻域内的pixels上，而不是考虑所有的pixels。这样的local multi-head dot-product self-attention blocks可以完全替代convolutions操作。

最接近我们这篇文章的是这篇文章，[On the relationship between self-attention and convolutional layers](https://arxiv.org/pdf/1911.03584.pdf)，其从image中获取$$2 \times 2$$的patches，并在这些patches上应用self-attention。这篇工作和我们这篇文章非常的相似，但是我们的工作更进一步的阐明大规模的预训练使得Transformers和sota的CNN框架效果差不多甚至更好。进一步的，这篇文章使用了较小的patches，只有$$2 \times 2$$，使得模型更加适用于小分辨率的图片，而我们的方法对于中等分辨率的图片仍然可以适用。

还有很多工作尝试将CNN和self-attention相结合，比如，[]()将CNN得到的feature map和self-attention得到的feature map结合起来用于image classification，或者说在CNN得到的feature maps上再继续用self-attention得到更好的features，在object detection，video processing，image classification，unsupervised object discovery和unified text-vision tasks上都有使用。

另一个相关的工作是image GPT (iGPT) [Generative pretraining from pixels](http://proceedings.mlr.press/v119/chen20s/chen20s.pdf)，其在减小了图片的分辨率和color space之后再在图片pixels上使用Transformers。模型是以一种非监督学习的方式被当做一个generative model来训练的，所得到的representation之后可以被fine-tune，也可以直接被用来做classification，在ImageNet上的classification有72%的准确率。


**3. Method**

在模型设计上我们尽可能使用原始的Transformer的设计。

![1]({{ '/assets/images/VIT-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. Model Overview. 我们将图片分为固定数量的patches，将其每个都进行linearly embedding，再加上position embedding是，喂给一个标准的Transformer encoder。为了能够实现classification，我们在sequence之前加了一个多余的可学习的classification token。*

**3.1 Vision Transformer (ViT)**

整个模型的样子如fig1所示。标准的Transformer的输入是一个1维的sequence，由token embeddings组成。为了处理2维图片，我们将图片$$\pmb{x} \in R^{H \times W \times C}$$ reshape到一个由展平的2D patches构成的sequence $$\pmb{x_p} \in R^{N \times (P^2 \dot C)}$$，其中$$(H,W)$$是原图片的分辨率，$$C$$是原图片的通道数，$$(P,P)$$是每个patch的分辨率，$$N = HW/P^2$$是patches的数量，其也是sequence的长度。Transformer在它的层之间使用的都是一个不变的latent vector size $$D$$，所以我们将patches展平，并使用一个可训练的线性投射（公式1）将其映射到一个$$D$$维的向量。我们将这个patch经过投射之后得到的$$D$$维向量成为patch embedding。

和BERT里的class token类似，我们在上述patch embeddings构成的sequence的头部加上一个可学习的embedding，$$z_0^0 = x_{class}$$，其在经过$$L$$个Transformer模块（也就是层）之后的值$$z_L^0$$用来表示image representation $$\pmb y$$，如公式4所示。在pre-training和fine-tuning的过程中，$$z_L^0$$都和一个classification head相连，这个classification head在pre-training的时候是具有一个隐层的MLP，而在fine-tuning的时候只是一层线性层。

position embeddings也被加入到patch embeddings当中，为了保留位置信息。我们使用标准的可学习的1D positional embeddings，因为更复杂的2D positional embeddings并没带来什么效果提升。现在这个embedding vectors sequence可以作为Transformer encoder的输入了。

Transformer encoder有multiheaded self-attention层（MSA）（公式2）、MLP blocks（公式3）以及每层计算完都会进行LayerNorm，并且在每个block之间还有residual连接。

$$\pmb{z_0} = \left[x_{class}; x_p^1 \pmb{E}; x_p^2 \pmb{E}; \cdots ; x_p^N \pmb{E} \right] + \pmb{E_{pos}} \tag{1}$$

其中$$\pmb{E} \in \mathbb{R}^{(P^2 \dot C) \times D}, \pmb{E_{pos}} \in \mathbb{R}^{(N+1) \times D}$$

$$\pmb{z_l^{'}} = MSA(LN(\pmb{z_{l-1}^{'}})) + \pmb{z_{l-1}^{'}} \tag{2}$$

其中$$l=1, \cdots, L$$。

$$\pmb{z_l} = MSA(LN(\pmb{z_l^{'}})) + \pmb{z_l^{'}} \tag{3}$$

其中$$l=1, \cdots, L$$。

$$\pmb y = MLP(z_L^0) \tag{4}$$


*Inductive bias*

我们发现vision transformer相比于CNN少了很多image-specific inductive bias。在CNN里，locality，二维的领域结构以及translation equivariance在模型的每一层里都存在。而在ViT里，只有MLP是local和translation equivariant的，self-attention层是global的。


*Hybrid Architecture*

作为原始图片patches的一个替代，输入Transformer模型的sequence也可以从一个CNN的feature map获得。在这个hybrid模型里，patch embedding projection $$\pmb E$$被用在从feature map上获取的patches上。


**3.2 Fine-tuning and higher resolution**

我们在大型数据集上预训练ViT，之后再在下游任务的小数据集上微调。在微调的时候，我们将pre=trained prediction head移除，加上一个初始化为0的$$D \times K$$的feedforward层，其中$$K$$表示的是下游任务的classes。一般来说，在分辨率更高的图片上微调效果会更好，这个时候我们保持patch大小不变，这样就会有更多的patches，ViT可以处理长度变化的patch embedding sequence，所以并没有问题，但这个时候预训练的positional encoding可能就不好使了，我们可以根据预训练的positional encoding embeddings在图片中的位置，使用2D插值来计算高分辨率图片的positional encoding embeddings。


**4. Experiments**

别的部分省略

**4.1 Inspecting vision transformer**

为了理解vision transformer是如何处理图片数据的，我们来分析其内部的representations。vision transformer的第一层是将展平的patches线性投射到一个满足transformer尺寸的embedding上，也就是一个更低维度的空间里。fig2左侧展示了所学习到的embedding filters的几个最主要的部分。这些部分有点像表示patches里的低维结构信息的basis functions。

在这个linearly projection之后，一个可学习的position embedding被加在patch representation上。fig2中间显示的是模型尝试去学习encode图片中patches之间的distances，和positional encoding类似，也就是相近的patches就有类似的positional encodings。而且，能看出来显示出了行列的结构，同一行/列的patches有着类似的embeddings。

![2]({{ '/assets/images/VIT-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. 左侧：RGB embedding filters（初始的28个主成分）。中间：position embeddings。右边：head的注意区域大小和网络深度的关系。每个点表示的是一层里16个head中的某个head的mean attention distance accross images。*

self-attention允许ViT在整张图片上整合信息，即使对于很低的层也是这样。我们来看看这个网络到底将这个特性实现了多少。基于attention weights，我们来计算在图片中的mean attention distance accross images，见fig2右侧，也就是每个点能关注到周围多大的区域。这个attention distance和CNN里的receptive field有点像。我们发现，有些heads在很浅的层就已经注意到了这张图的几乎所有部分，表明全局化的整合整张图的信息这个特性确实被这个模型用到了。我们还发现，模型会注意到那些有助于classification的图片区域，如fig3所示。

![3]({{ '/assets/images/VIT-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3.*


**5. Conclusion**

我们探索了将Transformer用在image recognition领域的可行性。和以往使用self-attention的方法不同，我们并不为架构引入image-specific inductive biases。相反的，我们将一张图片理解为patches的sequence，然后用标准的Transformer来处理。但这简单的，但是scable的方法和在大数据集上预训练结合起来效果很好。因此，Vision Transformer达到了或者超过一些image classification的sota结果。

虽然结果很鼓舞人，但是还有很多问题没有解决。一个是如何将ViT应用到其它的CV任务上，比如说detection和segmentation。而且，如何将文章中的large scale supervised预训练变成self-supervised预训练也是很重要的。

>实际上MAE就做到了。


### 6. [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)

*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick*

*[CVPR 2022 Oral]*

**Transformer $$\rightarrow$$ Bert(使用了Transformer架构，但是使用self-supervised的学习方法), Transformer $$\rightarrow$$ ViT(在CV中使用Transformer架构), ViT $$\rightarrow$$ MAE (正如Bert和Transformer的关系一样，MAE就是使用self-supervised的学习方法的CV中的Transformer架构)**

**Transformer $$\rightarrow$$ Bert $$\rightarrow$$ MAE $$\leftarrow$$ ViT $$\leftarrow$$ Transformer**

**Title**

标题里经常用两个词，scalable和efficient。如果做的算法很快，就可以是efficient，或者real-time，如果做的模型很大的话，就可以是scalable。vision learners说明不依赖于具体的模型种类，是一个backbone就行。autoencoders里的auto表明的是“自”的意思，不是“自动”的意思。“自”模型的特点就是标号$$y$$和样本$$x$$来自于同一个东西，比如说在NLP里，用前后的词来预测词，用来预测的标号和样本都是词，是同一种东西，所以是“自”模型（在Transformer里提到了这个是auto-aggressive模型）。在NLP里，因为“自”模型比较常见，所以只用encoder而省略前面的auto也可以，但在CV里，标号来自于图片本身的情况较少，所以需要加上auto，强调一下和其他工作的区别。

这个标题的格式，在GPT里也用到了，"a" are "b" "c", 其中"a" and "c"是名词，"b"是形容词。这种句式实际上把结论放在了标题里。而且这种说法比较客观。

**Abstract**

这篇文章说明masked autoencoders(MAE)对于CV来说是一个scalable的self-supervised learner。我们的MAE方法很简单：将输入图片的随机的patch遮起来，然后重构这些被遮住的地方。这个模型基于两个核心设计。首先，我们开发了一个非对称的encoder-decoder结构，encoder仅仅在那些可以看到的image patches上操作（并没有mask tokens的信息），而decoder从encoder给出的可见的image patches上学到的latent representations结合mask tokens来重构原图片。decoder的设计是很轻量级的。其次，我们发现，将原输入图片遮住很大一部分，比如说75%，会使得问题变成一个不那么简单并且有意义的self-supervisory任务（如果只遮住一小部分，那模型可能学会插值就能获得不错的效果，但如果遮住很大一部分，就会迫使模型学习那些更好的特征）。将这两点结合起来，使得我们能够高效且准确的训练大型模型。这个模型主要是用作迁移学习的，是一个预训练模型，可以用于一系列下游的CV任务。


**Figures**

![MAE]({{ '/assets/images/MAE-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. MAE architecture。在pre-training过程中，很大一部分随机挑选的image patches会被遮住（比如75%）。从而encoder实际上只需要对很小一部分的image patches进行编码。mask tokens在encoder编码之后被用进来和encoder所得到的features一起按顺序排列好传给一个轻量级的decoder来重构原始输入图片。在pre-training之后，decoder就不需要了，encoder就可以直接被用于获取图片的有效特征（这个时候就不需要遮盖了，这个encoder就是一个特征提取器），用于下游任务。*

![IMAGENET-TEST]({{ '/assets/images/MAE-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 2. 在ImageNet验证集上测试MAE的效果。这些用于测试的图片都是没有参预训练的。左侧一列是随机去掉80%的image patches部分的图片，中间是MAE重构的效果，右边一列是真实的原图片。*

![COCO-TEST]({{ '/assets/images/MAE-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 3. 在COCO验证集上测试MAE的效果，使用的MAE模型是在ImageNet上训练的（也就是和figure2里用的模型参数一样）。注意到最右侧两个例子，虽然说还原的不对，但是语义上能说得通。*

![PORTION-TEST]({{ '/assets/images/MAE-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 4. 使用遮盖比例75%的数据集来训练的MAE在ImageNet验证集上的重构效果，这时候我们采用了更高的遮盖比例。这说明我们的方法可以泛化。*


**Discussion and Conclusion**

简单的并且可以scale的方法是整个深度学习的核心。在NLP里，简单的self-supervised方法最近很火。但在CV领域，现有的pre-training框架还主要是监督的（也有一些self-supervised的工作进展）。在这片文中丽，我们发现autoencoder就可以做出一个很好的pre-training框架。现在，我们没准可以像在NLP里一样成功的在CV领域使用self-supervised方法了。

但是，我们要注意到images和languages是有天然区别的两类signals，而且这样的区别一定要被谨慎对待。images仅仅是被记录的光亮强度，并没有像language一样可以被语义分割为一个个的单词。我们并没有尝试在image中遮住objects，而是随机的遮挡住patches，这些patches并不一定会构成有意义的语义块。同样的，MAE也是重构像素值，而并不是重构有语义的块。但是，实验表明我们的模型仍然能够推出复杂的重构，这表明其学习到了很多视觉概念，也就是语义。我们认为者是因为MAE内部的hidden representations涵盖了这些语义。这可能是将来的研究方向。


**1. Introduction**

深度学习目睹了一系列capacity和capability增长的模型架构的诞生。这样越来越大的模型就需要越来越多的数据，特别是带有标签的数据。

这个问题最近在NLP领域被self-supervised pre-training方法的提出而解决了。他们的方法，基于GPT里的autoaggressive语言模型以及BERT里的masked autoencoding，实际上概念上很简单：将原数据一部分移除掉，然后让模型来学习预测这部分被移除掉的数据。这些方法现在让训练有数千亿参数的可以泛化的NLP模型都变得可行了。

masked autoencoders，是一种更加general的denosing autoencoders（在图片中人为添加噪音，然后通过去噪的方式来对图片的特征进行更好的提取），这个想法在CV领域也不那么新鲜。BERT在NLP领域获得了很大的成功，但是autoencoding方法在视觉领域的应用距离NLP里的成功还很远。于是我们提出了问题：是什么使得在vision和language里使用maksed autoencoding不一样呢？我们希望能从以下几个角度来回答。

1）直到最近，NLP和CV的模型结构都不一样。在视觉领域，CNN是最主流的模型。卷积是在regular grids上操作的，而且将mask tokens或者positional embeddings这种indicators加进去不是那么的容易。而这种模型结构上的差异，在ViT提出之后就不是问题了。

2）language和vision之间的information density不一样。languages是人类产生的信号，其具有高度的语义性，而且information-dense。让一个模型对一个句子只需要能够预测若干个缺失的单词，这个模型就需要有很复杂的language理解能力。但images是natural signals，而且有着很大的spatial redundancy，比如说，一个缺失的patch可以只需要有一些高层的很简单的信息就可以被其领域的patches重构回来。为了克服这个不同点，并且鼓励模型学习有用的features，我们说明一个很简单的策略在CV里很有效：随机遮挡很高比例的patches。这个策略很大程度上减少了redundancy，而且构造了一个具有挑战性的self-supervisory任务，其需要对图片有整体性的理解而不仅仅是low-level的image statistics。从之前的fig 2-4可以看出MAE的重构效果。

3）autoencoder的decoder，其的作用是将encoder给出的latent reprensentation再映射回输入，在language和image两个领域的重构任务也不一样。在视觉领域，decoder重构的是像素，所以其输出是低语义层次的，比普通的识别任务输出的语义层次要低。而对于language，deocder预测的是缺失的words，其含有丰富的语义信息。在BERT里，decoder可以随便选（BERT里就是一个MLP），但是在images领域，decoder的设计决定了encoder所学习到的latent representation的语义层次的高度。

由这些想法所启发，我们为visual representation learning提出了MAE这个模型。我们的MAE对于输入的图片随机遮盖patches，并且在像素空间内重构这些被遮盖的patches。MAE有一个非对称的encoder-decoder的设计。我们的encoder仅仅在那些可见的patches上操作（并没有mask tokens），而decoder是轻量级的，从encoder给出的latent representation和mask tokens一起来重构输入，如fig1所示。这样的设计大大减小了模型的计算量。在这种设计下，一个非常高比例遮挡的策略可以获得双赢：不仅让encoder能学习到更有效地representation，其还只需要encode一小部分patches。这可以大大减小训练的时间以及训练占用的内存，从而可以使MAE方法scale到更大规模的模型上。

我们的MAE学习到泛化效果很好的高capacity的模型。使用MAE pre-training，我们可以仅仅使用ImageNet-1K数据集就训练那些十分需要数据的模型，比如说ViT-Large/-Huage（[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://openreview.net/pdf?id=YicbFdNTTy)），并且泛化效果还更好。我们同时还在object detection，instance segmentation，semantic segmentation等后续任务上测试了MAE在transfer learning领域的效果。都要比利用其它的方法包括监督方法进行预训练的效果要好。这些结论和在NLP里发现的结果差不多，所以希望autoencoder在CV领域也能有在NLP领域那样的发展。


**2. Related works**

*Masked language modeling*

Masked语言模型和它们的autoregressive counterparts，比如BERT和GPT，在NLP领域是非常成功的pre-training模型。这些方法遮住输入sequence的一部分然后训练模型来预测这部分。这些方法的scaliability也很好，而且这些模型产生的预训练的representations对于下游任务来说泛化效果也很好。


*Autoencoding*

autoencoding是一个学习representations的经典方法。其有一个encoder来将输入映射到一个latent representation上，并有一个decoder来重构这个输入。比如说，PCA和k-means就是autoencoders。Denoising autoencoders（DAE）是一类autoencoders，其先给输入加上噪音，再学习如何重构原始的，没有被污染的输入。一系列方法都可以看作使用不同的噪音的DAE，比如，masking pixels，或者移除color channels等。我们的MAE也是一种DAE，但是和经典的DAE有着很大的不同。


*Masked Image Encoding*

masked image encoding方法通过遮盖的方式污染图片，然后再来学习图片的representations。最早的工作是将masking当作一种噪声，所以是DAE的一种特殊情况。context encoder使用CNN来学习那些缺失的部分。由NLP里的成功所启发，最近的方法都基于Transformers。iGPT在pixels构成的sequence上操作，像GPT一样预测被遮住的pixels。


*Self-supervised Learning*

非监督学习的方法在CV领域非常火热，通常集中研究于各种pre-training的任务。最近，contrastive learning非常的流行，[A Simple Framework for Contrastive Learning of Visual Representations](http://proceedings.mlr.press/v119/chen20j.html)，[Momentum Contrast for Unsupervised Visual Representation Learning](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)，[Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf?fbclid=IwAR2G_jEkb54YSIvN0uY7JbW9kfhogUq9KhKrmHuXPi34KYOE8L5LD1RGPTo)，[Unsupervised Feature Learning via Non-Parametric Instance Discrimination](https://openaccess.thecvf.com/content_cvpr_2018/html/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.html)，其在两张或者多张图片间对相似性进行表示。contrastive learning的方法太过于依赖data augmentation。autoencoding采用了概念上不同的另一个方向，也显示出了不一样的效果。


**3. Approach**

我们的MAE是一个简单的autoencoding的方法，其在给定原输入部分的patches的情况下重构原输入。和所有的autoencoders一样，我们的方法有一个encoder来将观察到的数据映射到一个latent representation，也有一个decoder来从这个latent representation重构原始的输入。和经典的autoencoders不一样，我们采用一种非对称的设计，encoder只在那些没有被遮住的patches上操作，不考虑mask tokens，并且使用一个轻量化的decoder，从mask tokens和latent representations来重构原始的输入。fig1描述了整个流程。

*Masking*

和ViT里的方法一样，我们先将图片分成regular的不重合的patches。然后我们随机挑选出一些patches，然后把剩下的都mask掉。我们的patch采样方式很简单：不重复的随机挑选，使用的是均匀分布（也就是每个patch被选中的概率相等）。

具有很高的遮挡比例的随机采样很大程度上消去了图片的redundancy，因此创建了一个任务，其不能够简单的只是通过领域patches插值就能解决。我们采用的是uniform distribution，其可以避免潜在的center bias（也就是更倾向于遮住图像中心的区域）。而且，这样很高比例的遮挡可以使我们能够scale我们的encoder到很大的模型，因为输入给encoder的计算量大大减小了。

*MAE encoder*

我们的encoder就是一个ViT，但是输入仅仅是那些没有被遮住的patches。正如在标准的ViT模型里一样，我们的encoder通过一个线性projection加上positional embeddings来embed patches，然后再用一系列Transformer模块来处理。然而，我们的encoder只需要在一小部分的输入图片patches上进行计算，而且并没有使用mask tokens。这使得我们训练很大的模型和很大的数据集成为可能。

*MAE decoder*

MAE的decoder的输入是full set of tokens，包括（1）encoded没被遮挡的patches；（2）mask tokens。由fig1可见。每个mask token是一个shared，learned向量，也就是每个被遮住的patch都用同一个向量来表示，而这个向量的值也是通过学习得到的。我们给这个集合里所有的tokens都加入了positional encodings；如果没有这个，mask tokens就没有它们在image中的位置信息了。decoder也有另外一些Transformer blocks。

MAE的decoder只用于在pre-training的时候做image reconstruction任务，只有encoder在生成后续所用的representation时会被用到。因此，decoder的结构可以被灵活的设计，和encoder的结构可以没有关系。

*Reconstruction target*

我们的MAE通过预测每个masked patch的像素值来重构输入。decoder的输出的每个element都是一个向量的像素值，表示的是一个patch。decoder的最后一层是一个linear projection，其输出的通道数等于每个patch里像素的个数。decoder的输出被reshape成为一个重构的image。我们的loss计算的是重构的image和原始输入之间的MSE。

我们还研究了另一个reconstruction target，也就是每个masked patch的normalized像素值。也就是说，我们计算一个patch里的所有像素的mean和deviation，然后normalize这个patch。使用normalized pixels能够改善encoder所学习到的representation的质量。

*Simple implementation*

我们的MAE pre-training可以被高效的部署。首先，我们对于每个input patch都生成一个token（这个token也就是通过linear projection和一个added positional embedding得到的）。然后我们随机打乱tokens再排成一列，直接去除掉这列后面那部分tokens（去除比例就是masking ratio）。这个过程为encoder产生了一小部分tokens（也就是没被遮住的那部分），也就等价于不重复的随机挑选patches。在encoding之后，我们将encoded patches构成的list通过加上mask tokens来延长，然后再unshuffle整个list（因为之前为了随机选patches而shuffle了，现在逆过来这个过程），从而tokens就对应了它们正确的位置。decoder被应用到这个list上（加上了positional embeddings）。

**4. The writing style of this paper**

这篇文章的introduction很长。第一是因为他使用了大量的图片，这是好事。第二个原因是因为他使用了一种问问题，回答问题，再引出论文想法的写法，这种写法是很好的。如果不这么写的话，那就是将abstract进行扩充，那其实最后就得到了这里introduction最后的两段话。因为这篇文章的技术细节并不复杂，所以作者采用了一种回到更本质的问题的方法，也就是将Bert从NLP用到CV的时候会有什么问题，然后再一一回答，之后引出MAE这个算法。也就是将这个算法为什么要设计成这样说的很清楚。这种写法是很推荐的。因为论文除了要讲清楚怎么做的，还要讲清楚为什么要这么做，也就是motivation。

但是在related work部分，虽然提到了很多相关工作，但是对于iGPT和Beit这两篇相关性最大的论文，并没有详细介绍他们的方法，这是不推荐的。




## Generative Models

### 1. [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*

*NIPS 2014*

**1. Abstract**

1. The abstract of this paper is very standard and unique. This abstract generally introduces the main concept of this paper, i.e., GAN, rather than introducing the problem and existing methods. This abstract is kind of like the Wikipedia introduction of some concept. If you are very confident that your paper is very novel and proposes a very useful idea, model or concept that can be recorded in this area, you can use this kind of writing style of abstract. This is very clear to the community, but not very clear to the people that do not know any about this area.

**2. Introduction**

2. This paper thinks that besides deep learning architectures, deep learning will learn the representations of data probability distributions over all kinds of data, including natural images, audio waveforms containing speech, symbols in natural language processing, etc. This idea is always held by Yoshua Bengio and his group.

3. Discriminative models in deep learning have already invovled into various kinds and solved many kinds of problems. But generative models, instead, still have lots of part remained mysterious. This due to the fact that in order to learn data distributions underlying generative models, you need many approximation methods to approximate the distribution in order to make calculation work. But this process will make the distribution unaccurate and even not working. So in this paper, they do not try to approximate the distributions, they use deep learning models to do this job.

4. Generative adversarial models will contain two parts: the generative model and the discriminator model. The generative model aims to generate data that the discriminator model can not distinguish apart from true training data, and the discriminator model aims to distinguish between them. The final goal is to let the generative model to generative plausible data that the distriminator model can not distinguish.

5. In this paper, they use MLP as models for the generative and discriminator models, and in this case, this generative adversarial model is called generative adversarial net (GAN). The input of the generative model is random noise (usually a Guassion noise), and they want the generative model to map this random noise distribution to any distribution we want to fitting, i.e., the training data distribution.


**3. Related work**

1. There are two kinds of former works on deep generative models. The first ones concentrated on building a probability distribution function, and these models are trained on maximizing the log likelihood, such as the deep Boltzmann machine. The biggest difficulty is that the calculation of sampling this distribution is hard, especially when the dimension is high. The other ones are called generative machines. They do not explicity construct the distribution, and they learn a model to approximate this distribution. There are intrinsic differences between these two kinds of methods. The first ones acutally learn the distribution, though using some kind of approximation method to make the calculation feasible, but in the end, you actually get a distribution, you can calculate the mean, variance and all kind of properties of this distirbution. But the other ones do not construct the distribution, and only learn a model to approximate this distribution. So in the end, we do not know what this distribution looks like. 

2. Variational Autoencoder (VAE) actually has similar ideas to this paper. And using a distriminator model to assist the generative model is also not novel, such as Noise-contrastive Estimation (NCE). 


**4. Adversarial nets**

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


**4. Theoretical Results**

1. There is a global optimum for the generator, $$p_g = p_{data}$$. Firstly, we see a lemma firstly. **Lemma**: if the generator is fixed, then the optimal discriminator will be

$$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$

i.e., the error probability (the training criterion of discriminator) of the distriminator will be the smallest. **Theorem**: Setting $$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$ as in Lemma in the equation $$\min_{G}\max_{D} V(D,G) = E_{x \sim p_{data}}\left[log D(x)\right] + E_{z \sim p_z(z)}\left[log(1-D(G(z)))\right]$$, we can get $$C(G) = E_{x \sim p_{data}}\left[log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + E_{x \sim p_g}\left[log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right] $$. Then $$C(G)$$ get its minimum when $$p_g = p_{data}$$.

2. The algorithm described above is able to train the discriminator and the generator, i.e., the training algorithm is convergent. 


**5. Experiments**

The experiments in this paper is not good enough and quite simple.


**6. Conclusion**

GAN is actually an unsupervised learning setting, but it leverages supervised learning framework by using the label of true or generated data for training. This idea insights the future self-supervised learning frameworks.


**The conclusion of writing style of this paper**

This paper proposes a very novel idea and model, thus it elaborates the details of design and ideas behind GAN very clearly. The authors are very ambitious and are confident that this work will be recorded in this area. So the whole writing style is kind of like Wikipedia, i.e., the very detailed description of the proposed model, with a little mention of existing works and comparison with other works. Also, experiments are few.

If you are confident that your work is novel and very important, you can use this kind of writing style. Otherwise, you need to describe clearly the existing works and your contribution to this problem.



---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

