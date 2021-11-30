---
layout: post
comments: false
title: "General Models"
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of general model structures, including computer vision, natural language processing, generative models, etc.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## Computer Vision

### AlexNet: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
*ALex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton* 

*NIPS 2012*

The model in this paper won the ILSVRC 2012 competition classification task.

Before AlexNet, the machine learning community mostly concentrated on classical models, including SVM, decision trees, etc. The neural networks had no advantages over these classical models but had less elegant mathematical principles compared to them. Thus for supervised learning tasks, such as image classification, the neural network is not a good candidate. And many efforts has been put on the unsupervised learning tasks for neural networks, because after all, neural networks have better performances over classical models in this area. But the AlexNet shows that even without unsupervised learning, this deep convolutional neural network can have much better result on the ImageNet Classification Task. This is a milestone for supervised learning, computer vision, and machine learning.

**But recently researches show that unsupervised learning still remains much more mysteries and there are many researches on it. Yan LeCun even thinks that it's the main future direction for the whole machine learning community in a talk in 2020**

The first worth noting point is the model does not requires fixed size of iuput image. It will down sample the images to a fixed resolution of $$256 \times 256$$. Given a rectangular image, the model first scale the image with the shorter side of length 256, and then crop out the central $$256 \times 256$$ patch from the resulting image. Also there are no pre-processing techniques, the network is trained on the centered raw RGB values of the pixels. This is a big inprovement then, since the existing works at that time always use some feature extractor to get useful features, such as SIFT, and then do the classification task. **This is so-called End-to-End**.

Another point is the authors use ReLU rather than tanh or sigmoid function as the non-linearity function of the model. They argue that ReLU is a non-saturating function, which is much better than those saturating functions since it can help the model to learn much faster. (An activation function is considered non-satured if $$lim_{x \rightarrow \infty} f(x) = \infty$$. A saturated activation function has a compact range such as \[-1,1] for tanh or \[0,1] for the sigmoid.) But from today's viewpoint, this is not the reason, or at least not the only reason that why ReLU are much more faster, but the biggest advantage of ReLU is that it's simple enough, thus it's the most prevalent non linear function now in neural networks.

![Model Structure]({{ '/assets/images/AlexNet-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Model Structure.*

**The model takes an RGB image as input, which is a human aware data, to vector, which is a computer aware data. The whole process can be understood as a knowledge compression. This is the key of deep neural networks.**

One big result is that the features from the last hidden layer really learn the semantic information. The authors show this result by first taking a random image, and then find six other images whose last layer features has the smallest Euclidean distance to the picked one's. This result shows that the feature vectors from the last hidden layer of this model have good semantic representation of this supervised learning task. The feature vectors space have a good semantic explanation of the original task, i.e., semantic similar images have distance close feature vectors.

![Comparison based on features]({{ '/assets/images/AlexNet-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. Five ILSVRC-2010 test images in the first column. The remaining columns show the six training images that produce feature vectors in the last hidden layer with the smallest Euclidean distance from the feature vector for the test image.*

Some techniques used in these papers are prevalent in the following works, including dropout, weight decay, momentum, SGD algorithm, parameter initialization, learning rate. There are several methods of setting learning rate. In this paper, they first set a number, and then manually divide this number by 10 if the training error becomes flat. Other methods including every fixed number of epochs, dividing the learning rate by 10, or using some smooth function to define the learning rate, or firstly raising and then reducing the learning rate, etc. (*one epoch means one cycle of the whole dataset.*)


### ResNet: [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*

*CVPR 2016*

**AlexNet(2012) $$\rightarrow$$ ResNet(2016)**

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

Suppose $$g(x)$$ is a neural network. Adding more layers to this net will make the model function become $$f(g(x))$$. $$\fraction{d f(g(x))}{d x} = \frac{d f(g(x))}{d g(x)} \frac{d g(x)}{d x}$$ is the gradients of the new model and $$\frac{d g(x)}{d x}$$ is the original one's. Gradients are always be quite small, thus the multplication will make the gradients of the new model be much smaller than the original one's, thus the training of deep model is very hard.

But if we use the ResNet structure in this paper, then our new deep counterpart of $$g(x)$$ becomes $$f(g(x))+g(x)$$, and the gradient becomes $$\frac{d f(g(x))}{d g(x)} \frac{d g(x)}{d x} + \frac{d g(x)}{d x}$$, which is comparable to the original one's.

The key idea is: **Always make the gradients be large enough, and then your model can be trained well!**

## Natural Language Processing

## Generative Models

---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

