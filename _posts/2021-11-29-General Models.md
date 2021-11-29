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

Before AlexNet, the machine learning community mostly concentrated on classical models, including SVM, decision trees, etc. The neural networks had no advantages over these classical models but had less elegant mathematical principles compared to them. Thus for supervised learning tasks, such as image classification, the neural network is not a good candidate. And many efforts has been put on the unsupervised learning tasks for neural networks, because after all, neural networks have better performances over classical models in this area. But the AlexNet shows that even without unsupervised learning, this deep convolutional neural network can have much better result on the ImageNet Classification Task. This is a milestone for supervised learning, computer vision, and machine learning.

**But recently researches show that unsupervised learning still remains much more mysteries and there are many researches on it. Yan LeCun even thinks that it's the main future direction for the whole machine learning community in a talk in 2020**

The first worth noting point is the model does not requires fixed size of iuput image. It will down sample the images to a fixed resolution of $$256 \times 256$$. Given a rectangular image, the model first scale the image with the shorter side of length 256, and then crop out the central $$256 \times 256$$ patch from the resulting image. Also there are no pre-processing techniques, the network is trained on the centered raw RGB values of the pixels. This is a big inprovement then, since the existing works at that time always use some feature extractor to get useful features, such as SIFT, and then do the classification task. **This is so-called End-to-End**.

Another point is the authors use ReLU rather than tanh or sigmoid function as the non-linearity function of the model. They argue that ReLU is a non-saturating function, which is much better than those saturating functions since it can help the model to learn much faster. (An activation function is considered non-satured if $$lim_{x \rightarrow \infty} f(x) = \infty$$. A saturated activation function has a compact range such as $$\[-1,1]$$ for tanh or $$\[0,1]$$ for the sigmoid.) But from today's viewpoint, this is not the reason, or at least not the only reason that why ReLU are much more faster, but the biggest advantage of ReLU is that it's simple enough, thus it's the most prevalent non linear function now in neural networks.

One big result is that the features from the last hidden layer really learn the semantic information. The authors show this result by first taking a random image, and then find six other images whose last layer features has the smallest Euclidean distance to the picked one's. This result shows that the feature vectors from the last hidden layer of this model have good semantic representation of this supervised learning task. The feature vectors space have a good semantic explanation of the original task, i.e., semantic similar images have distance close feature vectors.

![Comparison based on features]({{ '/assets/images/AlexNet-1.PNG' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 2. Five ILSVRC-2010 test images in the first column. The remaining columns show the six training images that produce feature vectors in the last hidden layer with the smallest Euclidean distance from the feature vector for the test image.*


## Natural Language Processing

## Generative Models


## Deep Learning Models


---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

