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

### AlexNet [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
*ALex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton* 
*NIPS 2012*

Before AlexNet, the machine learning community mostly concentrated on classical models, including SVM, decision trees, etc. The neural networks had no advantages over these classical models but had less elegant mathematical principles compared to them. Thus for supervised learning tasks, such as image classification, the neural network is not a good candidate. And many efforts has been put on the unsupervised learning tasks for neural networks, because after all, neural networks have better performances over classical models in this area. But the AlexNet shows that even without unsupervised learning, this deep convolutional neural network can have much better result on the ImageNet Classification Task. This is a milestone for supervised learning, computer vision, and machine learning.

**But recently researches show that unsupervised learning still remains much more mysteries and there are many researches on it. Yan LeCun even thinks that it's the main feature direction for the whole machine learning community in a talk in 2020**

One big result is that the features from the last hidden layer really learn the semantic information. The authors show this result by first taking a random image, and then find six other images whose last layer features has the smallest Euclidean distance to the picked one's.
![Comparison based on features]({{ '/assets/images/AlexNet-1.PNG' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 2. Five ILSVRC-2010 test images in the first column. The remaining columns show the six training images that produce feature vectors in the last hidden layer with the smallest Euclidean distance from the feature vector for the test image.*


## Natural Language Processing

## Generative Models


## Deep Learning Models


---

If you notice mistakes and errors in this post, don't hesitate to contact me at *wkwang0916 at outlook dot com* and I would be super happy to correct them right away!

