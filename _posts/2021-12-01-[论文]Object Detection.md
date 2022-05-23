---
layout: post
comments: false
title: "[论文]Object Detection"
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of object detection papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## [Object Detection in 20 years: A survey](https://arxiv.org/pdf/1905.05055.pdf?fbclid=IwAR0pD4GD6zW2n3G_lbYUDFn9wFL1ECuZHJfZc3-6-Xq_gYSO5xx2mgpKueA)

*Zhengxia Zou, Zhenwei Shi, Yuhong Guo, Jieping Ye*

*Arxiv 2019*

### Abstract

目标检测，作为CV领域最基础也是最具有挑战性的一个任务，在过去的很多年里受到了非常多的关注。其在过去二十年的发展可以被看作是CV历史的缩影。如果我们把现在的object detection任务看作是利用强大的deep learning的技术美学，那么将眼光放到二十年前，我们将会看到冷兵器时代的智慧。这篇文章十分广泛的包含了超过400篇object detection的文章，时间跨度从1990到2009，有1/4个世纪。这篇文章会包含很多的话题，包括历史上的milestone detectors，detection datasets，metrics，detection systems的基础的building blocks，speed up techniques，以及最近sota的detection methods。这篇文章同时也包含了一些重要的detection应用，比如说pedestrian detection，face detection，text detection等等，并且对它们的难点以及近些年技术上的突破进行了深度分析。


### 1. Introduction

object detection是一个重要的CV任务，其解决的是在图片里对于某个特定的类别的视觉目标的实例进行检测，类别包括humans，animals，cars等。object detection的目标是开发出模型来为CV应用提供一个最为基本的信息：what objects are where?

作为CV领域一个最基本的问题之一，object detection为很多其它的CV任务提供了基础，比如说instance segmentation，image captioning，object tracking等。从应用的角度来说，object detection可以被分为两种研究发现，general object detection和detection applications，前一种致力于模仿人类的视觉和意识来用一个普适的模型对不同的objects实例都可以进行检测，而后者则是致力于在某种特定的应用场景下的检测比如pedestrian detection，face detection，text detection等。在最近一些年里，deep learning的快速发展为object detection注入了新鲜的血液，带来了瞩目的成果并且将object detection推到了研究的热点。object detection现在在很多现实生活的应用中都得到了使用，比如说autonomous driving，robot vision，video surveillance等。Fig 1显示了在过去二十年间有关object detection的论文的数量，可以看到明显的增长趋势。






---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

