---
layout: post
comments: false
title: "Multiview Geometry"
date: 2021-11-29 01:09:00
tags: book-reading
---

> 这是Andrew Zisserman的Multiple view Geometry in Computer Vision这本书的翻译版，加上了个人一些拙略的理解。


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

# 前言（Oliver Faugeras）

四十年前，让计算机能够see，仍然是一个未解决而且看起来很困难的问题。Computer Vision这个领域，作为一个独立的研究分支而成立，其与mathematics，computer science有着很强的联系，与physics，psychology以及neuron sciences也有着一些联系。

早期计算机视觉之所以没有成功的一个原因是，人们把生物视觉的形成原理考虑的太简单了。当然，完全依赖于生物视觉原理来开发计算机视觉算法是不合理的，但是我们不得不认清以下两个事实：

* 生物视觉的原理至今很大一部分是未知的，因此也不可能用计算机去模拟
* 尝试绕过生物视觉，而重新建立一种硅基视觉是很困难的

但尽管有着上述一些困难，计算机视觉的研究者们仍然做出了一些理论上和技术上都很值得铭记的结果。

在技术上来说，自动驾驶技术已经在欧洲，美国以及日本得到了应用。自动驾驶需要计算机能够具有实时分析三维场景的能力。如今，汽车制造商已经将自动驾驶融入了车型设计之中，这已经成为了一项逐渐成熟的技术。

从理论上说，geometric computer vision方面有一些很不错的结果。这个理论包括如何将物体appearance的变化表示为物体的shape和相机viewpoint的函数。这也是这本书将要描述的重点内容。


# 序

在过去的几十年里，关于如何理解和建模multiple views之间的geometry这个问题有了很多的发展。理论和实践如今都达到了一个成熟的高度，过去认为无法被解决的问题如今都有了较好的结果。这些结果包括：

* 给定同一个物体的两张不同角度的图片，没有别的额外信息，我们可以计算出这两张图片点之间的对应关系，并且得到这些点的3D坐标，以及这两张图片的viewpoint（也就是相机角度）
* 给定同一个物体的三张不同角度的图片，没有别的额外信息，我们可以计算出这些图片点和线之间的对应关系，并且得到这些点和线的3D坐标，以及这些图片的viewpoint（相机角度）
* 从一部相机拍摄的一系列图片中计算出此台相机的内部参数

上述这些算法的一个特点就是，它们都是uncalibrated的，也就是说，我们不需要知道相机的内部参数（比如说focal length等）就可以得到这些结果。

支撑这些算法的背后的理论是一个新的而且全面的对multiple uncalibrated views的geometry的一个理论。


# 阅读顺序

本书被分为了六个部分，有七个短的附录。每个部分都介绍了一种新的geometry relation：背景的homography，single view的camera matrix，two views的fundamental matrix，three views的trifocal tensor，four views的quadrifocal tensor。在每个部分里，都有一个章节描述geometry relation、性质和应用，以及一个章节来描述具体算法。

* Part 0. Background。这一部分主要来介绍2维和3维空间里的projective geometry的基础知识，以及这些geometry是如何被represented、manipulated以及estimated的。
* Part 1. Single view geometry。单个相机如何将3维空间的物体映射到2维照片的过程将会被描述出来（叫做perspective projection），其中包括各种相机模型。
* Part 2. Two view geometry。这一部分描述了两个相机的epipolar geometry、从图片点之间的对应关系来进行projective reconstruction、解决projective ambiguity的方法、optimal triangulation以及如何利用平面来在各个view之间进行转换。
* Part 3. Three view geometry。这一部分介绍了三个相机的trifocal geometry，包括如何将两张图片之间的点的对应关系转移到第三张图片，以及如何从点的对应关系转移到线的对应关系、如何从点和线的对应关系来计算geometry、如何计算相机内部参数。
* Part 4. N-Views。这一部分有两个主要内容。首先，其将three view geometry拓展到了four view geometry，并且描述了如何从N-views里获取geometry（比如从N张图片里得到structure和motion信息）。其次，其将前面所有的内容用一种更general的理论框架来描述一遍，从而使得整本书的理论方法显得更加的统一。
* Appendices。介绍了tensors、statistics、parameter estimation、linear and matrix algebra、iterative estimation、solution of sparse matrix systems以及special projective transformations的背景知识。


# Chapter 1. Introduction - a Tour of Multiple View Geometry














        
        


                


















































---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
