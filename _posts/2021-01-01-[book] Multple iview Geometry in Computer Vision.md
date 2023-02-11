---
layout: post
comments: false
title: "Multiple view Geometry"
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

这一章主要介绍这本书所包含的主要的思路和内容。这一章将会给出这些内容的一个informal的介绍。准确无误的定义和数学描述将会在第二章之后的章节里陆续介绍。

## 1.1 Introduction - the ubiquitous projective geometry (普遍存在的射影几何)

我们所有人都对射影变换很了解。当我们看一张图片的时候，我们将会看到那些本该是方形的东西变得不够方形，本来是是圆形的东西也不圆了，但我们是能够从这些“不准确”的图片内容知道真实的物体的样子的。那些将真实的物体映射到图片上的变换就是射影变换的一个例子。

很重要的一个问题是是：通过射影变换之后，有哪些几何性质仍然被保留了下来？显然，shape不是，因为一个圆形的物体现在已经变成了一个椭圆形。lengths也不是，因为平行线也变得不再平行。angles，distance，distances的ratios，这些都不是。看起来，射影变换破坏了大部分的几何信息，然而有一个几何信息被保留了下来：straightness（也就是直线在映射后仍然是直线，曲线在映射后仍然是曲线）。之后我们将会发现这是很重要的一个特性，我们可以通过保留点与点之间的线的straightness来定义映射变换。

为了说明我们为何需要射影几何，我们可以先来看看欧氏几何。欧氏几何描述了角度以及物体的shapes。欧氏几何有个不好解决的问题就是：平行线永远不相交。我们可以在欧氏平面上添加一个理想中的无穷远点，认为所有的平行线都相交于这个点，从而将这个欧氏空间转换为一个新的几何空间，就叫做射影空间。这样做的好处是，我们可以将我们熟知的欧氏空间里的知识都用于这个新的射影空间里，只需要注意一点，就是那个新添加的无穷远的点就可以了。

**Coordinates**

2维欧氏空间里的一个点是用一个有序的实数对$$(x,y)$$来表示的。我们可以添加一个多余的坐标，就得到了一个triple，$$(x,y,1)$$，我们认为其表示的是同一个点。这是一一对应的，所以可以来回转换。但现在有个很重要的问题是是，为什么添加的新的坐标就得是1呢，因为另外两个坐标$$x,y$$也并没有任何的限制？那如果新定义的triple是$$(x,y,2)$$又会如何呢？实际上，我们这里认为$$(x,y,1)$$和$$(2x,2y,2)$$表示的是同一个点，也就是说，这个坐标的限制由最后一个index来表示，所有的$$(kx,ky,k)$$都表示的同一个点（$$k$$不能是0）。从数学上说，点被coordinate triple的equivalence class来表示，当两个triple成比例的时候就表示的同一个点。这个时候系统就不再是invertible而是surjective了，因为欧氏几何里的一个$$(x,y)$$可以对应无数个$$(kx,ky,k)$$，但任何一个$$(kx,ky,k)$$只能对应一个$$(x,y)$$。

尽管所有的$$(kx,ky,k)$$都可以有唯一一个$$(x,y)$$与之对应，但$$(x,y,0)$$就没有对应点了，这就是我们的射影空间里引入的无穷远点。

上述步骤就是将一个2维欧氏空间，拓展为一个射影空间，将欧氏空间内的点，拓展为射影空间内的homogeneous vectors。而上述步骤对于任意维的欧氏空间都是一样的。欧式空间$$\mathbb{R}^n$$可以被拓展为射影空间$$\mathbb{P}^n$$，我们只需要将长度为$$n$$的欧氏空间内的点的坐标，拓展到射影空间内的长度为$$n+1$$的homogenous vector就行了。我们会发现，在2维欧式空间里的无穷远点，在2维射影空间内变成了一条线，一般被称为line at infinity（这条线的方程就是$$(x,y,0)$$）。而3维射影空间内，则对应的是plane at infinity。

**Homogeneity**

在欧氏几何里，所有的点都是同等的，也就是说整个空间是homogeneous的（同质的）。当我们引入坐标系的时候，就需要挑选一个特殊的点作为坐标原点，从而建立坐标系。而实际上这个点是可以任意挑选的，而坐标系也是可以任意建立的。任意两个坐标系可以通过平移旋转而重合。这样的操作就叫做Euclidean transform（欧式变换）。

一个比欧式变换更加general的变换是先对$$\mathbb{R}^n$$施加一个linear transformation，然后再进行一个欧式变换。我们可以将上述过程理解为先平移空间，再旋转空间，之后再沿着不同的方向进行延展（stretching）。这样的变换被称为affine transformation（仿射变换）。

不管是欧式变换还是仿射变换，在无穷远处的点仍然还在无穷远处，也就是说它们在欧式变换以及仿射变换下是perserved的。所以说它们在这两种变换下是特殊的。

但在射影几何的观点下，无穷远处的点和其它位置的点并没有什么不同，所有的点（包括无穷远处的点）都是同等的。正如在欧氏空间里，欧式变化可以用一个矩阵来表示，我们也可以用一个非奇异的矩阵来表示一个射影变换，其将一个$$n+1$$的向量转变为另一个$$n+1$$的向量，这个射影变换矩阵的大小为$$(n+1) \times (n+1)$$。而最后一个坐标为0的点（无穷远点）也被转换到了其它的点，也就是说无穷远点并没有被perserved。

在CV问题里，projective space可以被用来表示真实的3D空间，也就是将一个3D的欧氏空间拓展为一个3D的projective space（里面的向量长度为4）。相似的，图片一般被认为是3D空间在2D空间内的投影，同样也被在2D projective space里考虑。

### 1.1.1 Affine and Euclidean Geometry

之前已经说了2维projective space可以通过在2维Euclidean space里添加一个无穷远的line来得到（3D就是添加一个无穷远的plane，以此类推）。我们现在来考虑如何从一个projective space回到一个Euclidean space。如下主要考虑2维和3维的情况。

**Affine geometry**
首先我们知道projective space是homogeneous的，也就是任意的坐标系的建立都不会有影响。而且在2维projective space里没有平行线的概念，因为所有的平行线都会在无穷远点相交（3维projective space里没有平行平面的概念，以此类推）。然而，在projective space里，也并不认为那些无穷远点就特殊，我们认为所有的这些点都是同等的。也就是说，在projective geometry里，讨论平行性是没有意义的，这个概念在projective geometry里不存在。












        
        


                


















































---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
