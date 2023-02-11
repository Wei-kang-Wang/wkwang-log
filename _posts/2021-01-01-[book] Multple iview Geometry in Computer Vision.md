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

>2维射影空间指的是对应的欧式空间是2维的，而实际上2维射影空间里的点是3维的，因为最后一个维度是1。而无穷远处的点，最后一个维度是0，但其只有一个自由度，所以相当于$$(x,1,0)$$，也就是构成了一条线。

**Homogeneity**

在欧氏几何里，所有的点都是同等的，也就是说整个空间是homogeneous的（同质的）。当我们引入坐标系的时候，就需要挑选一个特殊的点作为坐标原点，从而建立坐标系。而实际上这个点是可以任意挑选的，而坐标系也是可以任意建立的。任意两个坐标系可以通过平移旋转而重合。这样的操作就叫做Euclidean transform（欧式变换）。

一个比欧式变换更加general的变换是先对$$\mathbb{R}^n$$施加一个linear transformation，然后再进行一个欧式变换。我们可以将上述过程理解为先平移空间，再旋转空间，之后再沿着不同的方向进行延展（stretching）。这样的变换被称为affine transformation（仿射变换）。

不管是欧式变换还是仿射变换，在无穷远处的点仍然还在无穷远处，也就是说它们在欧式变换以及仿射变换下是perserved的。所以说它们在这两种变换下是特殊的。

但在射影几何的观点下，无穷远处的点和其它位置的点并没有什么不同，所有的点（包括无穷远处的点）都是同等的。正如在欧氏空间里，欧式变化可以用一个矩阵来表示，我们也可以用一个非奇异的矩阵来表示一个射影变换，其将一个$$n+1$$的向量转变为另一个$$n+1$$的向量，这个射影变换矩阵的大小为$$(n+1) \times (n+1)$$。而最后一个坐标为0的点（无穷远点）也被转换到了其它的点，也就是说无穷远点并没有被perserved。

在CV问题里，projective space可以被用来表示真实的3D空间，也就是将一个3D的欧氏空间拓展为一个3D的projective space（里面的向量长度为4）。相似的，图片一般被认为是3D空间在2D空间内的投影，同样也被在2D projective space里考虑。

### 1.1.1 Affine and Euclidean Geometry

之前已经说了2维projective space可以通过在2维Euclidean space里添加一个无穷远的line来得到（3D就是添加一个无穷远的plane，以此类推）。我们现在来考虑如何从一个projective space回到一个Euclidean space。如下主要考虑2维和3维的情况。

**Affine geometry**
首先我们知道projective space是homogeneous的，也就是任意的坐标系的建立都不会有影响。而且在2维projective space里没有平行线的概念，因为所有的平行线都会在无穷远点相交（3维projective space里没有平行平面的概念，以此类推）。然而，在projective space里，也并不认为那些无穷远点就特殊，我们认为所有的这些点都是同等的。也就是说，在projective geometry里，讨论欧氏空间里的那种平行性是没有意义的，这个概念在projective geometry里不存在，所以我们需要定义一种在射影空间里的平行性。

也就是说，我们需要选出某条特殊的线，然后决定这条线是不是就是无穷远处的那条线（对于2维射影空间来说的）。这就造成了一个状态，也就是说尽管所有的点都是同等的，那还是会有一些点要比其它的点更加同等（也就是构成无穷远处直线的那些点，因为无穷远处的点都应该是完全等价的）。我们如下来考虑一个思想实验。从一张A4纸出发（有界限的2维欧氏空间），然后将其延展到无限远处，也就是说构造了一个2维的projective space $$\mathbb{P}^2$$，而现在这张纸就可以表示这个2维projective space的一部分。然后，在纸上画一条直线，我们认为它就是line at infinity。然后再画两条直线与之相交，因为这两条直线与ilne at infinity相交，我们就可以认为它两是平行的。考虑一个非常平的平原，然后用一台照相机来拍照，照相机的image plane是垂直于平地的，将这块平地叫做world plane，那么world plane的无穷远处的点在相机平面上就是一条水平的线，在相机前面的所有的world plane上的点都在image上有唯一的对应，而image上还会有above world plane的点，比如说天空，但将这些点和相机光心的连线再延长，就会与相机背后的world plane有一个交点，从而image plane上的点和world plane上的点一一对应（假设相机的角度是180度，world plane是无穷大的）。从而world plane里无穷远处的点对应到image plane里的一条水平的线，world plane上平行的线（比如说铁轨）对应到image plane里两条和水平线有交点的线。实际上，world plane和image plane只是描述一个projective plane的两种不同的方法。projective plane和一条指定的line（也就是表示无穷远处的那条线）的geometry就叫做affine geometry，将某个projective space里的指定的line映射到另一个projective space里的指定的line的projective变换就叫做affine transformation。

>这是另一种理解affine transformation的办法。

通过定义了projective space里的line at infinity，我们就可以定义projective space里的平行线了。而且一旦有了平行性，我们就可以定义一些其它的相关概念。比如说如果$$A,B,C,D$$是projective space里的四个点，$$AC$$和$$BD$$平行，$$AB$$和$$CD$$平行，那么$$AB$$和$$CD$$的长度就相等。

**Euclidean geometry**

在projective plane里定义了line at infinity之后，我们就有了平行性，并且有了affine geometry。affine geometry就是针对projective plane的一个规范，其会在plane中按照某种规则挑选出一条线作为line of infinity。

如果我们选择了一些特殊的line作为line at infinity，那么affine geometry就会退化为Euclidean geometry。为了实现上述这种特殊的挑选，我们需要介绍一个本书最重要的概念：absolute conic。

我们从2维geometry开始，先考虑圆的情况。注意到，圆在affine geometry里是没有意义的，因为任意的affine transformation，就会将圆变为椭圆。因此，affine geometry认为圆和椭圆是同一种概念。

在Euclidean geometry里，圆和椭圆当然是不同的，而且有着非常不同的性质。从代数角度来说，一个椭圆可以被一个有两个变量的等式表示。因此两个椭圆最多可以有四个交点。然而，两个不同的圆最多只有两个交点。但是从代数的角度来说，描述圆的等式依然具有两个变量，而求交点的过程就是联立解两个2元2次方程组的过程，按道理应该有四个解，那是什么让圆变得特殊呢？实际上是因为方程的另外两个根是复数，所以就被舍去了。

对于projective space来说，其坐标是homogeneous坐标，表示为$$(x,y,w)$$，而在projective space里，圆的方程是：$$(x-aw)^2 + (y-bw)^2 = r^2 w^2$$。其表示的是一个在projective space的homogenous coordinates下的中心点为$$(x_0,y_0,w_0)^T = (a,b,1)^T$$的圆。可以验证$$(x,y,w)^T = (1, i, 0)^T$$以及$$(1,-i,0)^T$$都是圆上的点。而且我们会发现，这两个点和$$a,b$$无关，也就是说所有的圆都经过这两个点，也就是说这两个点是任意两个圆的交点。因为这两个点最后一个坐标是0，其位于line at infinity上。这两个点叫做这个plane的circular points。尽管这两个点是复数，其满足一对实数定义的等式：$$x^2 + y^2 = 0, w=0$$。

上述的结果给了我们如何从projective geometry定义Euclidean geometry的启示。对于一个projective geometry，我们挑出line at infinity，并在这条line上挑出两个circular points，就可以构造一个Euclidean geometry。虽然circular points是复数，但我们一般不需要去担心这个问题。现在，我们可以定义圆为任意的经过这两个circular points的conic（具有两个变量的2维曲线，也就是$$ax^2 + b y^2 + c xy + dx + ey + f = 0$$)。

>对于标准的Euclidean coordinate system，circular points的坐标是$$(1,i,0)^T$$和$$(1,-i,0)^T$$。而如果我们已经有了一个Euclidian space，只需要选出一条线作为line at infinity，再在这条线上任意选出两个complex points作为circular points，就可以得到一个projective plane。

>如何理解任意conic经过这两个circular points就是圆？因为任意的conic由五个点确定，而任意的圆由三个点确定，那剩下的两个点正好就是circular points。

angle、length ratios等概念也可以由ciruclar points给出，这在后续的章节里会说。

**3D Euclidean geometry**

我们已经看到了如何从一个projective space里定义一个Euclidean space：只需要指定line at infinity以及circular points就可以。同样的概念也可以被用于3D geometry里。正如2维里的情况一样，这时我们来考虑3维里的球，以及它们是如何相交的。在Euclidean space里，两个球相交的结果是一个圆，但两个general的quadric surfaces相交的结果并不是这样（从代数上看）。这是因为，在projective plane里，也就是在homogeneous coordinate $$(X,Y,Z,T)^T$$的角度来看，任意的球都和平面有如下的curve交线：$$X^2 + Y^2 + Z^2 = 0, T=0$$。这是一个位于infinity的second-order的曲线（也就是一个conic），仅仅由复数点构成。这个曲线就叫做absolute conic，这是本书最重要的内容，其与相机内部参数的计算有关，将在后面被详细说明。

同样的，从一个projective space，我们可以指定任意一个平面为plane at infinity，并在平面上指定任意一个曲线为absolute conic，这样就将一个projective space转换为了一个Euclidean space。


## 1.2 Camera projections














        
        


                


















































---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
