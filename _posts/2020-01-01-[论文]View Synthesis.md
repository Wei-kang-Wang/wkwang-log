---
layout: post
comments: false
title: "[论文]View Synthesis"
date: 2020-01-01 01:09:00
tags: paper-reading
---

> This post is a summary of synthesis related papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## 3D View Synthesis

[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392.pdf)

*Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*

*ECCV*

我们提出一个方法，使用一系列input views作为输入，通过优化一个潜在的continuous volumetric scene函数来对复杂场景生成新的角度的views。我们的算法使用一个MLP（而不是CNN）来表示一个scene，其输入是一个连续的5D coordinate（空间位置$$(x,y,z)$$以及view的方向$$(\theta, \phi)$$），输出是volume density和那个位置$$(x,y,z)$$的view dependent emitted radiance。我们通过沿着相机rays来查找5D coordinate，并使用经典的volume rendering方法来将网络输出的color和density投射到一张照片上。因为volume rendering本身就是differentiable的，所以我们唯一需要的输入就是一系列已知相机姿态的不同角度的照片，也就是一个场景不同角度的views。我们描述如何高效的优化neural radiance fields来渲染有很复杂geometry和appearance的场景的新的角度的photorealistic的图像，并且展示了效果要比之前的neural rendering和view synthesis的工作的效果要好很多。view synthesis的结果最好用视频来看，在补充材料里有。


这篇文章用一种全新的方法来解决view synthesis这个被研究了很久的问题，方法是，通过直接优化一个连续的5D scene representation的参数来最小化渲染images的error。

我们将一个静态的场景表示为一个连续的5D函数，输出空间中每个点$$(x,y,z)$$在每个角度$$(\theta, \phi)$$情况下的emitted radiance，以及一个density。这个density作为一个可微分的opacity存在，其控制着一条射线穿过$$(x,y,z)$$这个点时，需要有多少的radiance被考虑进去。我们的方法优化一个MLP（没有任何卷积）来表示上述这个5D的function，其输入为一个5D coordinate $$(x,y,z,\theta,\phi)$$，输出为一个volume density和view-dependent RGB color，所以是一个regression问题。



















---
