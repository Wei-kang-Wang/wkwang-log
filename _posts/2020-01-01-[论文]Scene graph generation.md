---
layout: post
comments: false
title: "[论文]Scene Graph Generation"
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of scene graph generation papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


### 1. [Learning Physical Graph Representations from Visual Scenes](https://neuroailab.github.io/physical-scene-graphs/)

[POST](https://neuroailab.github.io/physical-scene-graphs/)

*NeurIPS 2020*

*Daniel M. Bear, Chaofei Fan, Damian Mrowca, Yunzhu Li, Seth Alter, Aran Nayebi, Jeremy Schwartz, Li Fei-Fei, Jiajun Wu, Joshua B. Tenenbaum, Daniel L.K. Yamins*


**Abstract**

很多结果已经证明了CNN对于视觉物体分类（visual object categorization）这个任务能够学习到很好的representations。然而，CNN并没有显式的为图片里的object，parts以及它们的物理特性（physical properties）进行编码，这就限制了CNN在学习那些需要理解场景里的结构化信息的任务。为了克服这些限制，我们提出了Physical Scene Graphs（PSGs），其将一个场景（scene）表示为一个层次化的graph（hierarchical graph），graph里的nodes就自然对应各个不同scale下的object parts了，edges对应这些parts之间的物理联系（physical connections）。而每个node上还绑着一个向量，这个向量是表示这个node的代表的object part的特性，比如说surface shape、texture等。我们还提出了PSGNet，它是一个网络架构，其输入是一个场景，而其会将这个场景表示为一个PSG，然后再利用这个PSGNet来重构输入的场景，从而达到学会从场景图片里获取PSG的能力（有点像auto-encoder，而专业的表达这个过程的方式是通过一个PSG-structured bottleneck）。PSGNet在CNN上添加了以下的部分：（1）联合低层和高层图片信息的循环反馈链接（recurrent feedback connection）；（2）图池化（graph pooling）和vectorization操作，其将空间各向同性（spatially unform）的feature map转换为object-centric的图；（3）perceptual聚类约束，鼓励网络找到有意义的场景元素。我们证明PSGNet在scene segmentation任务上要比那些自监督scene representation learning的算法效果好，特别是对于现实世界里的很复杂的图片，而且PSGNet还能够泛化到没见过的object种类以及没见过的scene布置上。PSGNet还能够从物理运动（physical motion）里学到知识，从而为静态的场景图片提供更稳定的预测结果。我们提供了一系列的消融实验证明我们提出的PSGNet里的每个部分都是不可或缺的。我们的分析证明PSGNet确实能够学习到含有场景信息的feature embeddings，并且证明了PSGNet对compositional scene inference的作用。


**1. Introduction**

为了对它们所见到的环境做出解释，人工智能必须要对复杂的视觉场景（visual scene）构造出一个representation。最近，一类CV算法，CNN，已经证明了其从visual scene里获取用于分类的信息的能力。然而，人类的感知（同时也是CV的目标）不仅仅是图像分类。人类会将场景进行聚类，从而获取这个场景的以物体为中心的representations，这个representations里包含物体、物体的组成部分、物体的位置、物体的姿态、物体的3D几何信息、物体的物理材料特性、以及物体之间的联系。这样的以物体为中心（object-centric），几何信息丰富（geometrically-rich）的representations在认知领域十分重要，而且也天然的支持更高层次的视觉规划和推断任务。

最近的工作，比如说MONet（[Monet: Unsupervised scene decomposition and representation](https://arxiv.org/pdf/1901.11390.pdf)）和IODINE（[Multi-Object Representation Learning with Iterative Variational Inference](https://proceedings.mlr.press/v97/greff19a.html)），在以物体为中心（object-centric）的场景理解方面开始做出努力。以自监督学习的方式，这些模型可以将简单的人工生成的场景分离成一个个的物体。但是，这些方法对于所得到的物体并没有给出它们的物理结构信息，而且只是从静态的图片里去学习。从而，这些方法在复杂的真实世界的图片里就效果不好了。最近的3D视觉领域的方法，比如说3D-RelNet（[3D-RelNet: Joint Object and Relational Network for 3D Prediction](https://nileshkulkarni.github.io/relative3d/)），通过将关键的几何结构信息（比如说meshes）和由CNN获取的features结合起来来解决场景理解问题。这些方法的确获得了不错的结果，但是它们需要大量的、细节的关于scene结构的ground truth监督信号来完成学习。

在这篇文章里，我们提出了一个新的representation，physical scene graph（PSG）。PSG的概念借鉴了MONet/IODINE和3D-RelNet这两条research lines里的想法，想要去同时（1）表示复杂物体的shapes和textures；（2）显式的将场景分解为physical parts；（3）支持generative models对场景的自上而下的inference reasoning；（4）并且不需要监督信号，通过自监督的方式从真实世界的图片或者非常真实的生成图片里学习信息。PSGs将场景表示为层次化graph（hierarchical graphs），其中PSGs顶层的nodes直觉上表示的是更大的groupings（比如说单个的整个物体），而下层的nodes表示的是每个object的某些部分，而edges表示的是nodes之间的关系。PSGs同时还表示了每个node在图片里的位置（这种特性叫做spatial registered）。node的attributes表示这个node的物理意义上的性质（和这个node在哪个层级也是有关的），比如说物体的位置、物体的surface Normals、物体的形状以及物体的visual appearance。

我们最关键的贡献是提出了一个family的自监督网络框架，PSGNets，其可以从图片输入里学习PSGs。PSGNets是在基础的CNN上加了一些部分来实现的。为了在初始的feature extraction阶段更高效的结合高层次和低层次的视觉信息，我们在CNN的结构基础上加了局部的循环（local recurrent connections）以及长程反馈连接（long-range feedback connections），这样被改造后的CNN叫做convolutional recurrent neural network（ConvRNN）。我们还提出了一个可学习的图池化操作（graph pooling），将spatially-uniform的ConvRNN的feature map转换为object-centric的graph。我们还提出了一个将objects的特性综合起来的运算：graph vectorization operation。一系列交替的graph pooling和graph vectorization模块就构成了层次化的graph constructor。为了让这个graph constructor能够学习到具有物理意义的scene representations，我们在模型里编码了perceptual grouping的信息，包含静态和动态的grouping primitives。为了保证所学习到的node attributes包含场景图片里真实的可以被分开的（disentangled）objects的geometric和visual特征，我们还引入了一个decoder，从PSGs来自上而下的重构原输入的场景图片。

在多个数据集上，PSGNet都比那些使用无监督学习的工作在segmentation的任务上效果要更好，尤其是对于真实世界的图片来说。我们还发现，如果数据能够被提供的话，PSGNet能够充分从物体的运动里学到信息，并且将此应用到静态场景图片的学习里，从而和其它的motion-aware自监督方法比大幅度的提高了scene segmentation的效果。PSGNet所学习到的representations对于那些从没见过的objects和scenes来说泛化的很好，说明PSG结构里强力的限制使得网络确实学习到了普适的场景特性，这要比其它的方法所需要的训练数据量少得多。最后，我们展示了PSG的结构是如何表示关键的geometric和object-centric特性的，从而使得我们可以在一个场景里编辑objects的位置、结构以及改变objects的attributes。


**2. Related Work**

使用非监督方法来学习scene representations是CV领域一个长久的问题。经典的方法研究如何将像素点聚集为patches和object instance（都是2005年以前的方法，就不列出了）。最近，研究者们开发出利用deep generative models来做非监督segmentation的方法，其通过获取scene components之间的联系来实现的：[Monet: Unsupervised scene decomposition and representation](https://arxiv.org/pdf/1901.11390.pdf)，[Genesis: Generative scene inference and sampling with object-centric latent representations](https://openreview.net/pdf?id=BkxfaTVFwH)，[Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](https://proceedings.neurips.cc/paper/2016/file/52947e0ade57a09e4a1386d08f17b656-Paper.pdf)，[Multi-object representation learning with iterative variational inference](http://proceedings.mlr.press/v97/greff19a/greff19a.pdf)。除了静态的图片，研究者们同时还研究了利用运动信息来做object instance segmentation这个任务，他们的目标是同时得到object appearance和motion的信息，而且是无监督的方法：[Neural Expectation Maximization](https://proceedings.neurips.cc/paper/2017/file/d2cd33e9c0236a8c2d8bd3fa91ad3acf-Paper.pdf)，[Sequential Attend, Infer, Repeat: Generative Modelling of Moving Objects](https://proceedings.neurips.cc/paper/2018/file/7417744a2bac776fabe5a09b21c707a2-Paper.pdf)。这些方法都野心勃勃，但是实际上都只能在简单的，而且绝大多数情况下都是人工合成的场景图片上使用。用于从视频里进行motion clustering和segmentation的CV算法获取了很大的成功：[Towards segmenting anything that moves](https://openaccess.thecvf.com/content_ICCVW_2019/papers/HVU/Dave_Towards_Segmenting_Anything_That_Moves_ICCVW_2019_paper.pdf)，[Object discovery in videos as foreground motion clustering](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xie_Object_Discovery_in_Videos_as_Foreground_Motion_Clustering_CVPR_2019_paper.pdf)，但是它们都需要随着时间为每个object segmentation做大量精确的标注。还有一些工作尝试使用无监督的方法从视频里学习segmentation或者grouping，然后再辅助feature learning和instance segmentation任务：[Video segmentation by non-local consensus voting](https://www.wisdom.weizmann.ac.il/~alonf/papers/NonLocalVideoSegmentation.pdf)，[Learning features by watching objects move](https://openaccess.thecvf.com/content_cvpr_2017/papers/Pathak_Learning_Features_by_CVPR_2017_paper.pdf)，[Learning instance segmentation by interaction](https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w40/Pathak_Learning_Instance_Segmentation_CVPR_2018_paper.pdf)。然而，这些方法并没有学习object-centric representation，或者没有建立scene elements之间的关系。最近，graph neural network（GNN）证明了其再physical dynamics prediction方面的潜力，但是它仍需要graph-structured的输入或者监督信号：[Learning particle dynamics for manipulating rigid bodies](https://openreview.net/pdf?id=rJgbSn09Ym)，[Visual grounding of learned physical models](http://proceedings.mlr.press/v119/li20j/li20j.pdf)，[Flexible Neural Representation for Physics Prediction](https://proceedings.neurips.cc/paper/2018/file/fd9dd764a6f1d73f4340d570804eacc4-Paper.pdf)，[Learning to simulate complex physics with graph networks](http://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a.pdf)。而这也是我们希望能从图片输入学习到graph结构的输出的一个动机。

研究者们已经提出了将object-part层次化和object之间的关系考虑入内的scene representations。scene graphs（[Visual genome: Connecting language and vision using crowdsourced dense image annotations](https://arxiv.org/pdf/1602.07332.pdf)）是一个能够同时在一张图片里表示物体以及它们之间关系的representation。然而绝大多数的方法都是从有良好标注的数据里学习scene graph representations，只有少数研究是从无需标注的视频里去学习objects和它们的关系：[R-sqair: Relational sequential attend, infer, repeat](https://grlearning.github.io/papers/138.pdf)，[Relational Neural Expectation Maximization: Unsupervised Discovery of Objects and Their Interactions](https://www.sjoerdvansteenkiste.com/assets/papers/steenkiste2018relational.pdf)，[Unsupervised Discovery of Parts, Structure, and Dynamics](https://openreview.net/pdf?id=rJe10iC5K7)。这些非监督的方法有着和我们的方法最接近的设置，但是它们仅仅只是考虑2D像素层面上的segmentation。最近的一些工作开始关注从原始输入里获取3D的，以物体为中心的scene representations。最引人注目的工作包括Factor3D（[Factoring Shape, Pose, and Layout from the 2D Image of a 3D scene](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tulsiani_Factoring_Shape_Pose_CVPR_2018_paper.pdf)），MeshRCNN（[Mesh r-cnn](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gkioxari_Mesh_R-CNN_ICCV_2019_paper.pdf)）以及3D-SDN（[3D-aware scene manipulation via inverse graphics](https://proceedings.neurips.cc/paper/2018/file/64223ccf70bbb65a3a4aceac37e21016-Paper.pdf)）。3D-RelNet（[3d-relnet: Joint object and relational network for 3d prediction](https://openaccess.thecvf.com/content_ICCV_2019/papers/Kulkarni_3D-RelNet_Joint_Object_and_Relational_Network_for_3D_Prediction_ICCV_2019_paper.pdf)）将Factor3D更进一步拓展为可以预测物体之间的联系。然而，这些3D-aware scene representations都依赖于对物体geometry的标注信息，这就限制了它们再实际场景里对于没见过的物体的预测，因为这个时候就没有物体的3D几何信息了（因为压根就没见过这种物体）。一些无监督学习的模型，比如Pix2Shape（[Pix2shape: Towards unsupervised learning of 3d scenes from images using a view-based representation](https://arxiv.org/pdf/2003.14166.pdf)），GQN（[Neural Scene Representation and Rendering](https://www.gelecekburada.net/wp-content/uploads/google-yapay-zeka-perspektif-tamamlama-gelecek-burada.pdf)）以及neural rendering方法（[State of the art on neural rendering](https://onlinelibrary.wiley.com/doi/am-pdf/10.1111/cgf.14022)）也被提了出来。然而，它们还没有一个能够建立层次化的scene representation或者objects之间的联系。

PSGNet建立在上述这些方法之上，将它们的想法和思路综合起来，建立了一个层次化的（hierarchical），3D-aware的representation，而且不需要任何scene structure的监督信息。


**2. Method**

现在我们介绍PSG representation，PSGNet模型结构的组成部分，以及如何训练PSGNets。我们既解释了high-level的purpose，也解释了PSGNet每个组成部分的结构。正式的定义和实验implementation细节在补充材料里说明。

**2.1 Physical Scene Graphs**

PSGs是层次化的图结构（hierarchical graph），用来表示场景的层次化和物理结构。graph里的vertices，表示的是物体（objects）或者物体的一部分，用一系列hierarchical levels来表示：$$\lbrace V_l \vert l = 0,1,\cdots,L \rbrace$$。level $$l$$和level $$l+1$$之间的edges表示的是局部和整体（part-whole）之间的联系，叫做child-to-parent edges $$P_l$$。level $$l$$内部的vertices之间的edges表示objects之间或者object parts之间的关系，叫做within-level edges $$E_l$$。原则上，不同的within-level edge的集合应该能够编码不同的relationships，但是在这篇文章里，它们都表示的是physical connections。

除了vertices和edges，PSGs有两个额外的结构：（1）attributes vectors $$\lbrace A_l \rbrace$$，用于给每个vertex $$v$$添加标签，表示的是每个scene element的physical properties；（2）spatiotemporal registrations（SRs）$$\lbrace S_l \rbrace$$，其将每个level的PSG vertex显式的连接到一个base Tensor $$\mathcal F$$的某些像素点构成的子集，这个base Tensor $$\mathcal F$$形状是$$(T,H,W,C_0)$$，其中$$T$$表示的就是视频的帧数，$$H,W$$分别表示每一帧feature的大小，$$C_0$$是通道数，所以$$\mathcal F$$就是输入的视频通过CNN之后得到的feature maps。PSG就是由下而上在这个feature maps $$\mathcal F$$上构造出来的，因此PSG的第0个level就是$$\mathcal F$$，$$\mathcal F$$里的每个spatiotemporal点$$(t,h,w)$$就是一个vertex，而这个vertex的attribute就是这个点处长度为$$C_0$$的vector feature，记为$$A_0(v_{thw}) = \mathcal F \left[t,h,w,: \right]$$。一个registration $$S_l$$将这些spatiotemporal点划分为$$\lvert V_l \rvert$$个部分，也就是将base tensor $$\mathcal F$$按照第$$l$$层里vertex的个数进行了一个segmentation（从而反推回到CNN的输入，也就对输入也进行了一次分割）。更进一步的，SRs的层次化结构也要和vertex的层次化结构相匹配：如果base tensor $$\mathcal F$$里的一个spatiotemporal点通过$$S_l$$被归为vertex $$v$$了，那么通过child-to-parent edge $$P_l$$所找到的$$v$$在第$$l+1$$层的parent vertex $$P_l(v)$$，也应该通过$$S_{l+1}$$被归于vertex $$P_l(v)$$。因此，SRs表示的是场景图片里的pixels该如何层次化的被归类到每个object以及object的部分里去。一个node $$\pmb n_{l,i}$$表示的是PSG第$$l$$层的第$$i$$个vertex $$v_i$$、以及它的attribute vector $$A_l(v_i)$$，以及通过registraition $$S_l$$找到的base tensor $$\mathcal F$$上的spatiotemporal点的集合（实际上每层的registration $$S_l$$用值来表示base tensor上的点应该属于哪个vertex：比如说如果$$S_l$$里base tensor的某个点的值为$$i$$，那么它就归于第$$l$$层的第$$i$$个vertex）。


**2.2 The PSGNet Architecture**

PSGNets的输入是一张图片或者一系列图片（视频），输出是一个场景的PSG representation。PSGNet将PSG作为encoder所得出的representation，利用一个encoder-decoder的架构来进行训练：一个PSGNet encoder首先利用一个convolutional recurrent neural network（ConvRNN）（[Task-driven convolutional recurrent models of the visual system](https://proceedings.neurips.cc/paper/2018/file/6be93f7a96fed60c477d30ae1de032fd-Paper.pdf)）来从输入里获取features，然后再在features（也就是base tensor $$\mathcal F$$）上构造一个多层的PSG，最后再利用一个或者一系列decoder来还原输入（如果是视频的话就需要一系列decoders来还原每一帧）。我们接下来将会详细介绍encoder的结构，并在补充材料里提供全部的实现细节。我们之后再来介绍如何设计loss以及如何训练这个PSGNet。值得注意的是，PSG里的任何内容（vertices，edges，attributes，registrations）从没有受到任何ground truth的监督信号（比如说用来表明每个像素应该属于哪个objects的segmentation maps）。因此，PSGNets是自监督的图构造算法（self-supervised graph construction algorithm）。

**2.2.1 ConvRNN Feature Extraction**


**2.2.2 Graph Construction**

**2.2.2.1 Graph Vectorization**

**2.2.2.2 Learnable Graph Pooling**

**2.2.2.3 Graph Vectorization**













---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

