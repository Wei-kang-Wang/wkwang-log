---
layout: post
comments: True
title: "关键点检测(point clouds)"
date: 2024-07-30 01:09:00

---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## Unsupervised Methods

### \[**CVPR 2021 Oral**\] [KeypointDeformer: Unsupervised 3D Keypoint Discovery for Shape Control](https://openaccess.thecvf.com/content/CVPR2021/papers/Jakab_KeypointDeformer_Unsupervised_3D_Keypoint_Discovery_for_Shape_Control_CVPR_2021_paper.pdf)

[PAGE](http://tomasjakab.github.io/KeypointDeformer) [CODE](https://github.com/tomasjakab/keypoint_deformer/)

现在在Internet上有非常多的3D shapes，给用户提供简单的interface，让他们可以在保留关键shape性质的情况下对3D object做semantically manipulating。文章里为interative editing提出自动检测intuitive和semantically有意义的control points，从而通过控制这些control points来对每个object类别的3D模型进行deformation，并且还保留了他们的shape的细节。

更准确的说，文章将3D keypoints作为shape editing的intuitive和simple interface。Keypoints是那些在一个object类别所有的3D shape间都semantically consistent的3D points。文章提出一个学习框架使用非监督学习的方式来找到这样的keypoints，并且设计一个deformation model来利用这些keypoints在保留局部形状特征的前提下改变物体的shape。这个模型叫KeypointDeformer。

Fig 1描述了KeypointDeformer在inference时候的过程。给一个新的3D shape，KeypointDeformer在它的surface上预测3D keypoints。如果一个用户将chair leg上的keypoint向上移动，整个chair leg都会超相同的方向形变（fig 1下面一行）。KeypointDeformer在这些可操纵的keypoints上提供了可选择的categorical deformation prior，比如说如果一个用户将一个airplane一侧wing上的keypoints向后移动，这一侧的wing会整体向后移动，而另一侧的wing也会随之移动同样的程度（fig 1上面一行）。当用户仅仅希望移动一侧wing的时候，我们的方法同样也允许这种操作。KeypointDeformer可以仅仅对于shape进行editing，也可以对两个shapes做shape alignment，还可以生成新的shapes来扩充datasets。

![overs]({{ '/assets/images/DEFORM-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 用非监督学习到的3D keypoints来进行shape deformation。我们用非监督学习的方式学习到的3D keypoints可以对object的shape进行intuitive控制。这个figure显示了交互式控制的独立的步骤。红色的箭头标注了keypoints被操作要移动的方向。注意到移动keypoints造成的shape形变是局部的并且object shape是按照intuitive的方式形变的：比如说，将airplane的wing上的keypoint向后拉，则整个wing也会向后倾斜，而保持原本shape的细节不变化。*

尽管3D keypoints对于shape editing来说很有效果，但是获取3D keypoints和deformation models的明确的监督信息不仅很贵，而且是ill-defined的。文章提出了一个unsupervised框架来将寻找3D keypoints和构建deformation model这两个任务同时完成。模型有两个在一起作用的模块：1）一个detecting keypoints的部分；2）一个将keypoints的位移信息传递到shape的其它部分从而进行deformation的deformation model。模型利用将一个source shape align到一个target shape上这样一个任务来训练网络，而且这两个shapes可以是同一个object category里差别很大的两个instances。模型还利用了一个keypoint regularizer，来促进学习到semantically consistent的keypoints。这些keypoints分布的很好，靠近object的surface并且隐式的保留着shape symmetries。KeypointDeformer训练之后所得到的就是一个deformable model，可以基于自动监测到的3D control keypoints来deform一个shape。因为keypoints是低维的，我们还可以在这些keypoints上学习一个category prior，这样就可以进行semantic shape editing了。

文章有以下几个关键的优势：

* 其给了用户一个intuitive并且简单的方法来交互式的控制object shapes
* keypoint prediction和deformable的模型都是unsupervised的
* 由文章的方法所找到的3D keypoints对于shape control来说比其他的keypoints都要好，包括人为标注的
* 文章的unsupervised 3D keypoints对于同一类别的object的不同的instances来说是semantically consistent的，从而给了我们两个point cloud的sparse correspondences。


**Related Work**

*Shape deformation*

文章的方法和geometric modeling里的detail-preserving deformations十分相关，包括[Differential Representations for Mesh Processing](http://mesh.brown.edu/dgp/pdfs/sorkine-cgf2006.pdf)，[As Rigid As Possible Surface Modeling](https://diglib.eg.org/bitstream/handle/10.2312/SGP.SGP07.109-116/109-116.pdf?sequence=1&isAllowed=n)和[Mean Value Coordinates for Closed Triangular Meshes](https://www.cse.wustl.edu/~taoju/research/meanvalue.pdf)。这些方法通过各种类型的限制（比如说points在一个optimization框架里）来允许进行shape editing，但它们一个最主要的问题就是它们仅仅依赖于geometric properties而并没有考虑到semantic attributes或者category-specific shape priors。这样的priors可以通过利用stiffness性质给object surface涂色获得，或者从一系列已经知道correspondence的meshes上学习得到。然而，这种监督信息十分昂贵，而且对于新的shapes来说就不管用了（training set没有见过的shapes，或者有新的priors的shapes）。[Semantic Shape Editing Using Deformation Handles](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.723.3455&rep=rep1&type=pdf)用一个提供了多个控制shape的sliders的data-driven的框架来解决了这个问题。然而这个方法需要一系列从专家标注的信息中提取的predefined attributes。

另一个相关的问题是[deformation transfer](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.126.6553&rep=rep1&type=pdf)，也就是利用两个shapes之间已知的correspondences将source mesh上的deformation转移到target mesh上。近期有些工作利用deep learning来隐式的学习shape correspondences来align两个shapes，比如[1](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yifan_Neural_Cages_for_Detail-Preserving_3D_Deformations_CVPR_2020_paper.pdf)，[2](https://arxiv.org/pdf/1804.08497.pdf)和[3](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_3DN_3D_Deformation_Network_CVPR_2019_paper.pdf)。

*User-guided shape editing*

文章的方法和最近的利用deep learning来学习可以提供对shape做interactive editing的generative模型。[Tulsiani的文章](https://openaccess.thecvf.com/content_cvpr_2017/papers/Tulsiani_Learning_Shape_Abstractions_CVPR_2017_paper.pdf)用primitives来抽象代表shapes，然后通过surface的primitives的deformation来edit shape。但是，shape editing并不是它们主要的目标，而且也不清楚直接进行primitive transformation能多大程度的保留local shape details。近似的工作进一步改进了这个方法，他们通过学习一个[point-based](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hao_DualSDF_Semantic_Shape_Manipulation_Using_a_Two-Level_Representation_CVPR_2020_paper.pdf)、[shape handles](https://openaccess.thecvf.com/content_CVPR_2020/papers/Gadelha_Learning_Generative_Models_of_Shape_Handles_CVPR_2020_paper.pdf)或者[disconnected shape manifolds](https://openaccess.thecvf.com/content_ICCV_2019/papers/Mehr_DiscoNet_Shapes_Learning_on_Disconnected_Manifolds_for_3D_Editing_ICCV_2019_paper.pdf)的primitives的generative model来改进原先的基于primitives的model的缺点。这些方法通过找到最佳匹配用户editing的latent primitive representations来做到interactive editing。但是他们的方法所用到的用户interface比较复杂，需要素描或者直接操控primitives。而且最关键的，因为这些editing是基于generative models的，这些方法可能会改变original shape的local details。而相对而言，我们直接对原shape进行deform，会有更好的shape detail的保留。我们将提出的方法和[DualSDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hao_DualSDF_Semantic_Shape_Manipulation_Using_a_Two-Level_Representation_CVPR_2020_paper.pdf)的结果进行对比来阐述上述的优势。


*Unsupervised keypoints*

在2D keypoint discovery领域，unsupervised方法有很多论文都已经有了不错的结果，[Unsupervised learning of object landmarks through conditional image generation](https://proceedings.neurips.cc/paper/2018/file/1f36c15d6a3d18d52e8d493bc8187cb9-Paper.pdf)，[Self-supervised learning of interpretable keypoints from unlabelled videos](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jakab_Self-Supervised_Learning_of_Interpretable_Keypoints_From_Unlabelled_Videos_CVPR_2020_paper.pdf)，[Self-supervised learning of a facial attribute embedding from video](https://arxiv.org/pdf/1808.06882.pdf)，[Unsupervised learning of landmarks by descriptor vector exchange](https://openaccess.thecvf.com/content_ICCV_2019/papers/Thewlis_Unsupervised_Learning_of_Landmarks_by_Descriptor_Vector_Exchange_ICCV_2019_paper.pdf)[Unsupervised learning of object landmarks by factorized spatial embeddings](https://openaccess.thecvf.com/content_ICCV_2017/papers/Thewlis_Unsupervised_Learning_of_ICCV_2017_paper.pdf)[Unsupervised discovery of object landmarks as structural representations](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Unsupervised_Discovery_of_CVPR_2018_paper.pdf)，但在3D keypoint discovery领域，unsupervised的方法却还没有被研究完全。[Discovery of latent 3d keypoints via end-to-end geometric reasoning](https://proceedings.neurips.cc/paper/2018/file/24146db4eb48c718b84cae0a0799dcfc-Paper.pdf)利用3D pose information作为supervision来从两张关于同一个object的不同角度的图片中检测3D keypoints。我们这篇文章聚焦于在从3D shapes上学习3D keypoints。[Unsupervised learning of intrinsic structural representation points](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chen_Unsupervised_Learning_of_Intrinsic_Structural_Representation_Points_CVPR_2020_paper.pdf)输出一个结构化的3D representation来获取sparse或者dense的shape correspondences。和我们这篇文章里进行3D keypoints discovery方法最像的文章是[Unsupervised learning of category-specific symmetric 3d keypoints from point sets](https://arxiv.org/pdf/2003.07619.pdf)，他们采用了显式的对称性限制条件。在这篇文章里，我们是为了shape control这个任务来使用非监督的方式寻找keypoints。尽管我们重点在于shape editing，但是我们的模型构造使得我们学习到了semantic consistent的3D keypoints。这样的以非监督方式学到的3D keypoints对于机器人来说没准是有用的，那些机器人相关的任务可以将3D keypoints作为latent representation用来控制机器人，而他们现在还需要手动定义3D keypoints来作为控制机器人的信号。


**Method**

目标是学习一个keypoint detector，$$\Phi: x \longrightarrow p$$，来将一个3D object shape $$x$$映射到一个semantically consistent的3D keypoints的集合$$p$$。我们同时也想学习一个输入为keypoints的conditional deformation model，$$\Psi: (x, p, p^{'}) \longrightarrow x^{'}$$，将shape $$x$$利用deformed control keypoints映射到shape $$x^{'}$$，其中$$p$$描述的是initial（source）keypoint locations，$$p^{'}$$描述的是target keypoint locations。为keypoints和deformation model获取显式的监督信息十分expensive而且ill-defined。因此，我们提出了一个unsupervised的learning框架来训练上述提到的两个functions。我们通过设计了一个pair-wise shape alignment的辅助任务来实现，这个辅助任务的核心想法就是将keypoints learning和deformation model联合起来学习，从而可以对两个任意的shapes做alignment。更仔细地说，模型首先利用一个Siamense network在source和target shapes上预测3D keypoint locations。之后我们利用检测到的keypoints的对应关系来deform source shape（检测到的keypoints是默认有序的，从而有着对应关系）。为了保持local shape detail，我们使用了一个基于keypoints的cage-based deformation方法。我们使用了一个新颖的但十分简单高效的keypoint regularization term，使得keypoints是well-distributed的，并且距离object surface很近。Fig 2显示了我们的模型的整体框架。

![framework]({{ '/assets/images/DEFORM-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Model。我们的模型使用预测到的unsupervised keypoints $$p$$和$$p^{'}$$来将source shape $$x$$ aligns到target shape $$x^{'}$$。unsupervised keypoints描述了object的pose并用作deformation的control points。整个模型使用在deformed source shape $$x^{\*}$$和target shape $$x^{'}$$之间的similarity loss和keypoint regularization loss来进行end-to-end的训练。在interactive shape manipulation的test time，用户可以选择只输入一张source shape $$x$$，keypoint predictor $$\Phi$$就会预测一些unsupervised 3D keypoints $$p$$出来。然后用户可以手动控制keypoints $$p$$使其变成target keypoints$$p^{'}$$，然后再用deformation model $$\Psi$$来生成deformed source shape $$x^{\ast}$$，如fig 1，fig 9或者project page上的补充材料里的视频所示。*

**1. Shape Deformation with Keypoints**

我们将每个object表示为一个point cloud $$x \in R^{3 \times N}$$，是从object mesh里均匀采样得来的。我们先从source和target里预测keypoints。keypoint predictor $$\Phi$$使用$$x$$作为输入，输出一个ordered set，$$p = (p_1,...,p_K) \in R^{3 \times K}$$，表示的是3D keypoints。这个keypoint predictor的encoder对于source和target是公用的，使用Siamese architecture来实现。而shape deformation function $$\Psi$$的输入是source shape $$x$$，source keypoints $$p$$和target keypoints $$p^{'}$$。在test阶段，输入一张图片，得到了其的source keypoints，用户editing之后得到了target keypoints，之后生成输入图片的deformation，整个interactive shape deformation过程如fig 2所示。

为了在deform object的过程中保持它的local shape details，我们使用最近刚出现的[differentiable cage-based deformation algorithm](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yifan_Neural_Cages_for_Detail-Preserving_3D_Deformations_CVPR_2020_paper.pdf)。cages是个很经典的shape modeling方法，其使用一个粗糙的封闭的mesh将shape包起来。deform cage mesh就会导致里面包裹的shape也发生deformation。cage-based deformation function $$\beta: (x,c,c^{\ast}) \longrightarrow x^{\ast}$$的输入是source control cage $$c$$，deformed control cage $$c^{\ast}$$，以及source shape $$x$$（也就是一开始从mesh里采样得到的point cloud）。我们通过一开始用一个球体包住source shape $$x$$，之后再将每个cage vertex $$c_V$$向object的中心推进直到它和object surface之间只有一个很小的距离这样一种方法来为每个shape都自动的获取包裹其的cage。fig 2显示了所得到的cage的样子。尽管cages对于shape-preserving deformation来说是个有用的方法，但通过deform cages来获得内部的shape的deformation并不是那么的直观，特别是对新手用户来说，因为cage vertex并不直接落在shape的表面上，并没有一个粗糙的structure，而且在不同的shape之间（同一个object或者同一类object的不同姿态的shape）并不semantically consistent。我们提出keypoints用作操纵cage deformation的方式更为合理。

为了用我们检测到的keypoints来控制object deformation，我们需要将这些keypoints和cage vertices联系起来。我们通过使用一个linear skinning function，首先计算source和target keypoints之间的relative distance，$$\delta p = p^{'}-p$$，然后将一个可学习的influence matrix，$$W \in R^{C \times K}$$乘上$$\delta p$$，在加到source cage vertices，$$c_V$$上，就获得了新的target cage vertices，$$c_{V}^{\ast}$$。其中，$$p,p^{'},\delta p$$都是$$K \times 3$$的矩阵，$$c_V, c_V^{\ast}$$是$$C \times 3$$的矩阵，而$$K$$和$$C$$分别表示keypoints和cage vertices的个数。所以deformed cage vertices，$$c_V^{\ast}$$计算方式为：

$$c_V^{\ast} = c_V + W \delta p$$

为了满足对于每个shape来说cage是唯一的这样的事实，我们将上述的influence matrix，$$W$$，设置为输入shape $$x$$的一个函数。详细的说，influence matrix是一个composition，$$W(x) = W_C + W_I(x)$$，其中$$W_c$$是对于每一类object的所有instances都共用的canonical matrix，而$$W_I(x)$$则是每个instance独自的offset，是利用influence predictor $$W_I = \Gamma(x)$$以source shape $$x$$为输入计算而来。我们同时也通过最小化其Frobenius norm来regularize这个instance specific offset，$$W_I$$，为了防止它过拟合influence matrix $$W$$。我们将这个regularizer命名为$$L_{inf}$$。最后，我们限制$$W$$使得每个keypoint最多只能影响$$M$$个最近的cage vertices来实现locality。


**2. Losses和Regularizers**

我们的KeypointDeformer是通过最小化source和target shape之间的similarity loss，再加上keypoint regularization loss和instance-specific influence matrix regularization term，利用SGD实现的end-to-end的训练。

**Similarity loss**

理想情况下，我们希望利用已知的meshes之间的correspondence来计算deformed source shape $$x$$和target shape $$x^{'}$$之间的similarity。但是这样的correspondence是不存在的，因为我们希望能在最普遍的object category CAD模型上训练。我们通过将source shape和target shape都表示为point cloud，然后再计算他们之间的Chamfer distance来近似这个similarity loss。这个loss记为$$L_{sim}$$。


**Farthest Point Keypoint regularizer**

我们提出了一个简单有效的keypoint regularizer $$L_{kpt}$$来使得预测的keypoints $$p$$是well-distributed的，也就是再object surface上，并且能保持这个shape category本身的symmetric structure。具体来说，我们设计了一个**Farthest Sampling Algorithm**来从输入的shape mesh里采样一个无序集合$$q = \{q_1,...,q_J\} \in R^{3 \times J}$$作为point cloud。采样的起始点是随机的，所以每次我们计算这个regularization loss的时候我们都使用的不同的point cloud $$q$$。给定这些随机的farthest points，regularizer最小化所预测的keypoints $$p$$和这些采样到的点$$q$$之间的Chamfer distance。也就是说，这个regularizer希望keypoint detector $$\Phi$$能够学习到那些和$$q$$分布类似的keypoints。Fig 3展示了$$q$$的特性。这些采样到的点对于提供了输入的object shape $$x$$的一个均匀的覆盖，其再不同的instances之间比较稳定，而且保持了最初input shape的symmetric结构。

![LOSS]({{ '/assets/images/DEFORM-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. Farthest Point Keypoint regularizer. 我们使用一个随机的初始点来做farthest point sampling，来regularize预测到的keypoints。(a) 展示了对于一个给定的点，其被farthest point sampling algorithm所选到的频率。颜色越深表明这个点被选到的概率越大。采样点的期望locations对原shape有一个很好的覆盖而且保留了原shape的symmetry特征。而且，它们的子集在不同的object instances之间保持了semantically stable。使用采样点的期望locations作为keypoint location的prior效果很好，因为keypoint predictor会学会对这些采样点里的噪声比较robust。在airplane的例子里我们可以看到，fuel tank的顶点（红圈标记）并没有被keypoint predictor用作keypoint，(b) 而wing的顶点（绿圈）则被选中为keypoint，因为其在数据集里更加consistent（大多数飞机都有wings，但很多并没有fuel tank）。* 

这个regularization的另一个intuition是我们可以将这些采样的farthest points $$q$$理解为keypoint locations的一个noisy prior。这个prior并不是完美的——在某些shape上可能会遗失某些重要的点，或者有一些不合理的点——但是neural network keypoint predictor会以一种对这些noise robust的方式学到keypoint的locations，而且会偏向于学习那些consistent的keypoints，如fig 3所示。

**Full objective**

总结来说，我们的training objective是：

$$L = L_{sim} + \alpha_{kpt}L_{kpt} + \alpha_{inf}L_{inf}$$

其中$$\alpha_{kpt}$$和$$\alpha_{inf}$$是scalar loss系数。我们的方法很简单而且并不需要对于shape deformation增加额外的shape specific regularization，比如说point-to-surface距离，normal consistency，symmetry losses。这是因为keypoints提供了一个shapes之间的low-dimensional的correspondence，而且cage deformations是这些keypoints的一个linear function，从而阻止了那些会导致local deformation的极端的deformations。


**3. Categorical Shape Prior**

因为我们利用一系列semantically consistent的keypoints来代表一个object shape，我们可以通过计算training set里的shape对应的keypoints的PCA来获取categorical shape prior。这个prior可以用来指导keypoint manipulation，也就是上面提到的$$W_C$$。比如说，如果用户想要改变一个airplane一个wing上的一个keypoint，根据寻找到能够最佳重构这个被改变的keypoint的新位置的PCA basis coefficients，其余的keypoints就会被这些basis coefficients”同步协调“。从而这些keypoints就会根据这个prior（也就是这个PCA）落到新的位置。这个prior还可以通过采样一系列新的keypoints来生成新的shapes：调整某些keypoints，然后PCA经过计算basis coefficients来对所有的keypoints位置进行调整，从而得到了新的keypoints位置，然后利用上述的deformation model来生成新的shape，就可以将这个新的shape加入已有的3D shape datasets里。



### \[**CVPR 2024**\] [Back to 3D: Few-Shot 3D Keypoint Detection with Back-Projected 2D Features](https://github.com/wimmerth/back-to-3d-few-shot-keypoints)

### \[**Arxiv 2024**\] [ZeroKey: Point-Level Reasoning and Zero-Shot 3D Keypoint Detection from Large Language Models](https://sites.google.com/view/zerokey)

### \[**ECCV 2024**\] [SelfGeo: Self-supervised and Geodesic-consistent Estimation of Keypoints on Deformable Shapes](https://github.com/IIT-PAVIS/SelfGeo)
