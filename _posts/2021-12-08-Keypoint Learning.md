---
layout: post
comments: false
title: "Keypoint Learning"
date: 2021-12-08 01:09:00
tags: paper-reading
---

> This post is a summary of keypoint learning related papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## 3D Keypoint Learning
### [Unsupervised Learning of Visual 3D Keypoints for Control](http://proceedings.mlr.press/v139/chen21b/chen21b.pdf)
[Code](https://github.com/buoyancy99/unsup-3d-keypoints) [Post](https://buoyancy99.github.io/unsup-3d-keypoints/)

*Boyuan Chen, Pieter Abbeel, Deepak Pathak*

*ICML 2021*

This paper proposed a method to directly learn 3D keypoint coordinates with only 2D images of the same scene from different view points as input in an unsupervised learning manner. The camera parameters are known. 

The whole process is as below:

Important hyperparameter: Keypoint number $$K$$.

**Step1** For each input 2D image $$I_n$$, use an encoder $$\Phi_n$$ to learn $$k$$ heatmaps $$C_n^k \in R^{S\times S}$$ and depth maps $$D_n^k \in R^{S\times S}$$. For each $$C_n^k$$, they used softmax to turn it into a probabilistic heatmap $$H_n^k$$ to represent the probability that keypoint $$k$$ will be at each position. Each depth map $$D_n^k$$ is a dense prediction of distance from camera plane for 3D keypoint k being at each position, i.e., each element in $$D_n^k$$ is the distance of keypoint $$k$$ to the camera plane.
After getting $$H_n^k$$ and $$D_n^k$$, they used the expectation across the whole heatmap to calculate the coordinate of keypoint $$k$$ in camera $$n$$ plane, $$\left[u_n^k, v_n^k, d_n^k\right]$$ as:

$$u_n^k = \frac{1}{S}\Sigma_{u,v}u H_n^k(u,v)$$, $$v_n^k = \frac{1}{S}\Sigma_{u,v}v H_n^k(u,v)$$, $$d_n^k = \Sigma_{u=1}^{S}\Sigma_{v=1}^{S}D_n^k(u,v) H_n^k(u,v)$$

Note that for each camera, we know the projection function to project the world coordinates into camera coordinates, denote it as $$\Omega_n$$, and the opposite projection as $$\Omega_n^{-1}$$. 

**Step2** Each camera will give a prediction of keypoint $$k$$ coordinates, the authors used some weighted techniques to get the final keypoint coodinates from these $$n$$ predictions, i.e., $$x^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i x_n^i$$, $$y^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i y_n^i$$, $$z^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i z_n^i$$, where $$\left[x^i, y^i, z^i\right]$$ is the final coodinate of keypoint $$i$$ in the world coordination, and $$\left[x_n^i, y_n^i, z_n^i\right]$$ is calculated from $$\Omega_n^{-1}(\left[u_n^i, v_n^i, d_n^i\right])$$.
The calculation of weights $$A_n^k$$ is as:

$$A_n^k = \frac{exp{(\frac{1}{S^2}\Sigma_{p=1}\Sigma_{q=1}C_n^k(p,q))}}{\Sigma_{i=1}^K exp{(\frac{1}{S^2}\Sigma_{p=1}\Sigma_{q=1}C_n^i(p,q))}}$$

**Step3** This step is a reconstruction step. The authors believed that if the decoder can reconstruct the input image from the keypoints projected from the learned 3D keypoints $$\left[x^i, y^i, z^i\right]$$, the model learns the geometry of the scene. The input of the decoder is the scaled stacked Gaussian distribution matrix with $$u_n^i, v_n^i$$ as the center and $$I_2/d_n^i$$ as the coviriance matrix. Note that each input image $$I_n$$ has a decoder $$\phi_n$$.

![Model Structure]({{ '/assets/images/Unsupervised_3D.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Overview of our Keypoint3D algorithm. (a) For each camera view, a fully convolutional neural network encodes the input image into K heat maps and depth maps. (b) We then treat these heat maps as probabilities to compute expectation of spatial $$u,v$$ coordinates in camera plane. These expected values and the saptial variances are used to resample final $$u,v$$ keypoint coordinates which adds noise that prevents the decoder from cheating to hide the input information in the relative locations $$u,v$$ keypoints. We also take expectation of depth coordinate, $$d$$, using the same probability distribution. These $$\left[u; v; d\right]$$ coordinates are then unprojected into the world coordinate. (c) We take attention-weighted average of keypoint estimations from different camera views to get a single prediction in the world coordinate. (d) For decoding, we project predicted keypoints in world coordinate to $$\left[u; v; d\right]$$ in each camera plane. (e) Each keypoint coordinate is mapped to a gaussian map, where a 2D gussian is created with mean at $$\left[u, v\right]$$ and std inversely proportional to $$d$$. For each camera, gaussian maps are stacked together and passed into decoder to reconstruct observed pixels from the camera. (f) Together with reconstruction, we also jointly train a task MLP policy on top of predicted world coordinates via reinforcement learning.*


### [Weakly-Supervised Discovery of Geometry-Aware Representation for 3D Human Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Weakly-Supervised_Discovery_of_Geometry-Aware_Representation_for_3D_Human_Pose_Estimation_CVPR_2019_paper.pdf)
*Xipeng Chen, Kwan-Yee Lin, Wentao Liu, Chen Qian, Liang Lin*

*CVPR 2019*

This work proposed a method to learn 3D coordinates of human body joints in order to do human pose estimation. This model is based on skeleton extracted from the raw RGB images, not an End-to-end framework.

Hyperparameter: Number of Keypoints $$K$$.

**Step1** Inputs are source image $$I_s$$ and target image $$I_t$$, and the rotation matrix are known due to the parameters of cameras. First, they use existing skeleton algorithm to extract skeleton maps of $$I_s$$ and $$I_t$$.

**Step2** Instead of a traditional encoder-decoder framework, they use a novel view synthesis method, i.e., source image $$I_s$$ are encoded and combined with rotation matrix $$R_{s \rightarrow t}$$, target image $$I_t$$ are reconstructed from the decoder. The 3D keypoint coordinates are the output of the encoder, as a geometry-aware representation, as explained in the paper. They also design the bidirectional encoder-decoder framework, which hinges on two encoder-decoder networks with same architecture to perform view synthesis in the two directions simultaneously, i.e., from $$I_s$$ to $$I_t$$ and from $$I_t$$ to $$I_s$$. These two reconstructions will involve two losses.

**Step3** They believe that the 3D keypoints of these two images should be the same. There are two encoders and the outputs should be the same, thus involve a new loss.

![Model Structure1]({{ '/assets/images/weak-supervised.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 The framework of learning a geometry representation for 3D human pose in a weakly-supervised manner. There are three main components. (a)Image-skeleton mapping module is used to obtain 2D skeleton maps from raw images. (b)View synthesis module is in a position to learn the geometry representation in latent space by generating skeleton map under viewpoint $$j$$ from skeleton map under viewpoint $$i$$. (c) Since there is no explicit constrain to facilitate the representation to be semantic, a representation consistency constrain mechanism is proposed to further refine the representation.*



### [Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning](https://keypointnet.github.io/keypointnet_neurips.pdf)
*Supasorn Suwajanakorn, Noah Snavely, Jonathan Tompson, Mohammad Norouzi*

*NeurIPS 2018 Oral*

This work proposed an end-to-end framework that learns 3D keypoints from pairs of images of the same catogory, with two images of each pair 2D images of a same object from different camera view.

Hyperparameter: Number of Keypoint $$K$$.

**Step1** The architecture of this model has $$K$$ heads, with each output a 3 tuple, representing the image coordinates and depth of keypoint $$k$$. Thus the total output is a  $$K \times 3$$ matrix. The calculation of keypoint coordinates are the same to [Unsupervised Learning of Visual 3D Keypoints for Control], i.e., outputing a probabilistic map, and use expectations to calculate coordinates. In contrast to approaches that learn a supervised mapping from images to a list of annotated keypoint positions, they do not define the keypoint positions a priori. Instead, they jointly optimize keypoints with respect to a downstream task, relative pose estimation.

**Step2** They assume a perspective camera model with a known global focal length $$f$$. And they also assumed that the transformation between two images $$T$$ is known. Thus they can have the projection funciton $$\pi T \pi^{'}$$ that project $$\left[u,v,z\right]$$ of image $$I$$ to $$\left[u^{'}, v^{'}, z^{'}\right]$$ of image $$I^{'}$$ as $$\left[u,v,z\right] = \pi T \pi^{'} \left[u^{'}, v^{'}, z^{'}\right]$$. And the deprojection that project $$\left[u^{'}, v^{'}, z^{'}\right]$$ back to the original point is also defined. Thus for a given keypoint coordiate $$\left[u,v,z\right]$$ of image $$I$$, they do the projection and deprojection, the result $$\left[u^{''},v^{''},z^{''}\right]$$ should be the same to $$\left[u,v,z\right]$$. Since they do not have the depth information, the consistency loss is defined as: 

$$L_{con} = \frac{1}{K}\Sigma_{i=1}^K||(u,v,u^{'},v^{'})-(u^{''},v^{''}, u^{'''}, v^{'''})||^2_F$$

**Step3** They argue that the above loss is enough for the consistency of keypoints, but will get to the suboptimal point when training. The keypoints may converge to a same point. They use a downstream task pose estimation to solve this problem. They define a differentiable objective that measures the misfit between the estimated relative rotation $$R^{'}$$ (computed via Procrustes’ alignment of the two sets of keypoints) and the ground truth $$R$$. So the pose estimation loss is: 

$$L_{pose} = 2arcsin(\frac{1}{2\sqrt(2)}||R^{'}-R||^2_F)$$ 

They do not consider transition and only rotation because they think their model has the translation equivariance property.

### [Hand Keypoint Detection in Single Images using Multiview Bootstrapping](https://openaccess.thecvf.com/content_cvpr_2017/papers/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.pdf)
*Tomas Simon, Hanbyul Joo, Iain Matthews, Yaser Sheikh*

*CVPR 2017*

This paper proposed a framework to learn 3D keypoints of hand. 

The input is a keypoint detector $$d_0$$ trained on a small labelled dataset $$T_0$$, a sequence of images $$\left\{I_v^f, v=1,2,...,V, f=1,2,..,F\right\}$$, with $$v$$ denote the camera view and $$f$$ denote the time frame.

Hyperparameter: Number of Keypoints $$K$$.

**Step1** First use $$d_0$$ to calculate the image coordinates (no depth) and confidence of each keypoint $$k$$ of $$I_v^f$$, denoted as $$x_v^{f,k}$$ and $$C_v^{f,k}$$. Then use the random sample consensus to pick inliers out of each set $$\left\{(x_v^{k}, C_v^{k})\right\}$$ for each time frame $$f$$. Then the 3D wolrd coodinates are computed as:

$$X_v^{k} = argmin_X \Sigma_{v \in I_v^{k}} ||P_v(X)-x_v^k||^2_F$$

where $$I_v^k$$ is the inlier set, with $$X_f^k \in R^3$$ the 3D triangulated keypoint $$k$$ in frame $$f$$, and $$P_v(X) \in R^2$$ denotes projection of 3D point $$X$$ into camera view $$v$$. They use calibrated cameras, thus $$P_v$$ are known.

**Step2** Then they use a window through the time frame, and pick the frame with highest score. The score is defined as the sum of $$C_v^k$$, thus the frame that has the biggest confidence of all keypoints detection from all camera views.

**Step3** After picking this frame, they add the labelled images into the orignal training dataset and train a new keypoint detector, $$d_1$$. And so on.


### [Learning deep network for detecting 3D object keypoints and 6D poses](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhao_Learning_Deep_Network_for_Detecting_3D_Object_Keypoints_and_6D_CVPR_2020_paper.pdf)
*Wanqing Zhao, Shaobo Zhang, Ziyu Guan, Wei Zhao*

*CVPR 2020*


### [OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields](https://arxiv.org/pdf/1812.08008.pdf)
*Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh*

*CVPR2017 and TPAMI 2019*

我们所说的是TPAMI的那个版本，比CVPR的那一版多了一些内容。

#### 1. Title
realtime表明这个算法本身运行速度快，multi-person 2D pose estimation清晰的说明了这个算法是用来干什么的，而part affinity field则是说明所用的主要方法。而openpose是给这个算法起个好听的名字，可以发现并不是常见的标题名字的简写组成的名字，openpose后来成为了一个成熟的软件。这种起标题的方式简明，清晰，直接说明白了用什么办法干了什么事情，是一种很好的起标题的方式。


#### 2. Abstract
两句话介绍了这篇论文所要解决的问题的背景，也就是multiple 2D person pose estimation的背景，并说明了这篇论文利用的是PAF来做到实时的效果。然后介绍了一下之前利用PAF的论文效果如何，为什么这篇论文利用PAF效果好。然后介绍了实验部分，最后说明这篇文章还发布了Openpose这样一个开源软件。

#### 3. Introduction

3.1. 这篇文章的目的是要做multi-person pose estimation，而对于单人的pose estimation有很多论文已经做得很好了，但对于多人来说，有以下几个困难的地方：1）首先，我们并不知道图里到底有几个人，而且每个人所在的位置、大小都不清楚；2）其次，人与人之间可能存在干涉，比如说遮挡、关节的旋转等等，很难分清楚到底哪部分属于哪一个人；3）以往的论文里的算法的复杂度都会随着图里人数量的增加而增加，从而很难实现实时。

3.2. 对于multi-person 2D pose estimation，top-down的方式很常见，也就是先检测图片里有几个人，然后对每个人实行pose estimation，因为单人的pose estimation已经做得很好了，这个算法并不复杂。但它存在着两个很大的问题：首先如果一开始检测人的时候就检测错了或者遗漏了，那之后是没有补救办法的；其次，这样的方法需要对检测出来的每个人都做单人pose estimation，这会使得算法的复杂度和人的数量成正比。所以说，bottom-up的方法也被提了出来，这种方法有能够解决上述两个问题的潜力。但之前的bottom-up方法仍然效率不高，因为它们在最后还是需要利用全局信息来辅助判断，从而要花不少时间。

3.3. 我们在这篇文章里利用Part Affinity Field实现了实时的multi-person 2D pose estimation，而且在几个数据集上效果都是很具有竞争力的（这样表达说明它们的效果不一定比其他论文好，它们的卖点主要是快）。Part Affinity Field是一个2D向量的集合，表示的是四肢的位置和方向信息。我们之后会说明，利用bottom-up的方式，将detection和association结合起来（模型有两个主要部分，一个是PAF refinement，另一个是body part prediction refinement，而PAF就是实现association的方式）逐步推进，可以在利用很小的计算资源的情况下达到很好的效果。

3.4. TPAMI的这一版比CVPR的那一版多的内容有：1）说明了其实PAF refinement才是效果好的主要原因，不可或缺，而body part prediction refinement没那么重要。文章里也做了实验，加深了网络深度，而去掉了body part prediction refinement的部分，效果变好了，速度也变快了；2）我们又提出了一个有标注的foot数据集；3）为了显示我们这个模型很强的generality，我们在vehicle keypoint detection的任务上也做了实验；4）我们这篇论文是我们的开源软件OpenPose的说明文档，OpenPose是第一个实时的body, foot, hand和facial的keypoint识别软件，我们还和Mask R-CNN, Alpha-Pose等著名keypoint识别算法在运行时间上作了比较。


#### 4. Related Work

4.1. **Single Person Pose Estimation**。单人的pose estimation，因为人是关节性的，所以说传统的方法都是通过将对于身体部位的布局的检测和它们之间的空间关系联合起来看，然后使用某种inference的方式来学习。而关节型的pose estimation的各部位之间的空间关系的表示方法，要么就是1）tree-structured graphical models，在相邻的部位之间做一些encoding；或者是2）non-tree models，在第一种tree-structured model的基础上加一些别的连线，从而实现对遮挡、对称、以及长途关系的表示（因为tree-structured model只对邻居部分有表示）。而对于身体各个部位的检测，CNN就起到了主要的作用，其在body part estimation里比传统的方法要好很多。

> 也就是说，其实single person pose estimation主要就分为两大部分，先想办法检测出身体的各个部分，然后再想办法利用一种好的表示方式来将这些部分连起来，从而实现pose estimation，也就是keypoint detection。

后来也有文章直接构建一个深度的graphical model将两部分放在一起解决；还有文章通过使用感受野比较大的CNN来学习身体部分之间的空间关系。也有文章是multi-stage的方式，每次利用全局信息来进一步优化每个身体部分所在位置的信息。但所有的上面的这些方法都是针对单人的，人的位置和scale在数据集里都是差不多的（不存在非常边缘或者大小差别过大的情况）。

4.2. **Multi-person Pose Estimation**。对于多人的pose estimation来说，绝大多数方法使用的是top-down的方式，即首先检测人，然后再单独的在每个检测到人的区域实行单人pose estimation算法。尽管这种方法使得成熟的单人pose estimation算法可以得到利用，但它不仅会因为早早的就预设了每个人的问题而存在问题，也会因为人和人之间也会存在空间依赖关系而出现问题（比如说遮挡等）。有一些文章已经开始考虑引入人与人之间的依赖关系了。有篇bottom-up的文章就没有使用person detection，而是将每个部分联系到每个人，但这个算法需要解决一个integer linear programming的问题，这个问题的计算复杂度很高，所以处理一张照片的时间要个把小时。后续这篇文章的跟进工作改进了身体部分的检测算子，并且改进了优化算法，从而使得检测一张照片的时间变成了几分钟，但最多只能处理150个身体部分。

在这篇文章的早期版本里，也就是CVPR那个版本里，我们介绍了part affinity field，它是一个representation，是由一系列的flow field组成的，而这些flow fields包含了每两个所检测到的身体部分之间的信息（可能是属于不同人的，也可能是一个人的）。和我们上面提到的两个引入人与人之间的依赖关系的论文不同，我们不需要训练就可以直接从PAF里获取pairwise的信息。而这些pairwise的信息对于multi-person pose estimation就已经是足够的了。

在我们的那篇CVPR文章提出之后，又有了一些新的工作。有篇文章进一步简化了各个身体部位的关系graph从而能更快的inference，其还将关节化的人的tracking形式化的表示为身体部分的spatio-temporal的grouping。还有篇文章提出了利用associative embedding来为每个keypoint打上标签，从而可以将标签相似（也就是说embedding之间的距离小）的keypoint归属于同一个人。还有一篇文章提出检测每个keypoint和它的相对位移，再计算每个keypoint属于哪个人。还有一篇文章提出了Pose Residual Network，其输入是keypoint和所检测到的人，而输出则是将keypoint归属到每个人。

我们这篇文章，对于之前的那篇CVPR文章做了一些扩充。我们证明PAF refinement是必要的且是最重要的，还做了实验去除了body part detection refinement的部分，速度增加了，效果变好了。我们同时还提出了第一个body和foot keypoint联合检测的detector，并且提出了一个foot keypoint数据集。我们证明将它们两联合检测不仅会减少inference的时间，还会保持准确率。最后，我们提出了OpenPose，是第一个实时进行body, foot, hand和facial keypoint detection的实时开源软件。


#### 5. Method









---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

