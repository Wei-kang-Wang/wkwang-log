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

![overview]({{ '/assets/images/OPENPOSE-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Overall pipeline. (a) Our method takes the entire image as the input for a CNN to jointly predict (b) confidence maps for body part detection and (c) PAFs for part association. (d) The parsing step performs a set of bipartite matchings to associate body part candidates. (e) We finally assemble them into full body poses for all people in the image.*

Fig 1显示了我们的算法的一个整体的流程。整个系统的输入是一张大小为$$w \times h$$的RGB图片，而输出为图中每个人的生理结构上的2D keypoints的位置。首先，一个feedforward网络输出一个集合$$S$$，用来表示各个身体部位位置的2D confidence maps，和一个part affinity fields（也就是2D的向量）的集合$$L$$用来表示各个身体部位之间的从属程度。集合$$S = \(S_1, S_2, ..., S_J\)$$有J个confidence maps，每个对应一个身体部位，其中$$S_j \in R^{w \times h}$$，$$j = \{1,2,...,J\}$$。而集合$$L = \(L_1, L_2, ..., L_C\)$$有$$C$$个向量，每个对应一个肢体，其中$$L_c \in R^{w \times h \times 2\}$$，而$$c \in \{1,...,C\}$$。为了简洁，我们将身体部位的pairs描述为肢体（因为这里的身体部位就是一个keypoint，而keypoint pair就是将两个keypoint连起来，就表示了一部分肢体），但是某些身体部位的pair并不是肢体，比如说face，我们就笼统的这么说了。我们可以从fig 2看到，$$L_c$$里的每个点都是一个2D的vector。最终，confidence maps和PAFs（也就是集合S和集合L）通过greedy inference联系了起来用于输出图中所有人的2D keypoints位置。

![algorithm]({{ '/assets/images/OPENPOSE-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. 上方：多人pose estimation。同一个人的身体部分被连了起来，也包括了脚的keypoints（大脚趾，小脚趾和脚后跟）。下左：关于连接右手肘和手腕的肢体的PAFs。颜色表明了方向。下右：在关于连接右手肘和手腕的肢体的PAFs的每个像素点处的2D向量包含了肢体的位置和方向信息。*

##### 5.1. Network Architecture

![architecture]({{ '/assets/images/OPENPOSE-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. multi-stage CNN的结构。前一部分的stage用来预测PAFs$$L^t$$, 而后一部分的stage用来预测confidence maps $$S^t$$。每个stage的输出和图片feature连接起来，作为下一个stage的输入。.*

我们的模型结构，如fig 3所示，迭代的预测包含身体部位之间关系的PAFs，如fig 3里蓝色的部分，之后再来检测confidence maps，如fig 3里的米色。这样一种迭代的方式使得预测更加的准确（注意中间的迭代stage也都是有supervision的）。

在我们CVPR的那版里，我们用了$$7 \times 7$$的卷积核，而我们这篇论文里，用3个$$3 \times 3$$的卷积核来替代从而减小计算量。如同DenseNet里那样，这3个卷积层的输出也连接起来了，从而能既有高层feature又有低层的。

##### 5.2 Simultaneous Detection and Association

输入的图像通过一个CNN（用的是VGG-19的前10层）来生成一系列的feature maps $$F$$，之后再输出到第一个stage当中。在这个stage里，网络输出一个集合的PAFs $$L^1 = \phi ^ 1 (F)$$，其中$$\phi ^ 1$$表示stage1里用来inference的CNN。在之后的stages里，前一个stage的输出和原始的图像feature maps $$F$$连接起来作为输入，用来输出更加精确的结果

$$L^t = \phi ^ t (F, L^{t-1}), 2 \leq t \leq T_p$$

其中$$\phi ^ t$$表示stage t里用来inference的CNN，$$T_p$是PAF stage的总数。在$$T_p$$个PAF stage之后，再来计算confidence maps，利用的是最新的PAF结果：

$$S^{T_p} = \rho^t (F, L^{T_p}), t = T_p$$

$$S^t = \rho^t (F, L^{T_p}, S^{t-1}), T^p < t \leq T_p + T_C$$

其中$$\rho^t$$指的是stage t用来inference的CNN，而T_C是confidence maps所需要循环的次数。

这个方法和我们的CVPR那个版本里的方法有很大的不同，在那个版本里，每个stage都要进行PAF和confidence maps的计算。因此我们这个版本的计算量减少了一半。我们通过实验发现，增加PAF循环的次数，可以使得效果变好，而增加confidence maps的循环的次数并不会。直觉上来看，通过PAF的结果，我们可以猜到每个身体部位（也就是每个keypoint）的位置，但通过confidence maps的结果，我们无法知道每个身体部位属于哪个人。

![PAF]({{ '/assets/images/OPENPOSE-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. 右前小臂肢体的PAF。尽管在一开始的stage里还有一些不清晰，但在之后的stage里可以看到PAF很清晰。.*

Fig 4显示了随着stage的增加，PAF计算结果逐渐被精确的过程。confidence maps的计算基于最后一个PAF stage输出的结果，而随着confidence maps stage的推进，结果区别并不大。为了让模型能够预测PAF和body part的confidence maps，我们在每个stage的结尾都使用一个loss function。我们使用的是输出的结果和实际上的body part的confidence maps和PAF之间的$$L_2$$ loss。我们这里给loss function加了权重，因为有些数据集并没有标注完整。对于PAF的stage $$t_i$$和confidence map的stage $$t_k$$的loss function是：

$$ f_L^{t_i} = \Sigma_{c=1}^C \Sigma_p W(p) ||L_c^{t_i}(p) - L_c^{\*}(p)||^2_2 $$

$$ f_S^{t_k} = \Sigma_{j=1}^J \Sigma_p W(p) ||S_j^{t_k}(p) - S_j^{\*}(p)||^2_2 $$

其中$$L_c^{\*}$$是PAF的ground truth，$$S_j^{\*}$$是身体部分confidence map的ground truth，$$W$$是一个非0即1的二分掩码，如果某个位置没有标注就是0，有标注就是1。这个$$W$$是用来避免因为没有标注而导致的错误训练。我们在每个stage都使用loss function，用来解决梯度消失的问题，因为每个stage结尾都有loss function，对梯度的值进行了补充。从而整体的的loss function就是：

$$ f = \Sigma_{t=1}^{T_p} f_L^t + \Sigma_{t=T_p + 1}^{T_p + T_C} f_S^t $$


##### 5.3. Confidence Maps for Part Detection

为了能够在训练过程中计算上述的$$f_S$$，我们从有标注的2D keypoints上生成confidence maps $$S^{\*}$$的ground truth。一个confidence map是一个2D的矩阵，用来表示一个keypoint出现在一个像素点的概率值。理想状态下，如果图里只有一个人，那么每个confidence map应该只有1个峰（该keypoint没有被遮挡住的情况下）;如果有多个人，那么每个confidence map对于没有被遮挡住的这一类keypoint都应该有峰（比如说某个confidence map专门表示人的nose的keypoint）。

我们先来对于每个人$$k$$生成confidence map，$$S_{j,k}^{\*}$$。$$x_{j,k} \in R^2$$表示人$$k$$的身体部分$$j$$在图中位置的ground truth。从而$$S_{j,k}^{\*}的位置$$p \in R^2$$处的值就是：

$$ S_{j.k}^{\*}(p) = exp(-||p-x_{j,k}||^2_2 / \sigma^2) $$

其中$$\sigma$$控制峰的大小。从而整个图片（可能包含多个人）的ground truth就是：

$$ S_j^{\*}(p) = max_k S_{j,k}^{\*}(p) $$

##### 5.4 Part Affinity Fields for Part Association

![cal]({{ '/assets/images/OPENPOSE-5.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. 身体部位association策略。（a) 几个人的两种身体部分（也就是keypoint）分别用红蓝点表示，而所有的点之间都连上了线。（b) 利用中间点进行连线。黑线是正确的，绿线是错误的，但他们都满足连接了一个中间点。(c) 利用PAF来连接，黄色的箭头就是PAF的结果。利用肢体来表示keypoint的位置和keypoint之间的方向信息，PAF减少了错误association的可能性。.*

给定一些已经检测到了的body parts（Fig 5里的红色和蓝色的点），那我们该如何将它们组合起来从而构建未知数量的人的肢体呢？我们需要对每一对body part keypoints都有一个confidence measure，也就是说，measure它们是否属于同一个人。一种可能的方式就是检测这一对keypoints的中间是否还有附加的midpoint。但是当人聚集在一起的时候，很容易出错。这种方式之所以不好是因为1）它仅仅有位置信息，并没有一对keypoint之间的方向信息；2）它仅仅用了midpoint，而不是这两个keypoints之间的所有部分当成一个肢体来使用。

Part Affinity Fieds (PAFs)解决了这些问题。它对于每一对keypoints构成的肢体提供了位置和方向信息。每一个PAF都是一个2D的vector field，在Fig 2里有显示。对于每个肢体的PAF的每个位置的值，其都是一个2D的向量，包含了位置和这个肢体一个keypoint指向另一个keypoint的方向。每个类别的肢体都有一个对应的PAF（由对应的body part keypoints对组成）。

![fig]({{ '/assets/images/OPENPOSE-6.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 6.*

考虑fig 6所示的一个简单的肢体。$$x_{j_1, k}$$和$$x_{j_2, k}$$是人$$k$$的身体部位$$j_1$$和$$j_2$$的ground truth，而这两个部位组成了肢体$$c$$。对于肢体$$c$$上的一点$$p$$，$$L_{c,k}^{\*}(p)$$是一个单位向量，从$$j_1$$指向$$j_2$$；对于其它的点，$$L_{c,k}^{\*}(p)$$的值都是0。

为了能在训练过程中计算$$f_L$$的值，我们需要定义PAF的ground truth，也就是对于人$$k$$，$$L_{c,k}^{\*}在$$p$$点的值为：

$$ L_{c,k}^{\*}(p) = v if p on limb c, k$$ and $$0$$ otherwise。

这里，$$v = (x_{j_2, k} - x_{j_1, l}) / ||x_{j_2, k} - x_{j_1, l}||$$是肢体的有方向的单位向量。一个肢体上的点不仅仅只有两个keypoints连线上的，而是有一个距离阈值，比如说：

$$ 0 \leq v (p - x_{j_1,k}) \leq l_{c,k}$$ 和 $$|v_{verticle} (p - x_{j_1,k})| \leq \sigma_l$$

其中肢体宽度$$\sigma_l$$自定义的，肢体长度由两个keypoints决定，也就是$$l_{c,k} = ||x_{j_2, k} - x_{j_1, k}||，$$v_{verticle}$$是垂直于$$v$$的。


而整个图片的PAF的ground truth是对于所有人取了均值：

$$ L_c^{\*}(p) = 1/n_c(p) \Sigma_k L_{c,k}^{\*}(p) $$

在测试过程中，我们通过计算连接两个keypoints的线段间的PAF的积分来衡量这两个keypoints是否构成了一个肢体。对于两个身体部分$$d_{j_1}$$和$$d_{j_2}$$，我们计算：

$$ E = \int_{u=0}^{u=1} L_c(p(u)) (d_{j_2} - d_{j_1})/||d_{j_2} - d_{j_1}|| du $$

其中$$p(u) = (1-u) d_{j_1} + u d_{j_2}$$。在实践中，我们通过等距离采样来近似这个积分值。


##### 5.5. Multi-Person Parsing using PAFs












---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

