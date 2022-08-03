---
layout: post
comments: false
title: "[论文]Synthesis, Reconstruction & Generation"
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

### 1. [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392.pdf)

*Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng*

*ECCV*

![1]({{ '/assets/images/NERF-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1 我们提出一个方法从一系列图片来优化一个continuous的5D neural radiance field representation（在任何位置上的volume density和view-dependent color）去表示一个scene。我们使用volume rendering里的方法来沿着ray积累这个scene representation的samples，从而可以从任意相机角度来渲染这个scene。这里，我们可视化了100个Drums这个场景的views，其是被环绕的一个半球的区域内随机采样出来的。我们还展示了从我们的已经被优化了的NeRF representation里得到的两个这个场景的新角度的照片。*

我们提出一个方法，使用一系列input views作为输入，通过优化一个潜在的continuous volumetric scene函数来对复杂场景生成新的角度的views。我们的算法使用一个MLP（而不是CNN）来表示一个scene，其输入是一个连续的5D coordinate（空间位置$$(x,y,z)$$以及view的方向$$(\theta, \phi)$$），输出是volume density和那个位置$$(x,y,z)$$的view dependent emitted radiance。我们通过沿着相机rays来查找5D coordinate，并使用经典的volume rendering方法来将网络输出的color和density投射到一张照片上。因为volume rendering本身就是differentiable的，所以我们唯一需要的输入就是一系列已知相机姿态的不同角度的照片，也就是一个场景不同角度的views。我们描述如何高效的优化neural radiance fields来渲染有很复杂geometry和appearance的场景的新的角度的photorealistic的图像，并且展示了效果要比之前的neural rendering和view synthesis的工作的效果要好很多。view synthesis的结果最好用视频来看，在补充材料里有。


这篇文章用一种全新的方法来解决view synthesis这个被研究了很久的问题，方法是，通过直接优化一个连续的5D scene representation的参数来最小化渲染images的error。

我们将一个静态的场景表示为一个连续的5D函数，输出空间中每个点$$(x,y,z)$$在每个角度$$(\theta, \phi)$$情况下的emitted radiance，以及一个density。这个density作为一个可微分的opacity存在，其控制着一条射线穿过$$(x,y,z)$$这个点时，需要有多少的radiance被考虑进去。我们的方法优化一个MLP（没有任何卷积）来表示上述这个5D的function，其输入为一个5D coordinate $$(x,y,z,\theta,\phi)$$，输出为一个volume density和view-dependent RGB color，所以是一个regression问题。

为了从一个特定的视角来渲染这个neural radiance field（NeRF），我们需要：1)从相机发射rays穿过场景来采样一系列3D的点；2)使用这些点和对应的2D的角度输入作为5D输入给网络来输出每个点的color和density；3)使用经典的volume rendering方法来收集这些colors和densities，从而生成一张2D的照片。因为上述过程是可以微分的，所以我们可以使用梯度下降的方法通过最小化输入图片和网络生成图片两者之间的差异来优化这个模型。对于多个角度拍摄的这个场景的图片都进行上述最小化的操作，我们所学习到的这个网络就能够对于场景中真实存在物体的位置给出很高的volume density和准确的colors。fig 2给出了这整个过程的pipeline。

![2]({{ '/assets/images/NERF-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2 neural radiance field scene representation和differentiable rendering procedure的一个overview。我们通过沿着camera ray采样5D coordinates（location和viewing direction）来生成images。(a)将这些locations喂给一个MLP来输出color和volume density；(b) 使用volume rendering方法来将这些values composite到一个image里；(c) 这个rendering function是可微分的，所以我们可以通过最小化生成的images和ground truth images之间的差异来优化这个scene representation。*

我们发现对于一个复杂的scene，用基础的方法来优化上述这个neural radiance field representation是不能让其收敛到很好的情况的，而且对于每个camera ray都需要很多的采样点。我们通过给原始的5D输入加上positional encoding，从而使得这个MLP能够表示更高频率的函数，来解决这个问题。我们还提出了一个hierarchical的采样方法来减少所需要的采样点的个数。

我们的方法继承了volumetric representation的优势：可以表示复杂的真实世界的geometry和appearance，而且很适合利用projected images来进行基于梯度的优化。而且，我们的方法克服了表示高分辨率的场景的时候所需要存储大量离散的voxel grids的问题。总而言之，这篇文章的贡献包括：

* 提出了一个将具有复杂geometry和appearance的连续的scene表示为5D neural radiance fields的方法，将其用MLP来表示
* 基于经典的volume rendering方法的一个可微分的rendering方法，可以让我们通过RGB图片来优化上面提到的表示neural radiance field的MLP。而且还提出了一种hierarchical的采样方法。
* 提出一个positional encoding将每个5D输入map到一个更高维度的空间，从而使得优化能够表示更高频率的scene的neural radiance fields成为现实

我们定性定量的表明我们的方法要比现在state-of-the-art的view synthesis方法要好，包括那些将3D representations fit到scene的工作，也包括那些训练CNN来预测采样的volumetric representations的工作。以我们的了解，这篇论文提出了第一个能够从真实世界的scene或者objects的众多角度的RGB图片中学习到连续的neural scene representation，并且渲染出新的角度的高分辨率的photorealistic的图片的模型。


**2. Related Work**

近期CV领域一个有前途的方向是将objects和scenes encode到一个MLP的参数里，也就是直接从一个3D的location映射到一个shape的隐式的representation，比如说在某个位置的signed distance：[A Volumetric Method for Building Complex Models from Range Images](https://dl.acm.org/doi/pdf/10.1145/237170.237269)。但是到目前为止，这些方法（利用MLP来表示representations）都无法做到基于discrete representations（比如triangle meshes，voxel grids等）的方法那样，生成十分逼真的图片。在这一节里，我们回顾一下两条line of research，并将它们与我们的方法进行对比，突出neural scene representations表达复杂场景并能渲染出逼真照片的能力。

一些相似的方法也是使用MLP来从低维的coordinates映射到colors，但这些方法用来表示的是其它的graphics functions，比如说images（[Compositional pattern producing networks: A novel abstraction of development]()），textured materials（[Learning a neural 3d texture space from 2d exexmplars]()，[Unified neural encoding of BTFs]()，[Neural BTF compression and interpolation]()）以及indirect illumination values（[Global illumination with radiance regression functions]()）。

*Neural 3D shape representations*

最近的工作通过优化一个将xyz坐标映射到signed distance functions（[Local implicit grid representations for 3d scenes]，[DeepSDF: Learning continuous signed distance functions for shape representation]）或者occupancy fields（[Local deep implicit functions for 3d shapes]，[Occupancy networks: Learning a 3D reconstruction in function space]）的神经网络，将连续的3D shapes当作level sets来研究其隐式表示。然而，这些方法都需要ground truth的3D geometry，一般都是从某些synthesis的3D shape dataset里获得的，比如说ShapeNet（[Shapenet: An introduction-rich 3d model repository]）。后续的工作放宽了需要3D shapes的这个要求，他们通过构造可微分的rendering functions来实现，而这些rendering functions允许只利用2D images来优化neural implicit shape representations。[Differentiable volumetric rendering: Learning implicit 3D representations without 3D supervision]()将surfaces表示为3D occupancy fields，并且使用了某种数值方法来找到每个ray和surface的相交点，然后利用隐式微分来计算该点的导数。每个ray和surface的焦点都是网络的输入，输出是该点的diffuse color。[Continuous 3D-structure-aware neural scene representations]()使用一个并不是那么直接的3D representation，输出是每个3D coordinate的一个feature向量和RGB color，并且利用一个由RNN组成的可微分的rendring function来沿着每条ray计算，从而判断surface位于什么为止。

尽管这些方法潜在的可以表示复杂的、高分辨率的geometry，但是它们现在只被用于有着低geometry复杂度的很简答的shape上，从而导致过于平滑的渲染效果。我们这篇文章表明使用5D输入（3D的位置，2D的相机角度），能够让网络表示高分辨率的geometry，并且能对复杂场景的新的view也生成photorealistic的照片。


*View synthesis and image-based rendering*

给定views的dense采样，很逼真的新的views可以直接被简单的light field sample interpolation方法构造出来，[The lumigraph]()，[Unstructured light fields]()和[Light field rendering]()。而对于基于稀疏的views的新的view synthesis，CV和graphics研究团体通过从images里预测传统的geometry和appearance representations做出了很大进步。一类很流行的方法使用scenes的mesh representations，而这些scenes有着diffuse或者view-dependent的appearance。可微分的rasterizers或者pathtracers可以直接优化这些mesh representations从而重构输入图片。然而基于image reprojection的对mesh的优化往往效果不好。而且，这个方法需要一个有着固定拓扑结构的template mesh，这对于真实世界的场景往往是获取不到的。

另一类比较流行的方法使用volumetric representations来解决从一系列RGB图片中学习到高质量的逼真的view synthesis问题。volumetric方法能够很逼真的表示复杂的shapes和materials，而且很适合用来做基于梯度下降的优化。早期的volumetric方法使用图像直接给voxel grids染色。对于更近期的一些方法（[DeepView: view synthesis with learned gradient descent]()，[Single-image tomography: 3d volumes from 2d cranial x-rays]()，[Learning a multi-view stereo machine]()，[Local light field fusion: Practical view synthesis with prescriptive sampling guidlines]()，[Soft 3D reconstruction for view synthesis]()，[Pushing the boundaries of view extrapolation with multiplane images]()，[Multi-view supervision for single-view reconstruction via differentiable ray consistency]()，[Stereo magnification: Learning view synthesis using multiplane images]()），他们使用多个场景的大的数据集来训练网络能够从输入的图像里预测volumetric representation，之后在测试阶段再利用alpha-compositing或者可学习的compositing，沿着rays来渲染出新角度的views。另一些工作对于每个scene联合优化一个CNN和sampled voxel grids，使得这个CNN可以对于低分辨率voxel grids中阐述的discretization artifacts进行补偿。

虽然上述这些volumetric方法对于novel view synthesis获得了很大的成功，但是他们对于更高分辨率的图片的生成受限于时间和空间复杂度——因为他们需要discrete sampling——渲染高分辨率的图片需要对3D空间进行更精细的采样。我们通过利用MLP来encode一个volume来解决这个问题，其不仅能生成更加逼真的图像，而且和这些sampled volumetric representations的方法相比，所需要的存储空间要小得多。


**3. Neural Radiance Field Scene Representation**

我们将一个continous scene表示为一个5D vector-valued function，其输入是一个3D location，$$\pmb{x} = (x,y,z)$$和一个2D的view direction，$$(\theta, \phi)$$，输出是一个emitted color，$$\pmb{c} = (r,g,b)$$以及一个volume density $$\rho$$。在实践中，我们将方向表示为一个3D Cartesian单位向量$$\pmb{d}$$。我们用一个MLP来近似这个连续的5D scene representation：$$F_{\Theta}:(\pmb{x}, \pmb{d}) \longrightarrow (\pmb{c}, \rho)$$，并且优化这个MLP的weights，$$\Theta$$。

我们通过将volume density $$\rho$$仅仅表示为输入location $$\pmb{x}$$的函数，而输出的color $$\pmb{c}$$表示为输入location $$\pmb{x}$$和direction $$\pmb{d}$$的函数，来鼓励representation是multiview consistent的。为了实现这点，MLP $$F_{\Theta}$$首先利用8层MLP来处理输入的3D坐标$$\pmb{x}$$（每层256个通道，使用ReLU作为激活函数），然后输出$$\rho$$和一个256维的feature向量。这个feature向量再和相机角度$$\pmb{d}$$链接，喂给另一个MLP（每层128通道，使用ReLU作为激活函数）输出view-dependent RGB color。

我们可以从fig 3看到我们的方法如何利用输入的相机角度来表示non-Lambertian效果。fig 4表示，如果没有相机角度这个输入，仅仅利用$$\pmb{x}$$作为输入，那么表示高光反射的位置就会出现问题。


**4. Volume Rendering with Radiance Fields**

我们的5D neural radiance field将一个scene表示为空间中任意一个点的volume density和directional emitted radiance。对于任意穿过这个scene的ray的渲染color，使用的是经典的volume rendering方法（[Ray tracing volume densities]()）。volume density $$\rho(\pmb{x})$$可以被理解为从相机出发的ray在点$$\pmb{x}$$处被拦下来的概率。相机ray $$\pmb{r}(t) = \pmb{o} + t \pmb{d}$$的color，$$C(\pmb{r})$$是：

$$C(\pmb{r}) = \int_{t_n}^{t_f} T(t)\rho(\pmb{r}(t))\pmb{c}(\pmb{r}(t), \pmb{d})dt$$

其中$$T(t) = exp(-\int_{t_n}^t \rho(\pmb{r}(s))ds)$$，而$$t_n$$和$$t_f$$是相机ray的起始和终止点。


>这个$$C(\pmb{r})$$表示的就是从某个相机角度出发的ray的平均color，积分内的值，可以理解为，$$T(t)$$表示从出发点到$$t$$位置为止积累的透明度，如果在碰到$$t$$以前就碰到了其它的透明度很低的点了（也就是真实的scene的点，$$\rho$$很大），那这个点之后的$$T$$都很小了，所以即使以后再遇到真实的scene的点，$$\rho$$很大，也不会将这个新遇到的点的color记进去的，这也符合事实，因为我们从一个角度来看scene的话，前面的点会挡住后面的点，那看到的只能是前面的点的颜色。

$$T(t)$$这个函数表示的是沿着ray从$$t_n$$到$$t$$积攒的透明度，也就是ray从$$t_n$$到$$t$$没有碰到任何particle的概率。从continuous neural radiance field渲染一个view，那么对于每个相机角度，假设有个虚拟的相机放在该处，则对于该相机的每个像素点都需要计算通过该像素点的ray的$$C(\pmb{r})$$的值。

我们使用quadrature来近似上述这个integral的值。deterministic quadrature，经常被用来渲染discretized voxel grids，会限制我们的representation的分辨率，因为这样color的值只会在一些固定的离散区域被考虑，这样MLP就只会拟合这些特定离散位置点的color，从而限制了更精确、分辨率更高的渲染。所以，我们使用了一种stratified sampling方法，将$$\left[t_n,t_f\right]$$分为$$N$$等份个bins，然后在每个bin里随机取一个点：

$$t_i \sim \mathcal{U} \left[t_n + \frac{i=1}{N}(t_f - t_n), t_n + \frac{i}{N}(t_f - t_n)\right]$$

尽管我们使用的是一个采样的离散集合来估计这个integral，stratified sampling使得我们可以表示一个continuous scene representation，因为加入了随机性使得MLP是在连续的位置上被优化的。我们使用这些采样点来估计$$C(\pmb{r})$$的值：

$$\hat C(\pmb{r}) = \Sigma_{i=1}^N T_i(1 - exp(-\sigma_i \delta_i)) \pmb c_i$$

其中$$T_i = exp(-\Sigma_{j=1}^{i-1} \sigma_j \delta_j)$$，$$\delta_i = t_{i+1} - t_i$$是两个相邻采样点之间的距离。上述这个从一个集合的$$(\pmb c_i, \sigma_i)$$来d计算$$\hat C(\pmb{r})$$的函数是可微的。


**5. Optimizing a Neural Radiance Field**

在前面的章节里，我们描述了将一个scene建模为一个neural radiance field的核心内容，并介绍了如何从这个representation里渲染新的views。然而，这些对于生成state-of-the-art效果的图片来说并不够。我们介绍两个improvements来使得表示高分辨率的复杂scenes成为现实。第一个是对于输入coordinates的positional encoding，其可以帮助MLP拟合更高频率的函数（高频一般意味着更多的细节），第二个是使用了一种hierarchical的采样方法使得我们可以高效的采样这个高频的representation。


**5.1 Positional encoding**

尽管神经网络是universal function approximators，我们发现将网络$$F_{\Theta}$$直接作用在$$xyz\theta \phi$$上会导致渲染的结果对于color和geometry的高频变化效果不好（也就是变化很频繁的区域，而这往往是细节区域）。这和[On the spectral bias of neural networks]()$$里的关于神经网络倾向于学习low frequency function的观点一致。他们还说在输入网络之前，将输入使用high frequency function映射到更高维的空间，会使得网络能学习到含有高频变化的部分。

我们将这个想法用在neural scene representations的设计里，我们将$$F_{\Theta}$$重写为两个functions的复合函数：$$F_{\Theta} = F_{\Theta}^{'} \circ \gamma，$$F_{\Theta}^{'}$$是需要被学习的，而$$\gamma$$不需要。这样操作会显著提升效果，由fig 4可以看出。这里$$\gamma$$是一个从$$\mathbb R$$到更高维空间$$\mathbb R^{2L}$$的映射，而
$$F_{\Theta}^{'}$$还是个MLP。我们使用的encoding function是：

$$\gamma(p) = (sin(2^0 \pi p), cos(2^0 \pi p), \cdots, sin(2^{L-1}\pi p), cos(2^{L-1}\pi p))$$

上述的$$\gamma$$对于输入的3D坐标$$\pmb{x}$$里的每个坐标分别使用（这些坐标在被$$\gamma$$操作之前就被归一化到了$$\left[-1,1\right]$$之间），对于表示相机角度的Cartesian viewing direction unit vecotr $$\pmb{d}$$的三个坐标也分别使用（这个unit vector的三个坐标本身就在$$\left[-1,1\right]$$之间了）。在这篇文章里，对于$$\pmb{x}$$，$$L=10$$，对于$$\pmb{d}$$，$$L=4$$。

在Transformer里也用到了相似的encoding的方式，在那里叫做positional encoding。但是Transformer用它的原因和我们这里的截然不同（它是为了给无法识别顺序的网络框架输入的sequence的顺序信息）。而我们这里用到的原因是通过将输入映射到高维空间里，使得MLP能够更加轻易的拟合高频函数。

![3]({{ '/assets/images/NERF-3.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3 view-dependent emitted radiance的一个可视化。我们的neural radiance field representation输入为3D locations和2D viewing direction，输出为RGB color。这里，我们对于Ship scene，在neural representation里可视化两个位置的directional color distributions。在(a)和(b)里，我们展示了两个3D points在不同camera角度下的appearance：一个3D points是船舷，另一个是水面。我们的方法预测了这两个点在不同相机角度下的appearance，在(c)里我们显示了在viewing directions构成的半球上，每个3D points的appearance是怎么连续变化的。*

![4]({{ '/assets/images/NERF-4.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4 这里我们展示了我们的方法是如何从view-dependent emitted radiance以及positional encoding上受益的。将view-dependent移除，那我们的模型就无法预测那些镜面反射的点。将positional encoding移除，我们的模型对于那些高频的geometry和texture的预测就会很差，从而导致一个过于光滑的appearance。*

**5.2 Hierarchical volume sampling**

我们的渲染策略是沿着每个camera ray稠密的采样$$N$$个query points，来evaluate neural radiance field network。这样的方式是不高效的：没有物体的free space以及被遮住的occluded regions，它们对于渲染的图片没有贡献，但却被重复的采样了很多次。我们从之前的volume rendering的论文里获取了灵感，提出了一个通过根据正比于点在最终渲染的图片上的贡献来采样点的方式提高渲染效率的策略。

和仅仅用一个网络来表示scene不一样，我们同时使用两个网络，一个coarse，一个fine。我们先利用stratified sampling的方式采样$$N_c$$个点，然后用第四节里提到的方法来evaluate这个coarse网络。我们将coarse网络里活得的alpha composited color $$\hat C_c (\pmb{r})$$表示为沿着这个ray的所有采样点的colors $$c_i$$的加权和：

$$\hat C_c (\pmb{r}) = \Sigma_{i=1}^{N_c} w_i c_i$$

其中$$w_i = T_i ( 1 - exp(-\sigma_i \delta_i))$$。

将这些weights归一化为：$$\hat w_i = w_i / \Sigma_{j=1}^{N_c} w_j$$，就产生了一个沿着ray的piecewise-constant PDF。之后再从这个distribution上采样$$N_f$$个点，最后利用第四届里的方法使用所有的$$N_f + N_c$$个采样点来计算$$\hat C_f (\pmb{r})$$。这个方法为我们认为会有更多有用信息的位置增加了更多的点。


**5.3 Implementation details**

我们为每个scene优化一个单独的neural continuous volume representation network。我们仅仅需要一系列该scene的RGB图片、每张图片对应的相机角度和intrinsic parameters，以及scene bounds（对于生成数据，我们有相机角度、intrinsic parameters和scene bounds的ground truth，而对于真实数据，我们使用COLMAP structure-from-motion package（[Structure-from-motion revisited]()）来估计这些参数）。在每个optimization iteration，我们从数据集里所有的pixels里随机采样一个batch的camera rays，然后利用第五节里说的方法query $$N_c$$个点给coarse network，query $$N_c + N_f$$个点给fine network。我们之后再用第四节里说的volume rendering procedure来为这两个点的集合渲染出这条ray的color。我们的loss仅仅是渲染出来的颜色和真实的pixel颜色之间的$$L_2$$距离：

$$\mathcal L = \Sigma_{\pmb r in \mathcal R} \left[ \lVert \hat C_c (\pmb{r}) - C(\pmb r) \rVert_2^2 + \lVert \hat C_f (\pmb{r}) - C(\pmb r) \rVert_2^2 \right]$$

其中$$\mathcal R$$是每个batch里的rays，$$C(\pmb r)$$，$$C_c (\pmb r)$$和$$C_f(\pmb r)$$分别是ray $$\pmb r$$的ground truth，coarse volume predicted和fine volume predicted RGB颜色。我们同样在loss里加入了coarse network，因为这样才能使得采样的distribution是有效的。

在真实的实验里，我们的batch size是4096 rays，$$N_c = 64$$，$$N_f = 128$$。我们使用了Adam优化器，learning rate开始设置为$$5 \times 10^{-4}$$，decay是 exponentially设置为$$5 \times 10^{-5}$$。其余的Adam参数为默认值。对于一个scene，使用单张NVIDIA V100 GPU需要循环100-300K个循环，需要1-2天的时间。


**6. Conclusion**

我们的工作使用MLP来表示objects和scenes，解决了之前工作的不足。我们将scenes表示为5D neural radiance fields（一个MLP函数，输入是3D location和2D viewing direction，输出是volume density和view-dependent emitted radiance），相比较于之前的训练CNN来输出discretized voxel representations的方法，这个方法能够生成更好的渲染结果。

尽管我们提出了一个hierarchical sampling的方法使得渲染更加的高效，但在高效优化neural radiance field和采样的方向还有很多工作要做。另一个将来的方向是interpretability：sampled representations比如说voxel grids和meshes允许推理和思考所渲染的图片的效果以及失败的情况，但是将scenes表示为MLP的参数之后，我们就无法分析这些了。我们相信这篇文章推动了基于现实世界的graphics pipeline的发展，因为真实的objects和scenes现在可以被表示为neural radiance fields了，而多个objects或者scenes可以组合称为更复杂的scenes。



### 2. [KeypointNeRF: Generalizing Image-based Volumetric Avatars using Relative Spatial Encoding of Keypoints](https://arxiv.org/pdf/2205.04992.pdf)

[POST](https://markomih.github.io/KeypointNeRF/)

*Marko Mihajlovic, Aayush Bansal, Michael Zollhoefer, Siyu Tang, Shunsuke Saito*

*ECCV 2022*


**Abstract**

使用像素对齐（pixel-aligned）的features来做基于图片的体素avatars（image-base volumetric avatars）能保证结果可以泛化到没有见过的实体和姿态上去。先前的工作全局空间编码（global spatial encoding）和多角度的geometric consistency（multi-view geometric consistency）来减少空间不确定性（spatial ambiguity）。然而，global encodings经常会对训练数据overfitting，而从稀疏的几个views之间也很很难学习到multi-view consistency。在这篇文章里，我们研究了现有的spatial encoding方法的问题，并提出了一个简单但是很有效的方法来从稀疏的views（也就是只有几张不同角度的照片）来构造高保真的（high-fidelity）volumetric avatars。一个关键的想法就是使用稀疏的3D Keypoints来编码相对的空间3D信息。这个方法对于角度的稀疏性以及跨数据集之间的差异性是鲁棒的。我们的方法在head reconstruction领域超过了sota的方法的效果。对于未知身份的human body reconstruction，我们也达到了和之前工作差不多的效果，而之前的工作是通过使用参数化的human body模型以及使用时序feature aggregation实现的。我们的实验证明之前的那些方法之所以会效果不好，很大程度上是因为spatial encoding的方式选的不好，因此我们提出了一个构造基于图片的高保真的avatars的新方向。


**1. Introduction**

可渲染的3D human representations对于social telepresence，mixed reality，entertainment platforms等来说都是很重要的一个部分。经典的基于mesh的方法需要空间稠密的多个角度的立体数据（dense multi-view stereo）：[]()，[]()，[]()；或者需要融合深度信息：[]()。这些方法的结果的保真性不高，因为获取准确的geometry reconstruction的难度很大。最近，neural volumetric representations（[Neural Volumes: Learning Dynamic Renderable Volumes from Images](https://stephenlombardi.github.io/projects/neuralvolumes/)，[NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460392.pdf)）的提出使得高保真的avatar reconstruction成为了可能，特别是当准确的几何特性无法获得的时候（比如说头发的几何特性）。通过加入human-specific parametric shape models，获取大批的不同角度的数据只需要通过简单的设置相机参数就可以通过模型来获得了。然而，这些基于学习的方法是subject-specific的（也就是说每个模型只能针对某个具体的场景，因为实际上模型并没有学习到场景内到底有什么物体以及它的特性等这种high-level的semantic information，Nerf学习到的是一个不具有semantic information的整个空间的基于角度的颜色感知），而且需要对于每个场景训练好几天的时间，这样就大大限制了模型的可扩展性（scalability）。如果想要volumetric avatars迅速普及，需要能够做到，用户拿着手机给他自己从几个角度来几张自拍，传入网络里，然后很快就能构建出一个avatar出来。为了达到这个目标，我们需要能够从两三个views里学到准确的基于图像的volumetric avatars。

fully convolutiona pixel-aligned features（也就是说输出的feature map和输入的图片的尺寸是一样的，从而feature map每个位置的各个通道组成的feature就代表输入图片这个像素点位置的feature）使用了多个尺度的信息，从而让很多2D的CV任务效果都变好了（包括[PixelNet: Representation of the pixels, by the pixels, and for the pixels](https://arxiv.org/pdf/1702.06506.pdf)，[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.pdf)，[Fully Convolutional Networks for Semantic Segmentation
](https://openaccess.thecvf.com/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)，[Stacked Hourglass Networks for
Human Pose Estimation](https://islab.ulsan.ac.kr/files/announcement/614/Stacked%20Hourglass%20Networks%20for%20Human%20Pose%20Estimation.pdf)）。应用在occupancy和texture领域（[Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization]()）以及neural radiance fields领域（[Neural human performer: Learning generalizable radiance fields for human performance rendering]()，[Pva: Pixel-aligned volumetric avatars]()，[pixelnerf: Neural radiance fields from one or few images]()）的pixel-aligned representations使得这些方法能够泛化到没有见过的主体上去。pixel-aligned neural fields通过一个像素位置（pixel location）和空间编码函数（spatial encoding function）来推断field quantities，加入spatial encoding function是为了避免ray-depth不确定性。在这篇文章里，各种不同的spatial encoding functions都被研究了：[Portrait neural radiance fields from a single image]()，[Arch++: Animation-ready clothed human reconstruction revisited]()，[Arch: Animatable reconstruction of clothed humans]()，[Pva: Pixel-aligned volumetric avatars]()。然而，spatial encoding的效用并没有被完全理解。在这篇文章里，我们详细分析了使用pixel-aligned neural radiance fields方法为human faces建模的spatial encodings方法。我们的实验证明spatial encoding的选择影响了reconstruction的效果，也影响了模型泛化到未见过的人或者未见过的角度上的效果。那些使用了深度信息的模型会容易过拟合，过分拟合训练数据的分布，而使用multi-view stereo约束的方法对于稀疏的views的情况的效果就很不好了。

我们提出一个简单但是有效的方法利用了稀疏的3D keypoints来解决上述这些方法里存在的问题。3D keypoints可以很简单的用一个训练好的2D keypoints detector，比如说[Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://openaccess.thecvf.com/content_cvpr_2017/papers/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.pdf)，和简单的multi-view triangulation来计算而得，[Nultiple view geometry in computer vision](http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf)。我们将3D keypoints看作spatial anchors，然后将其它部分的相对3D空间信息（relative 3D spatial information）都编码到这些keypoints上去。和global spatial encoding（[Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization]()，[Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization]()，[pixelnerf: Neural radiance fields from one or few images]）不同，relative spatial encoding并不需要知道相机参数。这个特性使得基于relative spatial encoding的方法对于姿态（pose）来说很鲁棒。3D keypoints还让我们可以使用同样的方法来处理faces和bodies。我们的方法使得我们可以从两三个views上生成volumetric facial avatars，而且效果是sota的，而且提供更多的views会让效果更好。我们的方法同样在human body reconstruction这个任务上和Neural Human Performer(NHP)（[Neural human performer: Learning generalizable radiance fields for human performance rendering]()）达到了差不多的效果。NHP依赖于准确的参数化的human body模型，以及时序的feature aggregation，而我们的方法仅仅需要3D keypoints。我们的方法同时也不过过拟合训练数据的分布。我们可以将训练好的模型（不需要任何微调）对于从未见过的人的照片，生成其的avatars。我们认为这种能够对没见过的数据（人脸）生成volumetric avatars的能力是源于我们对于spatial encoding的选择。我们的主要贡献如下：

* 一个简单的利用3D keypoints的框架，可以从两三个不同角度的输入图片生成volumetric avatars，而且还可以泛化到从未见过的人。
* 对于已有的spatial encodings进行了一个细致的分析。
* 我们的模型在一个数据集上训练好之后，对于任何其他的没有见过的人的几个角度的照片，仍然可以生成volumetric avatars，这样好的泛化效果是之前的工作所没有的。


**2. Related work**

我们的目标是从很少的views（甚至两张）来构造高保真的volumetric avatars。

*Classical Template-based Techniques* 

早期的avatar reconstruction需要大量的图片，以及将一个template mesh匹配到3D point cloud等操作。[Real-time facial animation with image-based dynamic avatars]()提出使用一个粗造的geometry来为face建模，并使用一个可变性的hair model。[Avatar digitization from a single image for real-time rendering]()从一个数据集里获取hair model的信息，并组合facial和hair的细节信息。[Video based reconstruction of 3d people models]()使用基于轮廓的建模方法基于一个单目的视频来获取一个body avatar。对于geometry和meshes的以来限制了这些方法的发挥，因为某些区域比如hair，mouth，teeth等的geometry非常难以获得。


*Neural Rendering*

neural rendering（[State of the Art on Neural Rendering](https://arxiv.org/pdf/2004.03805.pdf)，[Advanced in neural rendering](https://arxiv.org/pdf/2111.05849.pdf)）通过直接从原始的sensor measurements里学习image formation的组成部分来解决那些template-based的方法解决不了的问题。2D neural rendering方法，比如DVP（[Deep Video Portraits](https://arxiv.org/pdf/1805.11714.pdf)），ANR（[Anr: Articulated neural rendering for virtual avatars](https://openaccess.thecvf.com/content/CVPR2021/papers/Raj_ANR_Articulated_Neural_Rendering_for_Virtual_Avatars_CVPR_2021_paper.pdf)），SMPLpix（[Smplpix: Neural avatars from 3d human models](https://arxiv.org/pdf/2008.06872.pdf)）使用了surface rendering和U-Net结构来减小生成的图片和真实图片之间的差距。这些2D方法的一个缺点是它们无法以一种时序紧密联系的方式（temporally coherent manner）来生成图片里物体新的角度的图片。Deep Appearance Models（[Deep appearance models for face rendering]()）联合使用了一个粗造的3D mesh和基于角度的texture mapping来从稠密的各个角度的images以监督学习的方式来学习face avatars。使用一个3D的mesh极大地帮助了新视角生成，但是这个方法在某些区域就很难生成真实的效果好的图片，比如说hair或者mouth里面，因为这些地方的3D mesh本身就缺乏对应的几何信息，因为根本就很难获取。现在的sota方法比如说NeuralVolumes（[Neural volumes: learning dynamic renderable volumes from images]()）和Nerf（[Nerf: Representing scenes as neural radiance fields for view synthesis]()）使用了differentiable volumetric rendering而不是meshes。这些方法能够在那些很难估计3D geometry的区域仍然有好的效果。之后还有更多的改进版本进一步改善了效果：[Flame-in-nerf: Neural control of radiance fields for free view face animation]()，[Mixture of volumetric primitives for efficient neural rendering]()，[Learning compositional radiance fields of dynamic human heads]()。但这些方法需要稠密的multi-view的图片，并且一个模型只针对一个场景，需要训练好几天才能有好的效果。


*Sparse View Reconstruction*

大规模的模型部署需要用户提供两到三张不同角度的照片，就可以生成avatars。通过使用pixel-aligned features（[Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization]()，[Pifuhd: Multi-level pixel-aligned implicit function for high-resolution 3d human digitization]()），我们就可以训练这样的网络了：从几个views来生成avatars的网络。各种不同的方法都将multi-view约束、pixel-aligned features与NeRF结合起来来学习可泛化的view-synthesis：[Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo]()，[Stereo radiance fields(srf): Learning view synthesis from sparse views of novel scenes]()，[Ibrnet: Learning multi-view image-based rendering]()，[pixelnerf: Neural radiance fields from one or few images]()。在这篇文章里，我们会发现这些方法对于没见过的human face和bodies很难只从几张views就生成细节很好的结果。


*Learning Face and Body Reconstruction*


### 3. [3D Moments from Near-Duplicate Photos](https://3d-moments.github.io/static/pdfs/3d_moments.pdf)

[POST](https://3d-moments.github.io/)

*CVPR 2022*

### 4. [NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing](https://arxiv.org/pdf/2207.11911.pdf)

[POST](https://zju3dv.github.io/neumesh/)

*ECCV 2022 Oral*

### 5. [Eikonal Fields for Refractive Novel-View Synthesis](https://eikonalfield.mpi-inf.mpg.de/paper/eikonalfield_2022.pdf)

[POST](https://eikonalfield.mpi-inf.mpg.de/)

*SIGGRAPH 2022*


### 6. [Block-NeRF: Scalable Large Scene Neural View Synthesis](https://waymo.com/intl/zh-cn/research/block-nerf/)

[POST](https://waymo.com/intl/zh-cn/research/block-nerf/)

*CVPR 2022*

### 7. [AutoRF: Learning 3D Object Radiance Fields from Single View Observations](https://sirwyver.github.io/AutoRF/)

[POST](https://sirwyver.github.io/AutoRF/)

*CVPR 2022*


### 8. [GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation](https://openaccess.thecvf.com/content/CVPR2022/papers/Deng_GRAM_Generative_Radiance_Manifolds_for_3D-Aware_Image_Generation_CVPR_2022_paper.pdf)

*CVPR 2022*


### 9. [NeRFRen: Neural Radiance Fields with Reflections](https://bennyguo.github.io/nerfren/)

[POST](https://bennyguo.github.io/nerfren/)

*CVPR 2022*


### 10. [Learning Object-Compositional Neural Radiance Field for Editable Scene Rendering](https://zju3dv.github.io/object_nerf/)

[POST](https://zju3dv.github.io/object_nerf/)

*ICCV 2021*


### 11. [ARF: Artistic Radiance Fields](https://www.cs.cornell.edu/projects/arf/)

[POST](https://www.cs.cornell.edu/projects/arf/)

*ECCV 2022*


### 12. [StylizedNeRF (Jittor): Consistent 3D Scene Stylization as Stylized NeRF via 2D-3D mutual learning](https://openaccess.thecvf.com/content/CVPR2022/papers/Huang_StylizedNeRF_Consistent_3D_Scene_Stylization_As_Stylized_NeRF_via_2D-3D_CVPR_2022_paper.pdf)

[CODE](https://github.com/IGLICT/StylizedNeRF)

*CVPR 2022*





## Reconstruction

### 1. [Interacting Attention Graph for Single Image Two-Hand Reconstruction](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Interacting_Attention_Graph_for_Single_Image_Two-Hand_Reconstruction_CVPR_2022_paper.pdf)

*CVPR 2022*


### 2. [Human Mesh Recovery from Multiple Shots](https://openaccess.thecvf.com/content/CVPR2022/papers/Pavlakos_Human_Mesh_Recovery_From_Multiple_Shots_CVPR_2022_paper.pdf)

*CVPR 2022*


### 3. [De-rendering 3D Objects in the wild](https://www.robots.ox.ac.uk/~vgg/research/derender3d/)

[POST](https://www.robots.ox.ac.uk/~vgg/research/derender3d/)

*CVPR 2022*


### 4. [OcclusionFusion: Occlusion-aware Motion Estimation for Real-time Dynamic 3D Reconstruction](https://arxiv.org/pdf/2203.07977.pdf)

[POST](https://wenbin-lin.github.io/OcclusionFusion/)

*CVPR 2022*


### 5. [NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video](https://zju3dv.github.io/neuralrecon/)

[POST](https://zju3dv.github.io/neuralrecon/)

*CVPR Oral & Best Paper Candidate*


### 6. [Robust Equivariant Imaging: a fully unsupervised framework for learning to image from noisy and partial measurements](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Robust_Equivariant_Imaging_A_Fully_Unsupervised_Framework_for_Learning_To_CVPR_2022_paper.pdf)

*CVPR 2022*




## Generative Models

### 1. [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*

*NIPS 2014*

**1. Abstract**

1. The abstract of this paper is very standard and unique. This abstract generally introduces the main concept of this paper, i.e., GAN, rather than introducing the problem and existing methods. This abstract is kind of like the Wikipedia introduction of some concept. If you are very confident that your paper is very novel and proposes a very useful idea, model or concept that can be recorded in this area, you can use this kind of writing style of abstract. This is very clear to the community, but not very clear to the people that do not know any about this area.

**2. Introduction**

2. This paper thinks that besides deep learning architectures, deep learning will learn the representations of data probability distributions over all kinds of data, including natural images, audio waveforms containing speech, symbols in natural language processing, etc. This idea is always held by Yoshua Bengio and his group.

3. Discriminative models in deep learning have already invovled into various kinds and solved many kinds of problems. But generative models, instead, still have lots of part remained mysterious. This due to the fact that in order to learn data distributions underlying generative models, you need many approximation methods to approximate the distribution in order to make calculation work. But this process will make the distribution unaccurate and even not working. So in this paper, they do not try to approximate the distributions, they use deep learning models to do this job.

4. Generative adversarial models will contain two parts: the generative model and the discriminator model. The generative model aims to generate data that the discriminator model can not distinguish apart from true training data, and the discriminator model aims to distinguish between them. The final goal is to let the generative model to generative plausible data that the distriminator model can not distinguish.

5. In this paper, they use MLP as models for the generative and discriminator models, and in this case, this generative adversarial model is called generative adversarial net (GAN). The input of the generative model is random noise (usually a Guassion noise), and they want the generative model to map this random noise distribution to any distribution we want to fitting, i.e., the training data distribution.


**3. Related work**

1. There are two kinds of former works on deep generative models. The first ones concentrated on building a probability distribution function, and these models are trained on maximizing the log likelihood, such as the deep Boltzmann machine. The biggest difficulty is that the calculation of sampling this distribution is hard, especially when the dimension is high. The other ones are called generative machines. They do not explicity construct the distribution, and they learn a model to approximate this distribution. There are intrinsic differences between these two kinds of methods. The first ones acutally learn the distribution, though using some kind of approximation method to make the calculation feasible, but in the end, you actually get a distribution, you can calculate the mean, variance and all kind of properties of this distirbution. But the other ones do not construct the distribution, and only learn a model to approximate this distribution. So in the end, we do not know what this distribution looks like. 

2. Variational Autoencoder (VAE) actually has similar ideas to this paper. And using a distriminator model to assist the generative model is also not novel, such as Noise-contrastive Estimation (NCE). 


**4. Adversarial nets**

1. Generator wants to learn the distribution of the input training data $$x$$, $$p_g$$. We give an example of GAN. Suppose there is a video game and it can generate images of the game, and now we want to learn a generator to generate the images of the game. Suppose that our display resolution is 4K, then each time we need to generate an image vector of length 4k. Each pixel could be considered as a random variable, thus this vector can be considered as a multi-dimensional ramdon variable of length 4k. We know that this vector is controled by the underlying game code, and this code is actually the underlying $$p_g$$ for this vector. Now how to let the generator to generate data? We define a prior on the input noise variable $$p_z(z)$$, this $$z$$ could be a 100 dimensional Guassion distribution with mean 0 and variable matrix I. The generator aims to map $$z$$ onto $$x$$, the generator model can be formed as $$G(z, \theta_g)$$. Return to our game example. In order to generate game images, one way is that we can conversly compile the game code, and know the underlying game code. In this way, we can acutally know how the game images are generated. This method can be considered similar to the methods described in the related work that aim to construct the underlying distribution. Another way is that we neglect the underlying code, we guess that this game is not very complicated, thus maybe there are only 100 variables underlying are used to control the generation of images. Thus we contructed a known distribution of dimension 100 $$z$$, and due to the fact that MLP is able to approximate any functions, we let this MLP to map the input $$z$$ into the image space $$x$$. 

2. The discriminator $$D(x, \theta_d)$$ is also an MLP, and its output is a scalar value, for distinguishing between the true data and generated data. Thus actually $$D$$ is a two-label classifier. We know where our input data is from (true or generated), thus we can give them labels. 

3. The training process involves training D and G simultaneously:

$$\min_{G}\max_{D} V(D,G) = E_{x \sim p_{data}}\left[log D(x)\right] + E_{z \sim p_z(z)}\left[log(1-D(G(z)))\right]$$

This is a two-player minimax game. When the G and D reach a stable state, they are actually arrive at a Nash equilibrium.

4. Look at figure1. This example is simple. The input noise distribution of $$z$$ is a uniform distribution (the lower line of $$z$$ has equal intervals), and our $$x$$ is a Guassian distribution (the black dotted line). The arrows between the line of $$x$$ and $$z$$ is the mapping, i.e., generator. From plot (a) in figure1, we see that, at first, the mapping maps $$z$$ to the behind part of $$x$$, so the output distribution of this mapping is the green line in the plot, and the blue line is the discriminator. The next step, we update the discriminator, we can see that, the discriminator choose the margin in between of the mean of the true distribution and the generator output distribution, as shown in plot (b). Then we update the generator, we can see that the generator output distribution will move closer to the true distribution, in plot(c). And then we update discriminator, update generator, ..., and finally ,we get plot(d), in that the generaotr output will be the same to the true distribution and the discriminator will show a horizontal line indicating that it can not distinguish between true and generated data. 

![GAN]({{ '/assets/images/GAN-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. An example of training process of a GAN.*

5. Each training iteration of the training algorithm of GAN involves two steps. In the first step, there is a for loop, loops over $$m$$ times. And in each loop, we get $$m$$ true data and $$m$$ generated data from the generator, and then calculate the gradient of the minimax loss defined above to update the discriminator. In the second step, we sample another $$m$$ samples from the generator to calculate the gradient of the minimax loss with respect to the generator for updating it. The for loop iteration time $$k$$ in the first step, is a hyperparameter of this algorithm. In each training iteration, we need the generator and distriminator be at the same levle, i.e., the performance of one should be be much better than the other. Only in this case, we can make the training trackable. The decision of whether a GAN is trained well is also a difficult question, i.e., the iteration time of the training algorithm. This still remains a hot area and unsolved.

6. One training tip is that, since when the discriminator trains well, the $$log(1-D(G(z)))$$ will be close to 0, thus the gradient will not be applicable, instead of minimizing $$log(1-D(G(z)))$$, we maxmizing $$log(D(G(z)))$$.


**4. Theoretical Results**

1. There is a global optimum for the generator, $$p_g = p_{data}$$. Firstly, we see a lemma firstly. **Lemma**: if the generator is fixed, then the optimal discriminator will be

$$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$

i.e., the error probability (the training criterion of discriminator) of the distriminator will be the smallest. **Theorem**: Setting $$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$ as in Lemma in the equation $$\min_{G}\max_{D} V(D,G) = E_{x \sim p_{data}}\left[log D(x)\right] + E_{z \sim p_z(z)}\left[log(1-D(G(z)))\right]$$, we can get $$C(G) = E_{x \sim p_{data}}\left[log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + E_{x \sim p_g}\left[log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right] $$. Then $$C(G)$$ get its minimum when $$p_g = p_{data}$$.

2. The algorithm described above is able to train the discriminator and the generator, i.e., the training algorithm is convergent. 


**5. Experiments**

The experiments in this paper is not good enough and quite simple.


**6. Conclusion**

GAN is actually an unsupervised learning setting, but it leverages supervised learning framework by using the label of true or generated data for training. This idea insights the future self-supervised learning frameworks.


**The conclusion of writing style of this paper**

This paper proposes a very novel idea and model, thus it elaborates the details of design and ideas behind GAN very clearly. The authors are very ambitious and are confident that this work will be recorded in this area. So the whole writing style is kind of like Wikipedia, i.e., the very detailed description of the proposed model, with a little mention of existing works and comparison with other works. Also, experiments are few.

If you are confident that your work is novel and very important, you can use this kind of writing style. Otherwise, you need to describe clearly the existing works and your contribution to this problem.


### 2. [Auto-Encoding Variational Bayes](https://openreview.net/forum?id=33X9fd2-9FyZd)

*ICLR 2014*


### 3. [Unsupervised 3D Shape Completion through GAN Inversion](https://graphics.stanford.edu/courses/cs348n-22-winter/PapersReferenced/Zhang%20et%20al.%20-%202021%20-%20Unsupervised%203D%20Shape%20Completion%20through%20GAN%20Inversion.pdf)

*CVPR 2021*


### 4. [EG3D: Efficient Geometry-aware 3D Generative Adversarial Networks](https://nvlabs.github.io/eg3d/)

[POST](https://nvlabs.github.io/eg3d/)

*CVPR 2022*

---
