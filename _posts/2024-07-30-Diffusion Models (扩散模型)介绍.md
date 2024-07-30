---
layout: post
comments: false
title: "installation of various systems, environments and platforms"
date: 2024-07-30 01:09:00

---

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

Diffusion models可以说是现如今机器学习领域最火的模型，其不仅在生成模型领域性能上大大超越了之前的那些前辈们，包括GAN，VAE，flow-based模型等，还在多模态领域也引起了革命，比如StableDiffusion等。而且由于diffusion模型，尤其是结合文本和图像的多模态diffusion模型，其所学习到的feature具有很好的semantics可解释性，所以pre-trained好的diffusion模型现在也被用于很多任务之中，最火的比如说基于diffusion模型loss改进的score distillation sampling (SDS)的各种diffusion-based 3D representation learning（比如DreamFusion，Magic3D，zero-1-to-3等）都具有惊艳的效果。而本文，则从原理到应用到代码，综合的介绍一下diffusion models，这个如今AI领域的核心模型。

在开始正式的介绍之前，首先来看看diffusion model和之前的那些生成模型的对比。

![1]({{ '/assets/images/diffusion_1.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}
*来自于[Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)*

GAN的缺点在于比较难以训练（discriminator过好的话，generator更新的梯度会很小，训练不动，而discriminator过差的话，generator的更新梯度方向就是错误的，训练结果不对。即使是后续改进的Wasserstein GAN也没有完全解决这个问题）。而VAE的缺点在于需要假设分布是简单的（比如高斯分布），否则难以求解。而flow-based模型则需要transformation functions是invertible的，也限制了模型的flexibility。

而diffusion模型不同于之前的生成模型范式，提出了个新的框架，其受到了物理学里的non-equilibrium thermodynamics的启发，简单举个例子，在水中滴入墨水，那么墨水就会逐渐扩散到整杯水都会变黑。diffusion模型的思路就是，在”干净“的原输入数据（比如说图片）上逐渐加上噪声，一般来说就是0均值的高斯噪声，每一步都是在前一次的基础上再往上添加。如果能够控制好这些高斯噪声的方差大小的话，理论上足够长的时间，我们就可以得到一个均值为0，方差为1的标准高斯分布了（这样说不太准确，因为最终得到的是一个被污染了的数据，准确来说是，最终我们得到的这个被污染的数据，其分布服从均值为0，方差为1的高斯分布）。但我们想要的并不是高斯噪声，我们想要的是能生成和输入数据”长得像“的新的干净的数据。而diffusion模型的做法是，从一个均值为0，方差为1的高斯分布出发，采样一个值，然后逐步让一个Denoiser来将这个值里的噪声逐步去除掉，也就是将之前的加噪声的过程逆过来，如果这个逆过程能够很好的模拟之前的添加噪声的逆过程，那么很多步之后，我们就应该能够得到去除掉噪声的干净的数据了。因为这个去噪过程的输入是随即从标准高斯分布采样的值，所以具有随机性，从而这样得到的干净的数据也是具有随机性的，也就是说每次采样都可以有不同的生成数据输出了。上述这两个过程，就分别叫做diffusion model的前向过程，和反向过程，也叫做加噪过程和去噪过程。

![2]({{ '/assets/images/diffusion_2.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}
*来自于[Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)*

一般情况下，上述过程里的denoiser，由一个神经网络来实现。对于每个”干净“的输入图片$$x_0$$，先采样一个时间$$t$$，然后给$$x_0$$添加噪声，这个噪声的大小和$$t$$有关，然后将加噪后的数据，以及$$t$$，同时输入给denoiser，其输出和$$x_0$$之间的差别，就是整个训练的loss。

介绍diffusion模型的博客和论文很多，而最重要的是提出diffusion模型的两篇论文：[DDPM](https://arxiv.org/pdf/2006.11239)和[NCSN](https://arxiv.org/pdf/1907.05600)，这两篇论文分别从不同的角度介绍了diffusion模型，也是接下来理论介绍diffusion模型的两个角度。

## 1. 基本原理 （DDPM）

如之前所说，diffusion模型可以从两个角度来理解其原理，其中DDPM（diffusion  probabilistic models）的解释更容易理解，从而先从此角度来说。

## (1) 前向扩散过程（forward diffusion process）

从一个给定的数据分布$$q(x)$$里采样一个数据$$x_0$$，$$x_0 \sim q(x)$$（$$x_0$$就可以被理解为我们手里已有的数据，而$$q(x)$$是已有的数据潜在的数据分布，是未知的），扩散模型的一个前向扩散过程，就是不断地给数据添加高斯噪声的过程，假设一共添加了$$T$$步，那么就会得到一系列noisy的数据：$$x_1, x_2, \cdots, x_T$$，而每一步具体添加多少噪声，则是由一组超参数$$\lbrace \beta_t \in (0,1) \rbrace_{t=1}^T$$决定的:

$$q(x_t \vert x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I}), \  \text{where}t=1,2,\cdots,T$$

可以看出来，$$x_0, x_1, \cdots, x_T$$是个马尔可夫链，从而$$x_1, \cdots, x_T$$在$$x_0$$条件下的联合分布就可以写成如下形式：

$$q(x_{1:T} \vert x_0) = \Pi_{t=1}^T q(x_t \vert x_{t-1})$$

原本“干净”的数据$$x_0$$，随着时间的推进，增加的噪声越来越多，逐渐就失去了原本的structure，在合适的$$\lbrace \beta_t \in (0,1) \rbrace_{t=1}^T$$的设置下，以及$$T \rightarrow \infty$$的情况下，$$x_T \sim \mathcal{N}(0, \mathbf{I})$$。

而实际上，前向过程的一个良好的性质是，我们可以得到$$x_t$$与$$x_0$$的closed form的关系（$$1 leq t \leq T$$）：

记$$\alpha_t = 1 - \beta_t$$，$$\bar{\alpha_t} = \Pi_{i=1}^t \alpha_i$$，那么：

\begin{aligned*}

x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \  \text{where} \epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-1} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1}, \  \text{where} \epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\

\end{aligned*}






























