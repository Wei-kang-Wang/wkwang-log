---
layout: post
comments: false
title: "扩散模型"
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

### (1) 前向扩散过程（forward diffusion process）

从一个给定的数据分布$$q(x)$$里采样一个数据$$x_0$$，$$x_0 \sim q(x)$$（$$x_0$$就可以被理解为我们手里已有的数据，而$$q(x)$$是已有的数据潜在的数据分布，是未知的），扩散模型的一个前向扩散过程，就是不断地给数据添加高斯噪声的过程，假设一共添加了$$T$$步，那么就会得到一系列noisy的数据：$$x_1, x_2, \cdots, x_T$$，而每一步具体添加多少噪声，则是由一组超参数$$\lbrace \beta_t \in (0,1) \rbrace_{t=1}^T$$决定的:

$$q(x_t \vert x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I}), \  \text{where} \  t=1,2,\cdots,T$$

可以看出来，$$x_0, x_1, \cdots, x_T$$是个马尔可夫链，从而$$x_1, \cdots, x_T$$在$$x_0$$条件下的联合分布就可以写成如下形式：

$$q(x_{1:T} \vert x_0) = \Pi_{t=1}^T q(x_t \vert x_{t-1})$$

原本“干净”的数据$$x_0$$，随着时间的推进，增加的噪声越来越多，逐渐就失去了原本的structure，在合适的$$\lbrace \beta_t \in (0,1) \rbrace_{t=1}^T$$的设置下，以及$$T \rightarrow \infty$$的情况下，$$x_T \sim \mathcal{N}(0, \mathbf{I})$$。

而实际上，前向过程的一个良好的性质是，我们可以得到$$x_t$$与$$x_0$$的closed form的关系（$$1 \leq t \leq T$$）：

记$$\alpha_t = 1 - \beta_t$$，$$\bar{\alpha_t} = \Pi_{i=1}^t \alpha_i$$，那么：

$$
\begin{align}

x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \  \text{where} \  \epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-1} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1}, \  \text{where} \  \epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + (\sqrt{1-\alpha_{t-1}}\epsilon_{t-2}) + \sqrt{1-\alpha_t}\epsilon_{t-1}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t}\alpha_{t-1}}\bar{\epsilon_{2}}, \  \text{where} \  \bar{\epsilon_{2}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \cdots \\
&= \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \bar{\epsilon_{t}}, \  \text{where} \  \bar{\epsilon_{t}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
\end{align}
$$

上述推导用到了一个结论：$$x \sim \mathcal{N}(\mathbf{0}, \sigma_1^2 \mathbf{I}), y \sim \mathcal{N}(\mathbf{0}, \sigma_2^2 \mathbf{I})$$，那么$$x+y \sim \mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2) \mathbf{I})$$。

上述过程的第一行里，$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \  \text{where} \  \epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$用到了一个重参数化技巧（reparametrization trick），将$$x_t$$的随机性，转移到了$$\epsilon_{t-1}$$上，而$$\epsilon_{t-1}$$是从一个无可学习参数的标准高斯分布采样来的。这样做的好处是，因为在计算loss反向传播的时候，需要计算loss对于$$x_t$$的导数，如果$$x_t$$是采样来的，采样过程是离散的而且不可导，这样就没法算了。

由上述推导，可以得到在$$x_0$$条件下$$x_t$$的分布是个高斯分布：

$$q(x_t \vert x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha_t}}x_0, (1 - \bar{\alpha_t})\mathbf{I})$$

一般来说，对于超参数$$\lbrace \beta_t \in (0,1) \rbrace_{t=1}^T$$的设置是，随着$$t$$的增大，噪声的程度可以越来越大，也就是说$$\beta_1 < \beta_2 < \cdots < \beta_T$$，从而$$\alpha_1 > \alpha_2 > \cdots > \alpha_T$$，且$$\bar{\alpha_1} > \bar{\alpha_2} > \cdots > \bar{\alpha_T}$$。

### (2) 反向扩散过程（reverse diffusion process）

如果我们可以建模上述前向扩散过程的逆过程，也就是建模$$q(x_{t-1} \vert x_t)$$的话，那么基于一个从标准高斯分布采样得到的纯噪声数据$$x_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$，就可以一步步回退，逐步去除噪声，最终得到一个新的”干净“的$$x_0$$，而因为$$x_T$$的采样是具有随机性的，所以每次得到的$$x_0$$也不一样，这样就可以源源不断地生成”干净“的数据了。

我们有如下结论：如果$$\beta_t$$足够小的话，那么如果$$q(x_t \vert x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$$是个高斯分布，那么$$q(x_{t-1} \vert x_t)$$也是个高斯分布。但这个高斯分布的均值和方差无法解析表达。从而我们考虑用一个带有可学习参数$$\theta$$的模型$$p_{\theta}$$来对$$q(x_{t-1} \vert x_t)$$进行建模：

$$p_{\theta}(x_{t-1} \vert x_t) = \mathcal{x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t, t)}$$

我们同样假设反向扩散过程也是个马尔可夫链，也就是说：

$$q(x_{0:T}) = q(x_T) \Pi_{t=1}^T q(x_{t-1} \vert x_t) = q(x_T) \Pi_{t=1}^T p_{\theta}(x_{t-1} \vert x_t)$$

如果我们在分布$$q(x_{t-1} \vert x_t)$$上再加上条件$$x_0$$的话，也就是考虑分布$$q(x_{t-1} \vert x_t, x_0)$$，有如下结果：

$$
\begin{align}

q(x_{t-1} \vert x_t, x_0) &= q(x_t \vert x_{t-1}, x_0) \frac{q(x_{t-1} \vert x_0)}{q(x_t \vert x_0)} = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I}) \frac{\mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha_{t-1}}}x_0, (1-\bar{\alpha_{t-1}})\mathbf{I})}{\mathcal{N}(x_{t}; \sqrt{\bar{\alpha_{t}}}x_0, (1-\bar{\alpha_{t}})\mathbf{I})}\\
& \propto exp(-\frac{1}{2} \frac{(x_t - \sqrt{\alpha_t} x_{t-1})^T(x_t - \sqrt{\alpha_t} x_{t-1})}{\beta_t} + \frac{1}{2} \frac{(x_{t-1} - \sqrt{\bar{\alpha_{t-1}}} x_{0})^T(x_{t-1} - \sqrt{\bar{\alpha_{t-1}}} x_{0})}{1 - \bar{\alpha_{t-1}}}) - \frac{1}{2} \frac{(x_{t} - \sqrt{\bar{\alpha_{t}}} x_{0})^T(x_{t} - \sqrt{\bar{\alpha_{t}}} x_{0})}{1 - \bar{\alpha_t}})\\
&= exp(-\frac{1}{2}((\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha_{t-1}}})x_{t-1}^Tx_{t-1} - (\frac{2 \sqrt{\alpha_t}}{\beta_t}x_t^T + \frac{2 \sqrt{\bar{\alpha_{t-1}}}}{1-\bar{\alpha_{t-1}}}x_0^T)x_{t-1})+ C(x_0, x_t)), \  \text{where} \  C(x_0, x_t) \  \text{is} \  \text{a} \  \text{constant} \  \text{w.r.t.} \  x_{t-1}
\end{align}
$$

从这个形式可以看出，$$q(x_{t-1} \vert x_t, x_0)$$也满足高斯分布，$$q(x_{t-1} \vert x_t, x_0) = \mathcal{N}(x_{t-1} \vert \tilde{\mu}(x_t, x_0), \tilde{\beta_t} \mathbf{I})$$，其中

$$\tilde{\beta_t} = 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha_{t-1}}}) = \frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_{t}}} \beta_t, \tilde{\mu}(x_t, x_0) = (\frac{2 \sqrt{\alpha_t}}{\beta_t}x_t + \frac{2 \sqrt{\bar{\alpha_{t-1}}}}{1-\bar{\alpha_{t-1}}}x_0) / (\frac{\alpha_t}{\beta_t} + \frac{1}{1-\bar{\alpha_{t-1}}}) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_t}}x_t + \frac{\sqrt{\bar{\alpha_{t-1}}}\beta_t}{1-\bar{\alpha_t}}x_0$$

而之前我们有结论：$$x_t = \sqrt{\bar{\alpha_t}} x_0 + \sqrt{1 - \bar{\alpha_t}} \bar{\epsilon_{t}}, \  \text{where} \  \bar{\epsilon_{t}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$，也就是，$$x_0 = \frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}}\bar{\epsilon_{t}})，带入$$\tilde{\mu}(x_t, x_0)$$上面的结果，可得：$$\tilde{\mu}(x_t, x_0) = \frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}} \bar{\epsilon_{t}})$$，与$$x_0$$无关了，因此也可以记为$$\tilde{\mu}_t$$。

上述的推导过程表明，$$q(x_{t-1} \vert x_t, x_0)$$可以有closed-form的结果。先把这个结果放在一旁。

### (3) 损失函数

既然我们希望用一个带参数的分布$$p_{\theta}$$来对$$q(x_{t-1} \vert x_t)$$进行近似，那么我们就需要定义损失函数来让$$p_{\theta}$$与$$q(x_{t-1} \vert x_t)$$尽可能靠近，如下的推导对$$p_{\theta}$$的形式不做任何假设。

我们可以发现，实际上$$p_{\theta}(x_{t-1} \vert x_t)$$很像VAE里的$$q_{\phi}(z \vert x)$$，而$$q(x_{t-1} \vert x_t)$$则是VAE的decoder的未知后验分布$$p_{\theta}(z \vert x)$$（这里的$$\theta$$和之前的$$\theta$$不一样，这里指的是VAE里的分布）。从而根据这个观察，我们也可以仿照VAE的损失函数，定义如下的损失函数：

$$
\begin{align}
-log p_{\theta}(x_0) &\leq -log p_{\theta}(x_0) + \textbf{D}_{\textbf{KL}} (q(x_{1:T} \vert x_0) \Vert p_{\theta}(x_{1:T} \vert x_0)) = -log p_{\theta}(x_0) + \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ log \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{1:T} \vert x_0)} \right] \\
&= -log p_{\theta}(x_0) + \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ log \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})} + log p_{\theta}(x_0) \right] = \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ log \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})} \right]
\end{align}
$$

对上个式子两侧以$$q(x_0)$$为分布取期望，则有：

$$ - \mathop{\mathbb{E}}\limits_{q(x_0)} \left[ log p_{\theta}(x_0) \right] \leq \mathop{\mathbb{E}}\limits_{q(x_{0:T})} \left[ log \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})} \right]$$

记上面式子的右侧为$$\mathcal{L}_{VLB}$$。

我们再来考虑$$q(x_0)$$和$$p_{\theta}(x_0)$$之间的cross entropy：

$$
\begin{align}
\mathcal{L}_{CE} &= -\mathop{\mathbb{E}}\limits_{q(x_0)} \left[ log p_{\theta}(x_0) \right] = -\mathop{\mathbb{E}}\limits_{q(x_0)} \left[ log \int p_{\theta}(x_{0:T}) dx_{1:T} \right] = -\mathop{\mathbb{E}}\limits_{q(x_0)} \left[ log \int q(x_{1:T} \vert x_0) \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \vert x_0)} dx_{1:T} \right] \\
&= -\mathop{\mathbb{E}}\limits_{q(x_0)} \left[ log -\mathop{\mathbb{E}}\limits_{q(x_{1:T} \vert x_0)} (\frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \vert x_0)}) \right] \leq -\mathop{\mathbb{E}}\limits_{q(x_0)} \mathop{\mathbb{E}}\limits_{q(x_{1:T} \vert x_0)} log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \vert x_0)} = - \mathop{\mathbb{E}}\limits_{q(x_{0:T} \vert x_0)} log \frac{p_{\theta}(x_{0:T})}{q(x_{1:T} \vert x_0)} = \mathcal{L}_{VLB}
\end{align}
$$

回到之前的结果：$$-log p_{\theta}(x_0) \leq \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ log \frac{q(x_{1:T} \vert x_0)}{p_{\theta}(x_{0:T})} \right]$$，记右侧这个式子为$$\mathcal{L}_{VLB}^{\ast}$$，也就是说，$$\mathcal{L}_{VLB} = \mathop{\mathbb{E}}\limits_{q(x_0)} \left[ \mathcal{L}_{VLB}^{\ast} \right]$$，那么：

$$
\begin{align}
\mathcal{L}_{VLB}^{\ast} &= \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ log \frac{\Pi_{t=1}^T q(x_t \vert x_{t-1})}{p_{\theta}(x_T) \Pi_{t=1}^T p_{\theta}(x_{t-1} \vert x_t)} \right] = \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ -log p_{\theta}(x_T) + \sum_{t=2}^T log(\frac{q(x_t \vert x_{t-1})}{p_{\theta}(x_{t-1} \vert x_t)}) + log \frac{q(x_1 \vert x_0)}{p_{\theta}(x_0 \vert x_1)} \right] \\
&= \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ -log p_{\theta}(x_T) + \sum_{t=2}^T log(\frac{q(x_t \vert x_{t-1}, x_0)}{p_{\theta}(x_{t-1} \vert x_t)}) + log \frac{q(x_1 \vert x_0)}{p_{\theta}(x_0 \vert x_1)} \right] = \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ -log p_{\theta}(x_T) + \sum_{t=2}^T log(\frac{q(x_{t-1} \vert x_{t}, x_0)}{p_{\theta}(x_{t-1} \vert x_t)}\frac{q(x_t \vert x_0)}{q(x_{t-1} \vert x_0)}) + log \frac{q(x_1 \vert x_0)}{p_{\theta}(x_0 \vert x_1)} \right] \\
&= \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ -log p_{\theta}(x_T) + \sum_{t=2}^T log(\frac{q(x_{t-1} \vert x_{t}, x_0)}{p_{\theta}(x_{t-1} \vert x_t)}) + \sum_{t=2}^T log (\frac{q(x_t \vert x_0)}{q(x_{t-1} \vert x_0)}) + log \frac{q(x_1 \vert x_0)}{p_{\theta}(x_0 \vert x_1)} \right] = \mathop{\mathbb{E}}\limits_{x_{1:T} \sim q(x_{1:T} \vert x_0)} \left[ log \frac{q(x_T \vert x_0)}{p_{\theta}(x_T)} - log p_{\theta}(x_0 \vert x_1) + \sum_{t=2}^T log(\frac{q(x_{t-1} \vert x_{t}, x_0)}{p_{\theta}(x_{t-1} \vert x_t)}) \right] \\
&= -\mathop{\mathbb{E}}\limits_{x_1 \sim q(x_1 \vert x_0)} \left[ log p_{\theta}(x_0 \vert x_1) \right] + \mathop{\mathbb{E}}\limits_{x_T \sim q(x_T \vert x_0)} \left[ log \frac{q(x_T \vert x_0)}{p_{\theta}(x_T)} \right] + \sum_{t=2}^T \mathop{\mathbb{E}}\limits_{x_{t-1}, x_t \sim q(x_{t-1}, x_t \vert x_0)} \left[ log(\frac{q(x_{t-1} \vert x_{t}, x_0)}{p_{\theta}(x_{t-1} \vert x_t)}) \right]\\
&= -\mathop{\mathbb{E}}\limits_{x_1 \sim q(x_1 \vert x_0)} \left[ log p_{\theta}(x_0 \vert x_1) \right] + \textbf{D}_{\textbf{KL}}(q(x_T \vert x_0) \Vert p(x_T)) + \sum_{t=2}^T \mathop{\mathbb{E}}\limits_{x_{t-1}, x_t \sim q(x_t \vert x_0)q(x_{t-1} \vert x_t, x_0)} \left[ log(\frac{q(x_{t-1} \vert x_{t}, x_0)}{p_{\theta}(x_{t-1} \vert x_t)}) \right]、、
&= -\mathop{\mathbb{E}}\limits_{x_1 \sim q(x_1 \vert x_0)} \left[ log p_{\theta}(x_0 \vert x_1) \right] + \textbf{D}_{\textbf{KL}}(q(x_T \vert x_0) \Vert p(x_T)) + \sum_{t=2}^T \mathop{\mathbb{E}}\limits_{x_t \sim q(x_t \vert x_0)} \left[ \textbf{D}_{\textbf{KL}}(q(x_{t-1} \vert x_{t}, x_0) \Vert p_{\theta}(x_{t-1} \vert x_t)) \right]
\end{align}
$$

其中，上面式子右侧第一项叫做reconstruction term，第二项叫做prior matching term，第三项叫做denoising matching term。

denoising matching term里的每一项，由之前的结果可知：

$$q(x_{t-1} \vert x_t, x_0) = \mathcal{N}(x_{t-1}; \frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}} \bar{\epsilon_{t}}), \frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_{t}}} \beta_t)$$

为了让$$p_{\theta}(x_{t-1} \vert x_t)$$和$$q(x_{t-1} \vert x_t, x_0)$$之间的$$DL$$散度尽可能小，我们则也假设$$p_{\theta}(x_{t-1} \vert x_t)$$为高斯分布（与之前的假设不谋而合）。因为我们已经计算出来$$q(x_{t-1} \vert x_t, x_0)$$的方差为$$\frac{1-\bar{\alpha_{t-1}}}{1-\bar{\alpha_{t}}} \beta_t$$是个常数，所以$$p_{\theta}(x_{t-1} \vert x_t)$$的方差也是该值，但是$$q(x_{t-1} \vert x_t, x_0)$$的均值是$$\frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}} \bar{\epsilon_{t}})$$，其中$$\bar{\epsilon_{t}}$$是每次前向扩散过程从标准高斯分布中随机采样的值，这是未知的（因为有$$x_t, x_0, \bar{\epsilon_{t}}$$之间的关系，$$x_t$$在反向扩散过程中是已知的，$$x_0$$是我们希望要得到的，$$x_0$$的未知性和$$\bar{\epsilon_{t}}$$的未知性等价），所以没办法直接让$$p_{\theta}(x_{t-1} \vert x_t)$$的均值就等于它，从而这就成为了扩散模型需要利用神经网络进行学习的部分，记$$p_{\theta}(x_{t-1} \vert x_t)$$的均值为$$\mu_{\theta}$$。

因为$$p_{\theta}(x_{t-1} \vert x_t)$$和$$q(x_{t-1} \vert x_t, x_0)$$都是高斯分布，实际上它们之间的$$DL$$散度是可以closed-form计算出来的：

$$\textbf{D}_{\textbf{KL}}(q(x_{t-1} \vert x_t, x_0) \Vert p_{\theta}(x_{t-1} \vert x_t)) = \frac{1}{2\tilde{\beta_t}} \Vert \mu_{\theta} - \tilde{\mu}_t \Vert_2^2$$

对于reconstruction term，在原论文中用另一个单独的网络来优化，即认为$$p_{\theta^{'}}(x_0 \vert x_1) \sim \mathcal{N}(x_0; \tilde{\mu}_{\theta^{'}}(x_1, t=1), \Sigma_{\theta^{'}}(x_1, t=1))$$。而prior matching term不含可学习参数$$\theta$$。

所以说，最小化$$\mathcal{L}_{VLB}^{\ast}$$（也就是等价于最小化$$\mathcal{L}_{VLB}$$）的重点就在于最小化denoising matching term里的每一项$$p_{\theta}(x_{t-1} \vert x_t)$$和$$q(x_{t-1} \vert x_t, x_0)$$之间的$$DL$$散度，$$2 \leq t \leq T$$。而根据上面的推导过程可知，也就等价于最小化每个高斯分布$$p_{\theta}(x_{t-1} \vert x_t)$$的均值和高斯分布$$q(x_{t-1} \vert x_t, x_0)$$的均值。$$p_{\theta}(x_{t-1} \vert x_t)$$的均值由网络预测出，网络的输入是$$x_t$$和时间$$t$$，而$$q(x_{t-1} \vert x_t, x_0)$$的均值是已知的，有两种写法：

$$\tilde{\mu}(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_t}}x_t + \frac{\sqrt{\bar{\alpha_{t-1}}}\beta_t}{1-\bar{\alpha_t}}x_0$$

$$\tilde{\mu}(x_t, x_0) = \frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}} \bar{\epsilon_{t}})$$

从而我们的神经网络输出也可以有两种：（1）输入$$x_t, t$$，预测$$x_0$$；（2）输入$$x_t, t$$，预测$$\bar{\epsilon_t}$$。记$$f_{\theta}(x_t, t)$$为网络的输出。

对于第一种情况：

$$\mu_{\theta} = \frac{\sqrt{\alpha_t}(1-\bar{\alpha_{t-1}})}{1-\bar{\alpha_t}}x_t + \frac{\sqrt{\bar{\alpha_{t-1}}}\beta_t}{1-\bar{\alpha_t}}f_{\theta}(x_t, t)$$

从而：

$$\arg\min\limits_{\theta} \textbf{D}_{\textbf{KL}}(q(x_{t-1} \vert x_{t}, x_0) \Vert p_{\theta}(x_{t-1} \vert x_t)) = \arg\min\limits_{\theta} \frac{\bar{\alpha_{t-1}}\beta_t^2}{2\tilde{\beta_t} (1-\bar{\alpha_t})^2} \Vert f_{\theta}(x_t, t) - x_0 \Vert_2^2$$

对于第二种情况：

$$\mu_{\theta} = \frac{1}{\sqrt{\bar{\alpha_t}}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha_t}}} f_{\theta}(x_t, t))$$

从而：

$$\arg\min\limits_{\theta} \textbf{D}_{\textbf{KL}}(q(x_{t-1} \vert x_{t}, x_0) \Vert p_{\theta}(x_{t-1} \vert x_t)) = \arg\min\limits_{\theta} \frac{1}{2\tilde{\beta_t} \alpha_t} \frac{(1-\alpha_t)^2}{1-\bar{\alpha_t}} \Vert f_{\theta}(x_t, t) - \bar{\alpha_t} \Vert_2^2$$

diffusion模型一般采用第二种方式，因为对噪声进行建模会更加关注细节（噪声相对于原数据$$x_0$$要更小一些）。

在第二种方式下，回到$$\mathcal{L}_{VLB}$$中，则是：

$$
\begin{align}
\arg\min\limits_{\theta} \mathcal{L}_{VLB} &= \arg\min\limits_{\theta} \sum_{t=2}^T  \mathop{\mathbb{E}}\limits_{x_t, x_0 \sim q(x_t, x_0)} \left[ \textbf{D}_{\textbf{KL}}(q(x_{t-1} \vert x_{t}, x_0) \Vert p_{\theta}(x_{t-1} \vert x_t)) \right] = \arg\min\limits_{\theta} \sum_{t=2}^T \mathop{\mathbb{E}}\limits_{x_t, x_0 \sim q(x_t, x_0)} \left[ \frac{\bar{\alpha_{t-1}}\beta_t^2}{2\tilde{\beta_t} (1-\bar{\alpha_t})^2} \Vert f_{\theta}(x_t, t) - \bar{\alpha_t} \Vert_2^2 \right] \\
&= \arg\min\limits_{\theta} \mathop{\mathbb{E}}\limits_{x_0 \sim q(x_0), \bar{\alpha_t} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), t \sim \left[2, T \right]} \left[ \Vert f_{\theta}(x_t, t) - \bar{\alpha_t} \Vert_2^2 \right]
\end{align}
$$

所以说反向扩散过程（denoising）过程的本质，在于让模型学会以任意时刻$$t$$时加噪的数据以及时间$$t$$作为输入，都可以学到这个时刻的数据在“干净”数据上所加上的噪声（也就等价于学到“干净”的数据）。

DDPM模型的训练过程如下左图。而在训练完成之后，想要生成数据（采样）的过程如下右图

![3]({{ '/assets/images/diffusion_3.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}
*来自于[Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)*


### (4) 一些补充说明

**关于超参数$$\lbrace \beta_t \in (0,1) \rbrace_{t=1}^T$$的选择**

根据假设，$$q(x_t \vert x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I}), \  \text{where} \  t=1,2,\cdots,T$$。也就是说，$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}, \  \text{where} \  \epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$，但是为什么均值和方差的系数有这样的关系，并没有说明。下面给出一些intuition。

首先，假设$$x_t$$与$$x_{t-1}$$的关系是线性的（因为最简单），即：$$x_t = a_t x_{t-1} + b_t \epsilon_{t-1}, \  \text{where} \  \epsilon_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$。因为前向扩散过程是个加噪的过程，所以$$x_t$$应该是相对于$$x_{t-1}$$是衰减的，从而$$a_t, b_t \in (0,1)$$。

那么，我们将$$x_t$$的表达式不断地往后迭代，使用$$x_0$$来表示$$x_t$$：

$$x_t = a_t x_{t-1} + b_t \epsilon_{t-1} = a_t a_{t-1} x_{t-2} + a_t b_{t-1} \epsilon_{t-2} + b_t \epsilon_{t-1} = \cdots = (a_t a_{t-1} \cdots a_1) x_0 + (a_t \cdots a_2)b_1 \epsilon_{0} + \cdots + a_t b_{t-1} \epsilon_{t-2} + b_t \epsilon_{t-1}$$

从而，将第二项到最后一项全部综合起来，其也满足一个高斯分布，方差为$$\lbrace (a_t \cdots a_2b_1)^2 + \cdots + (a_t b_{t-1})^2 + (b_t)^2) \mathbf{I}$$。如果再考虑将第一项$$x_0$$的系数的平方和考虑进来，那么此时$$x_0$$系数的平方，与后面的方差的系数的平方和就是：$$(a_t \cdots a_1)^2 + (a_t \cdots a_2b_1)^2 + \cdots + (a_t b_{t-1})^2 + (b_t)^2) = a_t^2(a_{t-1}^2(\cdots(a_2^2(a_1^2+b_1^2)+b_2^2)+\cdots)+b_{t-1}^2)+b_t^2$$。如果令$$a_i^2 + b_i^2 =1$$对于所有的$$1\leq i \leq t$$成立，则该平方和就是1，而此时这些超参数的选择，就是前文所述的。


## 2. 基本原理（NCSN）

noise-conditioned score network，简称NCSN，在2019年有宋飏等人提出，是早于DDPM的，但是由于形式更为复杂，所以并没有火起来，直到后来人们才发现，其蕴含着比DDPM更深刻的原理（实际上DDPM可以看作NCSN的一种特例）。而之后，宋飏又在ICLR2021上发表了[一篇论文](https://arxiv.org/pdf/2011.13456)用来在SDE框架下解释DDPM和NCSN的统一性。

NCSN，或者更广泛一点说，scored-based generative model这一类模型，的优点有：
* 可以有媲美GAN的生成质量，但无需对抗训练，从而避免了类似于GAN那种训练困境
* 灵活的模型框架选择，无需像flow-based models那样，只能选择可以表示invertible transformation的框架

### (1). 从生成模型到score-based model

**1). 生成模型**

从一个未知的数据分布$$p(x)$$中，独立的采样出了一系列的样本$$\lbrace x_1, \cdots x_N \rbrace$$，这些既是数据集。而生成模型的目的就是，从这个含有有限样本的数据集出发，去拟合数据背后的分布$$p(x)$$，从而就可以获得无穷无尽的样本了。

为了实现从数据集估计其分布$$p(x)$$的目标，首先需要去对$$p(x)$$进行建模。而最常见的建模方式就是parametric模型，即首先选择一类概率分布并且认为其覆盖了$$p(x)$$，用带参数的分布$$p_{\theta}(x)$$来表示这一簇分布。而我们的目的，则是从数据集来估计这个参数$$\theta$$，从而得到$$p_{\theta^{\ast}}(x) \approx p(x)$$。

而很多时候，我们都是用energy-based的方法来表示$$p_{\theta}(x)$$，即

$$p_{\theta}(x) = e^{-f_{\theta}(x)} / Z_{\theta}$$

其中$$f_{\theta}(x) \in \mathbb{R}$$是任意的以$$\theta$$为参数的函数，而与$$x$$无关但与$$\theta$$有关的$$Z_{\theta}$$则是归一化常数，用来获得最终的分布$$p_{\theta}(x)$$。

最常见的对数据集的拟合方式就是使用最大似然估计：

$$\theta^{\ast} = \arg\max\limits_{\theta} \sum_{i=1}^N log p_{\theta}(x_i) = \arg\min\limits_{\theta} NlogZ_{\theta} + \sum_{i=1}^N f_{\theta}(x_i)$$

但是由上面的式子可以看到，objective function里含有$$Z_{\theta}$$，而这个值往往是难以估计的。

为了解决这个问题，宋飏等人提出了score-based model，其基本思想是，与其对$$p(x)$$建模，不如对$$p(x)$$对数据$$x$$的梯度进行建模来间接获取数据分布$$p(x)$$，因为$$Z_{\theta}$$与$$x$$无关，所以其关于$$x$$的梯度就是0，从而就可以避免对$$Z_{\theta}$$进行估计。


**2). score-based model**

score function，或者称为score，也就是NCSN那篇论文标题中的gradients of the data distribution，具体来说是概率密度函数对数的梯度：$$\nabla_x log p(x)$$。而用来对score进行建模/拟合的模型，就叫做score-based model，记这类模型为$$s_{\theta}(x)$$，其中$$\theta$$是模型可学习参数。

和直接建模数据分布函数$$p(x)$$不同，score-based model并不会受到归一化系数$$Z_{\theta}$$的影响。如果使用energy-based model来建模$$p(x)$$的化，那么$$s_{\theta}(x) = \nabla_x log p_{\theta}(x) = \nabla_x (-f_{\theta}(x) - log Z_{\theta}) = -\nabla_x f_{\theta}(x)$$。这样的话，我们就可以使用各种灵活的模型，而不需要考虑归一化参数是否容易求解这样一个巨大的制约了。

score的物理意义是：对于每个点$$x$$来说，该点的score就是数据的对数概率密度函数在该点$$x$$增长最快的方向

![4]({{ '/assets/images/diffusion_4.png' | relative_url }})
{: style="width: 1200px; max-width: 100%;"}
*来自于[DiffusionModel-NCSN原理与推导]([https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://zhuanlan.zhihu.com/p/670052757))*

上图可视化了一个2维分布概率密度函数对数的梯度（在每个点都有方向和大小，因为梯度是个向量）。如图所示，图里有两个中心，而这即代表了$$log p_{\theta}(x)$$取极大值的地方，即最能代表此数据先验分布的区域。对于生成模型而言，我们期望生成的数据，就应该位于这些数据先验分布值大的区域。所以说如果我们可以估计score，在有了score之后，就可以利用score来确定$$p_{\theta}(x)$$极大值的方位，从而就可以有更理想的生成结果。而在有score的情况下，从任意点出发，到达$$p_{\theta}(x)$$某个极大值的方法，就是朗之万采样（Langevin Sampling）。

**3). Langevin采样**

假设我们已经有了一个训练好的score-based model $$s_{\theta}(x)$$，可以对于任意输入的$$x$$，输出该点的score了，那么该如何采样，才能靠近$$p_{\theta}(x)$$的极大值点呢？

这实际上是一个Langevin dynamics问题（朗之万动力学），其提供了一种仅利用某个分布$$p(x)$$的score function，即$$\nabla_x log p(x)$$（对数概率密度函数的梯度），就可以对分布$$p(x)$$进行采样的MCMC方法。其具体操作如下：

* 首先从任意的某个先验分布，比如Uniform分布或者高斯分布中，随机采样一个初始样本$$x_0 \sim \pi(x)$$
* 利用如下公式逐渐将样本像$$p(x)$$的高密度区域靠近：

$$ x_{i+1} \leftarrow x_i + \epsilon \nabla_x log p(x_i) + \sqrt{2 \epsilon} z_i, \  \text{where} \  z_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), i=1,2,\cdots, K$$

* 当步长$$\epsilon \rightarrow 0, K \rightarrow \infty$$时，$$x_k \sim p(x)$$

在实际操作中，使用训练好的score function model $$s_{\theta}(x)$$替代上面的$$\nabla_x log p(x)$$，并且取足够小的$$\epsilon$$采样足够多次，这样就可以保证在某次之后的采样值，均服从$$p(x)$$分布。这个过程就叫做Langevin采样。

**4). score matching**

朗之万采样解决了我们在有了score之后，该如何采样样本，使其服从$$p(x)$$分布的问题，而最重要的是如何获取score呢？score-based model的方法是训练一个score-based model来逼近score。训练方法如下所述。

首先写出损失函数（目标函数）。score-based model和似然函数模型类似，也是将最小化模型和数据分布之间的Fisher divergence作为训练的目标：

$$\mathop{\mathbb{E}}_{p(x)} \left[ \lVert \nabda_x log p(x) - s_{\theta}(x) \rVert_2^2 \right] = \int p(x) \lVert \nabda_x log p(x) - s_{\theta}(x) \rVert_2^2 dx$$

但$$p(x)$$是未知的，所以上述式子无法计算，所以实际上，是利用经验分布$$p_{data}(x)$$（即从数据中获得的分布）来代替真实分布$$p(x)$$来计算的，从而我们的目标函数如下：

$$\mathop{\mathbb{E}}_{p_{data}(x)} \left[ \lVert \nabda_x log p_{data}(x) - s_{\theta}(x) \rVert_2^2 \right] = \int p_{data}(x) \lVert \nabda_x log p_{data}(x) - s_{\theta}(x) \rVert_2^2 dx$$

经验分布和真实分布的差别可以看[这篇博客](https://blog.csdn.net/qq_44638724/article/details/120242712)

而基于上述目标函数，使得模型的score-function与根据数据得到的经验分布的score相matching的算法，就叫做score-matching算法。

首先，我们简化一下上述目标函数：

$$
\begin{align}
& \mathop{\mathbb{E}}_{p_{data}(x)} \left[ \lVert \nabda_x log p_{data}(x) - s_{\theta}(x) \rVert_2^2 \right] \propto \frac{1}{2} \mathop{\mathbb{E}}_{p_{data}(x)} \left[ \lVert s_{\theta}(x) - \frac{\partial log p_{data}(x)}{\partial x} \rVert_2^2 \right] \\
&= \frac{1}{2} \int p_{data}(x) \left[ \Vert s_{theta}(x) \rVert_2^2 + \lVert \frac{\partial log p_{data}(x)}{\partial x} \rVert_2^2 - 2(\frac{\partial log p_{data}(x)}{\partial x})^Ts_{\theta}(x) \right] dx
\end{align}
$$

而

$$
\begin{align}
& \int p_{data}(x) \left[- 2(\frac{\partial log p_{data}(x)}{\partial x})^Ts_{\theta}(x) \right] dx = -2 \int p_{data}(x) (\sum_{i=1}^N \frac{\partial log p_{data}(x)}{\partial x_i})s_{\theta}(x)_i) dx \\
&= -2 \sum_{i=1}^N \int p_{data}(x) \frac{1}{p_{data}(x)} \frac{\partial p_{data}(x)}{\partial x_i} s_{\theta}(x)_i dx = -2 \sum_{i=1}^N \int \frac{\partial p_{data}(x)}{\partial x_i} s_{\theta}(x)_i dx \\
&= -2 \sum_{i=1}^N \int (\frac{\partial(p_{data}(x) s_{\theta}(x)_i)}{\partial x_i} - p_{data}(x) \frac{\partial s_{\theta}(x)_i}{\partial x_i}) dx = -2 \sum_{i=1}^N (p_{data}(x) s_{\theta}(x)_i \vert_{-\infty}^{\infty} - \int p_{data}(x) \frac{\partial s_{\theta}(x)_i}{\partial x_i} dx) \\
&= 2 \sum_{i=1}^N \int p_{data}(x) \frac{\partial s_{\theta}(x)_i}{\partial x_i} dx = 2 \int \sum_{i=1}^N p_{data}(x) \frac{\partial s_{\theta}(x)_i}{\partial x_i} dx = 2 \int p_{data}(x) tr(\frac{s_{\theta}(x)}{\partial x})dx
\end{align}
$$

其中$$N$$是输入$$x$$的维度。

回到之前的目标函数，可以发现，$$\lVert \frac{\partial log p_{data}(x)}{\partial x} \rVert_2^2$$与$$\theta$$无关，从而，之前的目标函数可以简化为：

$$\mathop{\mathbb{E}}_{p_{data}(x)} \left[ \frac{1}{2} \Vert s_{theta}(x) \rVert_2^2 + tr(\frac{s_{\theta}(x)}{\partial x}) \right] dx$$

目标函数得到了简化，但是如果表示$$s_{\theta}(x)$$的网络很深，$$x$$的维度很大的时候，计算$$tr(\frac{s_{\theta}(x)}{\partial x})$$仍然非常的繁重，在实际代码里部署起来很困难，从而在NCSN那篇论文里，又提出了两种更进一步的改进方法

**改进一：sliced score matching（由宋飏于2019年提出）**

在计算目标函数的时候，我们需要计算矩阵$$\frac{s_{\theta}(x)}{\partial x}$$的迹，而对于矩阵迹的估计，恰好有一种技巧：Hutchinson Trace estimation。

其具体做法是，对于一个随机向量$$v \in \mathbb{R}^n$$，如果其协方差矩阵为$$I$$，均值为$$\mathbf{0}$$，那么对于任意矩阵$$A \in \mathbb{R}^{n \times n}$$，$$tr(A) = tr(A \mathbb{E}(vv^T)) = \mathbb{E}(tr(A v v^T)) = \mathbb{E}(tr(v^T A v)) = \mathbb{E}(v^T A v)$$，从而将求矩阵$$A$$的迹，变成了求标量$$v^TAv$$对$$v$$的期望。

从而将上述技巧用到上述目标函数里，$$tr(\frac{s_{\theta}(x)}{\partial x}) = \mathbb{E}(v^T \frac{s_{\theta}(x)}{\partial x} v) = \mathbb{E}(v^T \frac{v^T s_{\theta}(x)}{\partial x})$$，目标函数则变为：

$$\mathop{\mathbb{E}}_{p(v), p_{data}(x)} \left[ \frac{1}{2} \Vert s_{theta}(x) \rVert_2^2 + v^T \frac{v^T s_{\theta}(x)}{\partial x} \right]$$

而计算$$\frac{v^T s_{\theta}(x)}{\partial x}$$只需要计算$$N$$次（相较于之前的$$N^2$$次，减少了很多），但引入了一个新的期望需要进行采样估计，如果$$N$$很大的时候，这样的做法是有效的。

**改进二：denoising score matching（NCSN那篇论文里的方法）**

这个方法也是为了避免计算$$tr(\frac{s_{\theta}(x)}{\partial x})$$，但它直接回到了最初的目标函数$$\mathop{\mathbb{E}}_{p_{data}(x)} \left[ \lVert \nabda_x log p_{data}(x) - s_{\theta}(x) \rVert_2^2 \right]$$，对于未知的$$p_{data}$$，其如果仅出现在求期望的概率分布上，并不出现在被求期望的值里面的时候，还是好办的，因为其就是经验概率分布，所以就是将所有的真实数据对应的被求期望的值加起来除以总数据数就行了（这也是为什么简化了的目标函数$$\mathop{\mathbb{E}}_{p_{data}(x)} \left[ \frac{1}{2} \Vert s_{theta}(x) \rVert_2^2 + tr(\frac{s_{\theta}(x)}{\partial x}) \right] dx$$可以计算的原因，这个目标函数的问题只是在于它太难算了）。但是如果$$p_{data}(x)$$同时也出现在了被求期望的值的内部，就不能按照上述方法算了，而如果我们回到了最初的目标函数，那么该目标函数的被求期望的值里就含有$$p_{data}(x)$$，所以需要想另一种办法解决这个问题，而denoising score matching的办法就是：既然$$p_{data}(x)$$未知，就自行定义一个已知的数据分布$$q_{\sigma}$$（比如高斯分布），而且假设这个分布是在$$p_{data}$$上加噪声得来的。

具体来说，记原数据为$$x$$，加噪之后的数据为$$\tilde{x}$$，我们定义$$q(\tilde{x} \vert x) = \mathcal{N}(\tilde{x}; x, \sigma^2 \textbf{I})$$，而且$$\sigma$$是已知的固定参数。从而$$q_{\sigma}(\tilde{x}) = \int q_{\sigma}(\tilde{x} \vert x) p_{data}(x) dx$$。我们希望用$$q_{\sigma}(\tilde{x})$$的score来近似$$p_{data}(x)$$的score（在$$\sigma$$很小的时候，它们是很相近的）。

那么对于这个新的数据$$\tilde{x}$$来说，其score-matching的目标函数就是:

$$\mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x})} \left[ \lVert \nabda_{\tilde{x}} log q_{\sigma}(\tilde{x}) - s_{\theta}(\tilde{x}) \rVert_2^2 \right]$$

这个式子可以显式的计算对于$$\tilde{x}$$score-matching算法的目标函数的值（因为$$q_{\sigma}(\tilde{x})$$显式的给定了），所以它叫做explicit score matching（ESM）。

$$ESM = \mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x})} \left[ \lVert s_{\theta}(\tilde{x}) \rVert_2^2 \right] - 2 \mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x})} \left[ \langle s_{\theta}(\tilde{x}), \nabda_{\tilde{x}} log q_{\sigma}(\tilde{x}) \rangle \right] + c_1, \  \text{where} \  c_1 \  \text{is} \  \text{irrelavant} \   \text{w.r.t.} \  \theta$$

再定义一个denoising score matching（DSM）：

$$
\begin{align}
DSM &= \mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x} \vert x) p_{data}(x)} \left[ \lVert \nabda_{\tilde{x}} log q_{\sigma}(\tilde{x} \vert x) - s_{\theta}(\tilde{x}) \rVert_2^2 \right] = \mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x}, x)} \left[ \lVert s_{\theta}(\tilde{x}) \rVert_2^2 \right] - 2 \mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x}, x)} \left[ \langle s_{\theta}(\tilde{x}), \nabda_{\tilde{x}} log q_{\sigma}(\tilde{x} \vert x) \rangle \right] + c_2, \  \text{where} \  c_2 \  \text{is} \  \text{irrelavant} \   \text{w.r.t.} \  \theta
\end{align}
$$

而

$$\mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x})} \left[ \lVert s_{\theta}(\tilde{x}) \rVert_2^2 \right] = \int_{\tilde{x}} q_{\sigma}(\tilde{x}) \lVert s_{\theta}(\tilde{x}) \rVert_2^2 d \tilde{x} = \int_{\tilde{x}} \int_{x} q_{\sigma}(\tilde{x} \vert x) p_{data}(x) \lVert s_{\theta}(\tilde{x}) \rVert_2^2 d \tilde{x} dx =  \mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x}, x)} \left[ \lVert s_{\theta}(\tilde{x}) \rVert_2^2 \right]$$

以及

$$
\begin{align}
&\mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x})} \left[ \langle s_{\theta}(\tilde{x}), \nabda_{\tilde{x}} log q_{\sigma}(\tilde{x}) \rangle \right] = \mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x})} \left[ \langle s_{\theta}(\tilde{x}), \frac{\partial log q_{\sigma}(\tilde{x})}{\partial \tilde{x}} \rangle \right] = \int_{\tilde{x}} q_{\sigma}(\tilde{x}) \langle s_{\theta}(\tilde{x}), \frac{\partial log q_{\sigma}(\tilde{x})}{\partial \tilde{x}} \rangle d \tilde{x}\\
&= \int_{\tilde{x}} q_{\sigma}(\tilde{x}) \langle s_{\theta}(\tilde{x}), \frac{1}{q_{\sigma}(\partial \tilde{x})}\frac{\partial q_{\sigma}(\partial \tilde{x})}{\tilde{x}} \rangle d \tilde{x} = \int_{\tilde{x}} \langle s_{\theta}(\tilde{x}), \frac{\partial q_{\sigma}(\partial \tilde{x})}{\partial \tilde{x}} \rangle d \tilde{x} = \int_{\tilde{x}} \langle s_{\theta}(\tilde{x}), \frac{\partial}{\partial \tilde{x}} \int_x q_{\sigma}(\tilde{x} \vert x) p_{data}(x) dx \rangle d \tilde{x} = \int_{\tilde{x}} \langle s_{\theta}(\tilde{x}), \int_x \frac{\partial q_{\sigma}(\tilde{x} \vert x)}{\partial \tilde{x}} p_{data}(x) dx \rangle d \tilde{x}\\
&= \int_{\tilde{x}} \langle s_{\theta}(\tilde{x}), \int_x \frac{\partial log q_{\sigma}(\tilde{x} \vert x)}{\partial \tilde{x}} q_{\sigma}(\tilde{x} \vert x) p_{data}(x) dx \rangle d \tilde{x} = \mathop{\mathbb{E}}_{q_{\sigma}(\tilde{x}, x)} \left[ \langle s_{\theta}(\tilde{x}), \nabda_{\tilde{x}} log q_{\sigma}(\tilde{x} \vert x) \rangle \right]
\end{align}
$$

从而惊讶地发现，ESM和DSM只相差了一个与$$\theta$$无关的常数。从而现在可以用DSM来替代ESM作为优化基于$$\tilde{x}$$的score-matching的目标函数了，也就是说，之前我们引入$$q_{\sigma}(\tilde{x})$$的score来近似$$p_{data}(x)$$的score，现在我们可以用$$q_{\sigma}(\tilde{x} \vert x}$$的score来近似$$p_{data}(x)$$的score了。








