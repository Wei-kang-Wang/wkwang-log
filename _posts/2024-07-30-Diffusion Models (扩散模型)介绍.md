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

