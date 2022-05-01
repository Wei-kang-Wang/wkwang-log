---
layout: post
comments: false
title: "Machine Learning Techniques"
date: 2021-12-18 01:09:00

---

> This post of various machine learning knowledge.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## Batch Normalization

### Feature Scaling (or Feature Normalization)

Starting from feature scaling or feature normalization.

Firstly, we see an example. Suppose $$x_1$$ and $$x_2$$ have different value ranges, $$x_1 \sim O(1)$$ and $$x_2 \sim O(100)$$. $$\omega_1$$ and $$\omega_2$$ are weights corresponding to $$x_1$$ and $$x_2$$ respectively. The output are defined as $$f=\omega_1 \times x_1 + \omega_2 \times x_2$$ + b. Since $$x_2$$ are much bigger than $$x_1$$, so the change of $$\omega_2$$ influence the result much more than the change of $$\omega_1$$.

Draw the function of loss with respect to $$\omega_1$$ and $$\omega_2$$. In the left part of figure below, we see that in the direction of $$\omega_1$$, the gradient is small, while in the direction of $$\omega_2$$, the gradient is big.

![feature scaling]({{ '/assets/images/batch_normalization_1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Feature Scaling.*

Feature scaling are proposed to normalize different features in order to make the error surface be a "hypersphere", like the right of the figure above, i.e., in each direction of parameters corresponding to different intputs, the gradients change in some same degree. This will make the training much more easier, since we do not need to set different learning rates for different parameters.

**Feature Scaling Algorithm**:
* Inputs $$x_1, x_2, ... , x_m \in R^{n}$$
* For each dimension $$d \in \left\{1,2,...,m\right\}$$, calculate mean $$m_d = \frac{1}{m} \Sigma_{i=1}^m x_i^d$$ and variance $$\sigma_d^2 = \frac{1}{m-1} \Sigma_{i=1}^m (x_i^d-m_d)^2$$. Then $$x_i^d \leftarrow \frac{x_i^d - m_d}{\sigma_d}$$.



## Optimization Algorithms

---
