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

Firstly, we see an example. Suppose $$x_1$$ and $$x_2$$ have different value ranges, $$x_1 \sim O(1)$$ and $$x_2 \sim O(100)$$. $$\omega_1$$ and $$\omega_2$$ are weights corresponding to $$x_1$$ and $$x_2$$ respectively. Since $$x_2$$ are much bigger than $$x_1$$, so the change of $$\omega_2$$ influence the result much more than the change of $$\omega_1$$.

Draw the function of loss with respect to $$\omega_1$$ and $$\omega_2$$. In the left part of figure below, we see that in the direction of $$\omega_1$$, the gradient is small, while in the direction of $$\omega_2$$, the gradient is big.

![feature scaling]({{ '/assets/images/batch_normalization_1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Feature Scaling.*

Feature scaling are proposed to normalize different features in order to make the error surface be a "hypersphere", which will make the training much more easier.

## Optimization Algorithms

---
