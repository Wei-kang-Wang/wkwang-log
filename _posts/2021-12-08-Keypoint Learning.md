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
### [Unsupervised Learning of Visual 3D Keypoints for Control]()
*Boyuan Chen, Pieter Abbeel, Deepak Pathak*

*ICML 2021*

This paper proposed a method to directly learn 3D keypoint coordinates with only 2D images of the same scene from different view points as input in an unsupervised learning manner. The camera parameters are known. 

The whole process is as below:
Important hyperparameter: Keypoint number $$K$$.

**Step1** For each input 2D image $$I_n$$, use an encoder $$\Phi_n$$ to learn $$k$$ heatmaps $$C_n^k \in R^{S\times S}$$ and depth maps $$D_n^k \in R^{S\times S}$$. For each $$C_n^k$$, they used softmax to turn it into a probabilistic heatmap $$H_n^k$$ to represent the probability that keypoint $$k$$ will be at each position. Each depth map $$D_n^k$$ is a dense prediction of distance from camera plane for 3D keypoint k being at each position, i.e., each element in $$D_n^k$$ is the distance of keypoint $$k$$ to the camera plane.
After getting $$H_n^k$$ and $$D_n^k$$, they used the expectation across the whole heatmap to calculate the coordinate of keypoint $$k$$ in camera $$n$$ plane, $$\left[u_n^k, v_n^k, d_n^k\right]$$ as:

$$u_n^k = \frac{1}{S}\Sigma_{u,v}u H_n^k(u,v)$$, $$v_n^k = \frac{1}{S}\Sigma_{u,v}v H_n^k(u,v)$$, $$d_n^k = \Sigma_{u=1}^{S}\Sigma_{v=1}^{S}D_n^k(u,v) H_n^k(u,v)$$

Note that for each camera, we know the projection function to project the world coordinates into camera coordinates, denote it as $$\Omega_n$$, and the opposite projection as $$\Omega_n^{-1}$$. 

**Step2** Each camera will give a prediction of keypoint $$k$$ coordinates, the authors used some weighted techniques to get the final keypoint coodinates from these $$n$$ predictions, i.e., $$x^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i x_n^i$$, $$y^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i y_n^i$$, $$z^i = \frac{1}{n}\Sigma_{i=1}^n A_n^i z_n^i$$, where $$\left[x^i, y^i, z^i\right]$$ is the final coodinate of keypoint $$i$$ in the world coordination, and $$\left[x_n^i, y_n^i, z_n^i\right]$$ is calculated from $$\Omega_n^{-1}(\[u_n^i, v_n^i, d_n^i])$$.
The calculation of weights $$A_n^k$$ is as:

$$A_n^k = \frac{exp{\frac{1}{S^2}\Sigma_{p=1}\Sigma_{q=1}C_n^k(p,q)}}{\Sigma_{i=1}^K exp{\frac{1}{S^2}\Sigma_{p=1}\Sigma_{q=1}C_n^i(p,q)}}$$

**Step3** This step is a reconstruction step. The authors believed that if the decoder can reconstruct the input image from the keypoints projected from the learned 3D keypoints $$\left[x^i, y^i, z^i\right]$$, the model learns the geometry of the scene. The input of the decoder is the scaled stacked Gaussian distribution matrix with $$u_n^i, v_n^i$$ as the center and $$I_2/d_n^i$$ as the coviriance matrix. 

![Model Structure]({{ '/assets/images/Unsupervised_3D.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Overview of our Keypoint3D algorithm. (a) For each camera view, a fully convolutional neural network encodes the input image into K heat maps and depth maps. (b) We then treat these heat maps as probabilities to compute expectation of spatial $$u,v$$ coordinates in camera plane. These expected values and the saptial variances are used to resample final $$u,v$$ keypoint coordinates which adds noise that prevents the decoder from cheating to hide the input information in the relative locations $$u,v$$ keypoints. We also take expectation of depth coordinate, $$d$$, using the same probability distribution. These $$\[u; v; d]$$ coordinates are then unprojected into the world coordinate. (c) We take attention-weighted average of keypoint estimations from different camera views to get a single prediction in the world coordinate. (d) For decoding, we project predicted keypoints in world coordinate to [u; v; d] in each camera plane. (e) Each keypoint coordinate is mapped to a gaussian map, where a 2D gussian is created with mean at $$\[u, v]$$ and std inversely proportional to $$d$$. For each camera, gaussian maps are stacked together and passed into decoder to reconstruct observed pixels from the camera. (f) Together with reconstruction, we also jointly train a task MLP policy on top of predicted world coordinates via reinforcement learning.*


---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

