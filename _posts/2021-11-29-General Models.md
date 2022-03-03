---
layout: post
comments: false
title: "General Models"
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of general model structures, including computer vision, natural language processing, generative models, etc.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## Computer Vision

### AlexNet: [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
*ALex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton* 

*NIPS 2012*

The model in this paper won the ILSVRC 2012 competition classification task.

Before AlexNet, the machine learning community mostly concentrated on classical models, including SVM, decision trees, etc. The neural networks had no advantages over these classical models but had less elegant mathematical principles compared to them. Thus for supervised learning tasks, such as image classification, the neural network is not a good candidate. And many efforts has been put on the unsupervised learning tasks for neural networks, because after all, neural networks have better performances over classical models in this area. But the AlexNet shows that even without unsupervised learning, this deep convolutional neural network can have much better result on the ImageNet Classification Task. This is a milestone for supervised learning, computer vision, and machine learning.

**But recently researches show that unsupervised learning still remains much more mysteries and there are many researches on it. Yan LeCun even thinks that it's the main future direction for the whole machine learning community in a talk in 2020**

The first worth noting point is the model does not requires fixed size of iuput image. It will down sample the images to a fixed resolution of $$256 \times 256$$. Given a rectangular image, the model first scale the image with the shorter side of length 256, and then crop out the central $$256 \times 256$$ patch from the resulting image. Also there are no pre-processing techniques, the network is trained on the centered raw RGB values of the pixels. This is a big inprovement then, since the existing works at that time always use some feature extractor to get useful features, such as SIFT, and then do the classification task. **This is so-called End-to-End**.

Another point is the authors use ReLU rather than tanh or sigmoid function as the non-linearity function of the model. They argue that ReLU is a non-saturating function, which is much better than those saturating functions since it can help the model to learn much faster. (An activation function is considered non-satured if $$lim_{x \rightarrow \infty} f(x) = \infty$$. A saturated activation function has a compact range such as \[-1,1] for tanh or \[0,1] for the sigmoid.) But from today's viewpoint, this is not the reason, or at least not the only reason that why ReLU are much more faster, but the biggest advantage of ReLU is that it's simple enough, thus it's the most prevalent non linear function now in neural networks.

![Model Structure]({{ '/assets/images/AlexNet-2.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Model Structure.*

**The model takes an RGB image as input, which is a human aware data, to vector, which is a computer aware data. The whole process can be understood as a knowledge compression. This is the key of deep neural networks.**

One big result is that the features from the last hidden layer really learn the semantic information. The authors show this result by first taking a random image, and then find six other images whose last layer features has the smallest Euclidean distance to the picked one's. This result shows that the feature vectors from the last hidden layer of this model have good semantic representation of this supervised learning task. The feature vectors space have a good semantic explanation of the original task, i.e., semantic similar images have distance close feature vectors.

![Comparison based on features]({{ '/assets/images/AlexNet-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. Five ILSVRC-2010 test images in the first column. The remaining columns show the six training images that produce feature vectors in the last hidden layer with the smallest Euclidean distance from the feature vector for the test image.*

Some techniques used in these papers are prevalent in the following works, including dropout, weight decay, momentum, SGD algorithm, parameter initialization, learning rate. There are several methods of setting learning rate. In this paper, they first set a number, and then manually divide this number by 10 if the training error becomes flat. Other methods including every fixed number of epochs, dividing the learning rate by 10, or using some smooth function to define the learning rate, or firstly raising and then reducing the learning rate, etc. (*one epoch means one cycle of the whole dataset.*)


### VGGNet: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
*Karen Simonyan, Andrew Zisserman*

*ICLR 2015*


### GoogleNeet: [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf)
*Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich*

*CVPR 2015*


### ResNet: [Deep Residual Learning for Image Recognition](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
*Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun*

*CVPR 2016*

**AlexNet(2012) $$\rightarrow$$ ResNet(2016)**

The biggest contribution of this paper is to offer a model strucuture that makes the training of very deep neural networks possible. The model in this paper won the ILSVRC 2015 competition classification task.

For naive convolutional neural networks, using very deep architecture will not cause over-fitting easily, but cause under-fitting, i.e., the model is very hard to train. Because not only the testing error of deep models are higher, the training error is also higher. The phenomenon is shown in the below figures. 


![Hard to Train]({{ '/assets/images/ResNet-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 1. Training error (left) and test error (right) on CIFAR-10 with 20-layer and 56-layer “plain” networks. The deeper network has higher training error, and thus test error. Similar phenomena shows on ImageNet.*

![Main result]({{ '/assets/images/ResNet-2.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 2. Training on ImageNet. Thin curves denote training error, and bold curves denote validation error of the center crops. Left: plain networks of 18 and 34 layers. Right: ResNets of 18 and 34 layers. In this plot, the residual networks have no extra parameter compared to their plain counterparts.*

When the network becomes deep, the problem of gradients vanishing/exploring will be remarkable. Former methods to alleviate this problem include setting good initiation parameters, using batch normalization. These methods makes training deep networks become possible, but actually the performance becomes worse. In principle, this should not be the case. Because if we have a shallow network, and then we add several more layers to create its deep counterpart. The deep network should be at least good as the shalow one, because it can let the added layers be just identity mapping. But these kind of parameters are very hard for deep networks to learn, thus the above phenomenon exists.

In this paper, having the above ideas in mind, the authors create a model that explictly having structures to represent this **identity mapping**. In this paper, they use a shortcut directly add $$x$$ from the input to the output of two layers. If the groundtruth is $$H(x)$$, they actually want the two layers to learn the "residual", i.e., $$f(x)=H(x)-x$$, where $$H(x)$$ is the desired output and $$x$$ is the input, with respect to this two layer structure. The shorcut here resemble the identity mapping.
zero-padding or $$1 \times 1$$ convolution are used to solve the problem that $$f(x)$$ and $$x$$ has different width $$&$$ length and channels.

![Residual Block]({{ '/assets/images/ResNet-3.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 3. Residual learning: a building block.*

Some implementation details are different from the AlexNet, for example, the shorter side of input image is firstly randomly scaled to a number in \[256,480] and then a $$224 \times 224$$ patch is sampled from the original input. There are no fully-connected layers except for the last softmax classifier, thus drop out is also not implemented.

Another structure design in this paper is the bottleneck design. This design helps not increase parameter numbers too much while using very deep architectures (bigger than 50). Use the below figure as an example. The input has size $$width \times length \times 256$$. Firstly the input is compressed into $$width \times length \times 64$$ by using $$1 \times 1$$ convolution, then normal convolutional layers are deployed. In the end, $$1 \times 1$$ convolution is used again to raise the output channel to 256. This process will make the computation complexity of this structure be similar to the left one, but the model depth will be much deeper. Also, due to the existence of the shortcut connection, the information loss in this process only happens in the residual computation, and the information in the input is not influenced.

![Bottleneck]({{ '/assets/images/ResNet-4.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}
*Fig 4. Left: normal residual block. Right: Bottleneck residual block.*

Suppose $$g(x)$$ is a neural network. Adding more layers to this net will make the model function become $$f(g(x))$$. $$\frac{d f(g(x))}{d x} = \frac{d f(g(x))}{d g(x)} \frac{d g(x)}{d x}$$ is the gradients of the new model and $$\frac{d g(x)}{d x}$$ is the original one's. Gradients are always be quite small, thus the multplication will make the gradients of the new model be much smaller than the original one's, thus the training of deep model is very hard.

But if we use the ResNet structure in this paper, then our new deep counterpart of $$g(x)$$ becomes $$f(g(x))+g(x)$$, and the gradient becomes $$\frac{d f(g(x))}{d g(x)} \frac{d g(x)}{d x} + \frac{d g(x)}{d x}$$, which is comparable to the original one's.

The key idea is: **Always make the gradients be large enough, and then your model can be trained well!**


### MAE: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)

*Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, Ross Girshick*

*Arxiv 2021*

**Transformer $$\rightarrow$$ Bert(using transformer framework, but with self-supervised learning), Transformer $$\rightarrow$$ ViT(using transformer idea in CV), ViT $$\rightarrow$$ MAE (just like Bert' relation to Transformer)**

**Transformer $$\rightarrow$$ Bert $$\rightarrow$$ MAE $$\leftarrow$$ ViT $$\leftarrow$$ Transformer**

#### 1. Title

"Scalable" are usually used when your model is very big, and "efficient" are usually used when your model is quite quick. These are two pupolar words in a title. "Vision learners" is a more broad word rather than classifier or other specific models. The "auto" in "Autoencoders" means that the input $$x$$ and label $$y$$ comes from the same data. "Auto" models are a kind of general models. In NLP, most works are actually "auto", so they usually just neglect this word. But in CV, these kind of works are not very much, because the label for images seldom comes from images themselves, thus "Autoencoders" in this title is essential. It means that in this work, the labels for images come from images themselves, distinguished from other works.

The whole sentance, "a" are "b" "c", where "a" and "c" are nouns and "b" are adjectives, are a popular naming pattern recently. This is a very useful sentance, since it contains the conclusion in the title, and it's objective.

#### 2. Abstract

The idea is to mask random patches of an image and let the model to reconstruct the masked pixels. This idea is quite similar to Bert, since in Bert, they let the model to learn masked words according to other words. There are two core ideas of this model: firstly the whole model is a asymmetric encoder-decoder architecture, since the masked patches will not be encoded and the decoder will reconstruct the whole input image from non-masked representations and masked tokens. Secondly, masking high portion of the image (e.g., 75%) will yield a nontrivial sefl-supervised learning task. Because only masking few pixels, we can just using interpolation to archive pretty good result. These two ideas make training a large model efficiently becomes possible: because large portion will be masked and not encoded, and this problem is hard enough to implement large models to learn. Since this work is a counterpart to Bert, it actually used as some pretrained technic for transfer leaning.


#### 3. Figures

Have a look at figure1. Figure1 is usually at the upper right part of a CV paper and should be the most important figure of this paper to generally explain the idea of this paper. Figure1 describes the whole process of the MAE model. The input image will be cut into patches, and patches are randomly masked. Only non-masked patches are encoded by the encoder to get the representations of each non-masked patches. Then the representations are streched, and arranged according to the positions of there non-masked patches, and masked patches are also included, using only the position information. And finally the decoder will reconstruct the pixel information based on this streched representations. Note that in the figure, the box of encoder is bigger than the one of decoder, indicating that the encoder is more complex and is the key structure in this model.

![MAE]({{ '/assets/images/MAE-1.PNG' | relative_url }})
{: style="width: 800px; max-width: 100%;" class="center"}
*Fig 1. MAE architecture.*

Now turn to figure2 and figure3. Figure2 shows the reconstruction ability of MAE model on the ImageNet validation set, while figure3 shows the same results on COCO dataset. 

Figure4 shows that for the same image, masking different portion of the original image will result in different reconstructions.

These three figures (2, 3, 4) have very good results on reconstruction based on very large portion of original images been masked. I think one reason could be that the model have learned the common information of one category, and once it encountered an image of this category with large portion occluded, it can use the information of other images of the same category to assist the reconstruction of this image.

#### 4. Discussion and Conclusion

The authors think this model is simple and can be scaled well. Being simple is because it's based on Vision Transformer, and being scaled-well is because it does not encode masked patches and saves lots of calculation for encoding.

The authors also argue that supervised learning still dominates the CV community, different than the NLP groups. And this model has an unsupervised setting, and could be an important future direction for CV.

They also mention that we need to note the differences between CV and NLP tasks. Images and languages are signals of different nature and this intrinsic difference must be addressed carefully. A word is itself a semantic part, and different words form sentences. But for images, a pixel or a patch is not a semantic part of an image. Even having these differences, MAE (based on transformer architecture) could learn some good hidden representations. This part needs future works.


#### 5. Introduction

1. Tell a story which is that deep learning has made great progress in CV, but still requires plenty of labeled data. And in NLP, self-supervised techniques are very popular, e.g., BERT, GPT. etc. Masked autoencoders are also not a novel thing in CV, for example, denoising autoencoders (decades before) involve adding noise to the image and then denoising the noise, and by doing this can let the model learn the useful representations of the images. Some works recently have used BERT structure into CV, but still not having very good results. So the question is: **what makes masked autoencoding different between vision and languages?** The authors give three answers to this question. 1). The architecture is different. For NLP, popular architecture is Transformer. And once you mask a word, you can always know the position information of the masked word easily. But for CV, the popular architecture is CNN. CNN uses convolution calculation sliding over the whole feature maps. Thus you mask one area of the image, and you can hardly track the positions of the masked area after several layers of convolution. But this problem has been issued by the introduction of Vision Transformer (ViT). But actually, why transformer needs position information is that attention mechanism in transformer does not have position information. But CNN naturaly contains position information because the convolution window slides over the feature maps, and record the results of each area into specific location of the resulting next layer feature maps. 2). Information density is different. For language, a word is a semantic identity, and removing words, even few, can make one sentences vague. But there are much information redundacy within an image. Only removing a patch of an image, you can reconstruct this part by interpolating using neighborhood pixels. Because of this reason, masked autoencoding is not very useful in CV, because it does not learn useful semantic representations. In this paper, the authors just simply mask large portion of the images to make the task challenging and thus the leaRned representations in this way is information rich enough, not just interpolating but learning global information instead. 3). The autoencoder's decoder. The decoder for language models usually output a word, which is a very high-level semantic information. So the decoder in NLP tasks are quite simple, just MLP could be enough. But for CV tasks, the output of the decoder of the autoencoder of the CV task are normly pixels, i.e., the reconstruction of the original image, which is a very low-level semantic representation. Thus if the CV task is difficult, for example, the semantic segmentation, the decoder should be more complex rather than the NLP models. Note that for simple CV tasks, such as classification, decoders are also very simple, such as in ResNet, the decoder is just several layers of MLP.

2. Based on the ideas above, they propose MAE. MAE uses a asymmetric architecture of encoder and decoder to reconstruct images with large portion been masked. Being asymmetric means that the input information of the encoder and the decoder is different, in their setting, encoder only encodes non-masked patches, while decoder also has the position information of masked patches. This setting could make the computation of encoder be much less, and more layers can be added and model can be easily scaled. 

3. They finally show that MAE trained on only ImageNet-1K dataset can have very good performance as pre-trained model for downstream tasks such as object detection, instance segmentation, semantic segmentation, etc. So MAE in CV serves as some similar role as the BERT in NLP.


#### 6. Related works

1. Masked language models, including BERT, GPT, etc.

2. Autoencoding models in CV community. 

3. Masked image encoding in CV community. iGPT, BEiT are very closed papers, but they do not explan them clearly. Recommend that for very closely-related papers, you need to explain them and show the differences between your paper and their's.

4. Self-supervised learning in CV community.


#### 7. Approach




#### The writing style of this paper

1. The introduction part is a little long, partly because this paper uses lots of figures. Fruitful figures in CV paper is a good thing. The writing style of the introduction is not just an extension of the abstract (like GAN paper), but raising the topic into a more high level, and raise questions. This paper aims to solve this kind of question. It's a very good writing style. This can explain the necessity of this paper and give the basic idea of how the authors understand this problem. This could make the paper insightful, rather than just a technic report explaining the model details (like AlexNet paper).


## Natural Language Processing

### Transformer: [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
*Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin*

*NIPS 2017*



## Generative Models

### GAN: [Generative Adversarial Nets](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)
*Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio*

*NIPS 2014*

#### 1. Abstract
1. The abstract of this paper is very standard and unique. This abstract generally introduces the main concept of this paper, i.e., GAN, rather than introducing the problem and existing methods. This abstract is kind of like the Wikipedia introduction of some concept. If you are very confident that your paper is very novel and proposes a very useful idea, model or concept that can be recorded in this area, you can use this kind of writing style of abstract. This is very clear to the community, but not very clear to the people that do not know any about this area.

#### 2. Introduction
2. This paper thinks that besides deep learning architectures, deep learning will learn the representations of data probability distributions over all kinds of data, including natural images, audio waveforms containing speech, symbols in natural language processing, etc. This idea is always held by Yoshua Bengio and his group.

3. Discriminative models in deep learning have already invovled into various kinds and solved many kinds of problems. But generative models, instead, still have lots of part remained mysterious. This due to the fact that in order to learn data distributions underlying generative models, you need many approximation methods to approximate the distribution in order to make calculation work. But this process will make the distribution unaccurate and even not working. So in this paper, they do not try to approximate the distributions, they use deep learning models to do this job.

4. Generative adversarial models will contain two parts: the generative model and the discriminator model. The generative model aims to generate data that the discriminator model can not distinguish apart from true training data, and the discriminator model aims to distinguish between them. The final goal is to let the generative model to generative plausible data that the distriminator model can not distinguish.

5. In this paper, they use MLP as models for the generative and discriminator models, and in this case, this generative adversarial model is called generative adversarial net (GAN). The input of the generative model is random noise (usually a Guassion noise), and they want the generative model to map this random noise distribution to any distribution we want to fitting, i.e., the training data distribution.


#### 3. Related work

1. There are two kinds of former works on deep generative models. The first ones concentrated on building a probability distribution function, and these models are trained on maximizing the log likelihood, such as the deep Boltzmann machine. The biggest difficulty is that the calculation of sampling this distribution is hard, especially when the dimension is high. The other ones are called generative machines. They do not explicity construct the distribution, and they learn a model to approximate this distribution. There are intrinsic differences between these two kinds of methods. The first ones acutally learn the distribution, though using some kind of approximation method to make the calculation feasible, but in the end, you actually get a distribution, you can calculate the mean, variance and all kind of properties of this distirbution. But the other ones do not construct the distribution, and only learn a model to approximate this distribution. So in the end, we do not know what this distribution looks like. 

2. Variational Autoencoder (VAE) actually has similar ideas to this paper. And using a distriminator model to assist the generative model is also not novel, such as Noise-contrastive Estimation (NCE). 


#### 4. Adversarial nets
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


#### 4. Theoretical Results

1. There is a global optimum for the generator, $$p_g = p_{data}$$. Firstly, we see a lemma firstly. **Lemma**: if the generator is fixed, then the optimal discriminator will be

$$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$

i.e., the error probability (the training criterion of discriminator) of the distriminator will be the smallest. **Theorem**: Setting $$ D_{G}^{\*}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$ as in Lemma in the equation $$\min_{G}\max_{D} V(D,G) = E_{x \sim p_{data}}\left[log D(x)\right] + E_{z \sim p_z(z)}\left[log(1-D(G(z)))\right]$$, we can get $$C(G) = E_{x \sim p_{data}}\left[log \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}\right] + E_{x \sim p_g}\left[log \frac{p_g(x)}{p_{data}(x) + p_g(x)} \right] $$. Then $$C(G) get its minimum when $$p_g = p_{data}$$.

2. The algorithm described above is able to train the discriminator and the generator, i.e., the training algorithm is convergent. 


#### 5. Experiments
The experiments in this paper is not good enough and quite simple.


#### 6. Conclusion

GAN is actually an unsupervised learning setting, but it leverages supervised learning framework by using the label of true or generated data for training. This idea insights the future self-supervised learning frameworks.


#### The conclusion of writing style of this paper

This paper proposes a very novel idea and model, thus it elaborates the details of design and ideas behind GAN very clearly. The authors are very ambitious and are confident that this work will be recorded in this area. So the whole writing style is kind of like Wikipedia, i.e., the very detailed description of the proposed model, with a little mention of existing works and comparison with other works. Also, experiments are few.

If you are confident that your work is novel and very important, you can use this kind of writing style. Otherwise, you need to describe clearly the existing works and your contribution to this problem.









---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

