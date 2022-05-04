---
layout: post
comments: false
title: "[论文]Transformers
date: 2021-11-29 01:09:00
tags: paper-reading
---

> This post is a summary of Transformer-related papers.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

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



---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*

