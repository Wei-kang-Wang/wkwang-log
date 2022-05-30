---
layout: post
comments: false
title: "Machine Learning知识点和炼丹技巧"
date: 2021-12-18 01:09:00

---

> This post of various machine learning knowledge.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

## 1. Batch Normalization

### Feature Scaling (or Feature Normalization)

先从Feature scaling，或者叫做Feature normalization说起。

假设$x_1$和$x_2$的数据差距很大：

$x_1$的范围是个位数，而$x_2$的范围是1e+3左右，假设$x_1$所乘的weight是$w_1$，$x_2$所乘的weight是$w_2$，那么因为$x_2$会比$x_1$大很多，所以$w_2$和$w_1$同等变化，$w_2$的影响会更大。

将$w_1$和$w_2$对于loss的影响作图，如fig 1所示。在$w_1$方向上的梯度，在$w_2$方向上的梯度较大。这样会让training变得不容易，因为要在不同的方向上给不同的learning rate。

所以说我们通过feature scaling的方式把不同的feature做normalization，从而使得loss关于不同参数的图看起来像正圆形（2维），从而就可以设置一个training rate了。如fig 1右侧。

![feature scaling]({{ '/assets/images/batch_normalization_1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Feature Scaling.*

**Feature Scaling的算法**

对于第i个维度的每一个data，都减去该维度的mean，再除以该维度的standard deviation。如fig 2所示。

![feature scaling algorithm]({{ '/assets/images/batch_normalization_2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Feature Scaling Algorithm.*

### Batch Normalization

而在deep learning中，我们经常会对hidden layer输出的值做feature scaling：也就是input在做完feature scaling之后进入layer1，经过layer1运算后的output在进入layer2之前也做一次feature scaling，经过layer2运算后的output在进入layer3之前也做一次feature scaling，一直这样下去。如fig 3所示。不仅仅对input做scaling，还对neural network的中间层做scaling，粗糙的说，这就叫做batch normalization（实际上还有两个要学习的参数）。

![feature scaling nn]({{ '/assets/images/batch_normalization_3.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 3. Feature Scaling for neural network.*

按照上述的方式对于每个layer的输出都做一次feature scaling的方法对于neural network的训练是很有好处的。因为它使得internal covariante shift这个现象减弱了。

>internal covariate shift问题：
>将每个人想象成neural network的一层，想象三个人通过话筒传话，对于中间的人来说，如果左边的人让他把话筒降低，右边的人让他把话筒升高，他就会将左手话筒放低，右手话筒升高，这样的话，他所做的移动太多，两侧都要调整很多，反映到neural network上，就会导致训练的结果不好，如fig 3所示。实际上internal covariate shift的意思就是在深层网络训练的过程中，由于网络中参数变化而引起内部结点数据分布发生变化。

当一个neural network很深的时候，后面的layer会根据前面的layer输出的结果做出调整，但是前面的layer会根据更前面的layer的变动进行调整，这样每次训练，见到了新的数据，所有的layer都动了，这样很难训练到一个好的结果。

为了解决这个问题，过去传统的方式就是将learning rate设置的小一点，这样每次模型的weights不会进行很大的改变（即使因为internal covariate shift导致gradients很大，或者方向不正确）。但这样会导致模型的训练很慢。

而feature scaling则很好的减弱了这个问题，其对每个layer的output都做了标准化的normalization，从而每个layer的输入都有mean=0，covariance=1，这样一个固定的statistics。

normalization可以放在activation function之前，也可以放在其之后。将normalization放在activation function之前，对于某些activation functions，比如说tanh或者sigmoid，因为其会有saturation region，而我们最好避免input落在saturation region里，因为这样会使得训练不动。所以先做normalization可以确保其不在saturation region里。

![bn]({{ '/assets/images/batch_normalization_41.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 4. Batch Normalization.*

在batch normalization中，我们计算的mean和variance都是根据一个batch来算的，因此我们希望这个batch就代表了整个training size，所以说实践上batch size不能太小，太小的话效果不好。

而batch normalization和我们之前所说的feature scaling的区别就在于它仍然还有可训练的参数。因为有时候我们不希望经过normalization之后的mean都是0，variance都是1，所以会为它增加两个可学习的参数，将normalization之后的数据再进行一步线性操作。如fig 5所示。

![fbn]({{ '/assets/images/batch_normalization_5.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 5. Full batch normalization.*

上述是training过程中batch normalization的做法，而在testing的时候，testing的data是一个个进来的，所以没法计算mean和variance。方法是在training的时候，记录下来每一个batch对应的$\mu$和$\sigma^2$，然后利用某种加权和的方式（比如说exponential weighted average, $E_{t+1} = \alpha a + (1-\alpha)E_{t}$）计算出整个training集的$\mu$和$\sigma^2$，从而在test时候用。

如fig 6所示。

![fbnt]({{ '/assets/images/batch_normalization_6.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 6. Batch normalization at testing time.*

使用batch normalization带来的好处：
* 减少训练实际，使得训练很深的网络变得可能。因为internal covariate shift被减弱了，所以可以用大的learning rate了。而且因为exploding/vanishing gradient的现象被减轻了（特别是对于activation function有saturation part的情况，比如说tanh或者sigmoid）
* initialization的影响减弱了。

### Andrew Ng对于BatchNormalization的解释（https://www.youtube.com/watch?v=nUUqwaxLnWs）

**Intuition 1**: 对input做feature scaling可以帮助训练，而同样的操作也可以针对neural networks的中间层输出来做。

**Intuition 2**: 对于一个网络来说，如果输入的training data的mean和variance一直在变，这个网络很难训练好，或者说如果训练集和测试集的mean和variance差别很大，那么这个训练好的网络在测试集上效果也不会好。比如说，仅仅在黑猫图片上训练的检测猫的神经网络，对于彩色猫的测试集效果不好。那么对于一个神经网络来说，对于较深的层来说，它的输入就是较浅的层的输出，而因为较浅的层也有可学习的权重，所以这个输出一直在变化，也就是covariance shift，这样导致较深层的输入一直在变化，从而很难学习。而batch normalization的作用就是让深层的输入较少的收到浅层的输出的影响。

**Intuition 3**: batch normalization还有一些regularization的作用。因为每一层的mean和variance都是根据一个batch算出来的，而不是所有的数据，这引入了一些noise。




## 2. Optimization Algorithms

---
