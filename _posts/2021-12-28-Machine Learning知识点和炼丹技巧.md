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

## 1. Batch Normalization 批标准化

### 1.1 Feature Scaling (or Feature Normalization)

先从Feature scaling，或者叫做Feature normalization说起。

假设$$x_1$$和$$x_2$$的数据差距很大：

$$x_1$$的范围是个位数，而$x_2$的范围是1e+3左右，假设$$x_1$$所乘的weight是$$w_1$$，$$x_2$$所乘的weight是$$w_2$$，那么因为$$x_2$$会比$$x_1$$大很多，所以$$w_2$$和$$w_1$$同等变化，$w_2$的影响会更大。

将$$w_1$$和$$w_2$$对于loss的影响作图，如fig 1所示。在$$w_1$$方向上的梯度，在$$w_2$$方向上的梯度较大。这样会让training变得不容易，因为要在不同的方向上给不同的learning rate。

所以说我们通过feature scaling的方式把不同的feature做normalization，从而使得loss关于不同参数的图看起来像正圆形（2维），从而就可以设置一个training rate了。如fig 1右侧。

![feature scaling]({{ '/assets/images/batch_normalization_1.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 1. Feature Scaling.*

**Feature Scaling的算法**

对于第i个维度的每一个data，都减去该维度的mean，再除以该维度的standard deviation。如fig 2所示。

![feature scaling algorithm]({{ '/assets/images/batch_normalization_2.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 2. Feature Scaling Algorithm.*

### 1.2 Batch Normalization

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

上述是training过程中batch normalization的做法，而在testing的时候，testing的data是一个个进来的，所以没法计算mean和variance。方法是在training的时候，记录下来每一个batch对应的$$\mu$$和$$\sigma^2$$，然后利用某种加权和的方式（比如说exponential weighted average, $$E_{t+1} = \alpha a + (1-\alpha)E_{t}$$）计算出整个training集的$$\mu$$和$$\sigma^2$$，从而在test时候用。

如fig 6所示。

![fbnt]({{ '/assets/images/batch_normalization_6.png' | relative_url }})
{: style="width: 800px; max-width: 100%;"}
*Fig 6. Batch normalization at testing time.*

使用batch normalization带来的好处：
* 减少训练实际，使得训练很深的网络变得可能。因为internal covariate shift被减弱了，所以可以用大的learning rate了。而且因为exploding/vanishing gradient的现象被减轻了（特别是对于activation function有saturation part的情况，比如说tanh或者sigmoid）
* initialization的影响减弱了。

### 1.3 Andrew Ng对于BatchNormalization的解释（https://www.youtube.com/watch?v=nUUqwaxLnWs）

**Intuition 1**: 对input做feature scaling可以帮助训练，而同样的操作也可以针对neural networks的中间层输出来做。

**Intuition 2**: 对于一个网络来说，如果输入的training data的mean和variance一直在变，这个网络很难训练好，或者说如果训练集和测试集的mean和variance差别很大，那么这个训练好的网络在测试集上效果也不会好。比如说，仅仅在黑猫图片上训练的检测猫的神经网络，对于彩色猫的测试集效果不好。那么对于一个神经网络来说，对于较深的层来说，它的输入就是较浅的层的输出，而因为较浅的层也有可学习的权重，所以这个输出一直在变化，也就是covariance shift，这样导致较深层的输入一直在变化，从而很难学习。而batch normalization的作用就是让深层的输入较少的收到浅层的输出的影响。

**Intuition 3**: batch normalization还有一些regularization的作用。因为每一层的mean和variance都是根据一个batch算出来的，而不是所有的数据，这引入了一些noise。




## 2. Singular Value Decomposition (SVD) 奇异值分解

### 2.1 Eigendecomposition

SVD的原始想法起源于eigendecomposition，所以我们先来介绍eigendecomposition。

对于一个方矩阵$$A \in C^{n \times n}$$，如果存在向量$$v \in C^{n}$$和标量$$\lambda \in C$$使得$$Av = \lambda v$$，那么就称$$\lambda$$为$$A$$的eigenvalue，而$$v$$为这个eigenvalue所对应的一个eigenvector。可以很直接地看出，对eigenvector进行scaling并不影响结果。

求解eigenvalue的方式是，因为$$Av = \lambda v$$，所以$$(A - \lambda I)v = 0$$。而如果$$det(A-\lambda I) \neq 0$$，那么$$A-\lambda I$$就是invertible的，那么就会得到$$v=0$$，与eigenvector不能为0矛盾。从而$$det(A-\lambda I)=0$$。这个式子是关于$$\lambda$$的$$n$$阶多项式，从而在复数域内有$$n$$个解。而且可以看出来，所有的eigenvalues的乘积等于$$det(A)$$（重复的根也要重复乘上）。

一个方矩阵$$A$$的所有eigenvalues构成的集合叫做这个矩阵的spectrum，记为$$\sigma(A)$$。

下面我们只考虑实矩阵$$A \in R^{n \times n}$$。其有以下的性质：

* 如果$$\lambda$$是$$A$$的eigenvalue，那么$$\bar{\lambda}$$也是$$A$$的eigenvalue。
* 不同的eigenvalues对应的eigenvectors是线性无关的。

对于对称实矩阵$$S \in R^{n \times n}$$，$$S^{T} = S$$，我们有很多特殊的很好的性质。

如果一个对称实矩阵$$S$$满足对于任意$$x \in R^{n}$$都有$$x^{T}Sx \geq 0$$，那么称$$S$$是positive semi-definite的（如果是严格大于，那就是positive definite）。

* 所有的eigenvalues都是实数。
* 对应于不同的eigenvalues的eigenvectors是orthogonal的（不仅限于linear independent了）。
* 如果$$S$$是positive semi-definite（positive definite）的，那么其所有的eigenvalues都是非负（正）的。
* 如果$$\lambda_1$$和$$\lambda_n$$分别是最大的和最小的eigenvalues，那么$$\lambda_1 = max_{||x||=1} <x, Sx>$$，以及$$\lambda_1 = min_{||x||=1} <x, Sx>$$。

对于eigenvalues的计算公式，我们有$$det(S-\lambda I) = 0$$，也就是关于$$\lambda$$的方程。假设这个关于$$\lambda$$的方程只有$$k$$个不同的根，即$$det(S-\lambda I) = (\lambda - \lambda_1)^{n_1} (\lambda - \lambda_2)^{n_2}...(\lambda - \lambda_k)^{n_k}=0$$。称$$n_i$$为eigenvalue $$\lambda_i$$的代数重数。对于特定的一个eigenvalue $$\lambda_i$$，我们有$$Sv = \lambda_i v$$，即$$(S-\lambda_i I) v = 0$$，从而其对应的eigenvectors都在矩阵$$S-\lambda_i I$$的null space里，这个null space对应的维数$$m_i$$称为eigenvalue $$\lambda_i$$的几何重数。我们有代数重数不小于几何重数这样的限制。

对称实矩阵的好处就是，对于任何eigenvalue，都有其代数重数等于几何重数，也就是说假设我们将对称实矩阵$$S$$的$$n$$个eigenvalues排成一个对角矩阵$$\Sigma$$（带重复的）：

$$
\begin{pmatrix}
\lambda_1 & 0 & 0 & \cdots & 0 \\
0 & \lambda_2 & 0 & \cdots & 0 \\
 & & \vdots & & \\
0 & 0 & \cdots & 0 & \lambda_n \\
\end{pmatrix}
$$

>为什么对称的实数方矩阵满足每个eigenvalue的代数重数等于几何重数，可以通过归纳法来证明。

我们可以对于每个eigenvalue都找到其对应的eigenvector，而且因为每个eigenvalue的几何重数等于代数重数，所以说对于重复的eigenvalue，仍然可以找到足够个数的线性无关的eigenvectors。将上述这些eigenvectors按照列排成一个矩阵，也就是每一列都是一个eigenvector，再将每一列的norm都变成1，那么这个矩阵就是一个orthonormal矩阵，记为V，有性质$$VV^T = V^TV$$。我们有$$SV = V \Sigma$$。从而$$S = V \Sigma V^T$$。而这个形式就是对称实矩阵$$S$$的eigendecomposition，满足中间的对角矩阵$$\Sigma$$都是$$S$$的eigenvalues，而$$V$$的每一列都是对应的eigenvector，而且满足$$V$$是个orthonormal的矩阵。


>对于不是对称的一般的实矩阵$$A \in R^{n \times n}$$，也有上述类似的decomposition。会存在orthogonal matrix $$V \in R^{n \times n}$$，和$$\Sigma \in R^{n \times n}$$，满足$$A = V \Sigma V^T$$，而$$\Sigma$$是一个block对角矩阵，也就是说$$\Sigma = diag(A_1, A_2, \cdots, A_m, 0, \cdots, 0)$$，而每个$$A_i$$都是一个二维的skew-symmetric矩阵，满足

$$
\begin{pmatrix}
0 & a_i \\ 
-a_i & 0 \\ 
\end{pmatrix}
$$

>其中$$a_i$$是实数。

### 2.2 Singular Value Decomposition

现在我们再来看singular value decomposition。singular value decomposition是eigenvalue decomposition对于一般矩阵$$A \in C^{m \times n}$$的推广。

SVD的正式定义为：对于矩阵$$A \in R^{m \times n}$$，$$m \geq n$$，并且$$rank(A) = p$$，我们有三个矩阵$$U \in R^{m \times p}$$，$$V \in R^{n \times p}$$和$$\Sigma \in R^{p \times p}$$具有以下性质，$$U$$和$$V$$的列都是orthonormal的，且$$\Sigma = diag(\sigma_1, \sigma_2, \cdots, \sigma_p)$$，$$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_p$$，满足：

$$A = U \Sigma V^T$$

注意到，上述SVD的定义是对eigenvalue decomposition的推广，后者将一个对称的方阵分解为：$$A = V \Sigma V^T$$，而$$V$$是orthonormal的，$$\Sigma=diag(\lambda_1, \cdots, \lambda_n)$$。SVD使得分解任何非满秩的非方矩阵成为了可能。从下述证明我们也可以看出其实SVD也是从eigenvalue decomposition推导出来的。

**Proof:**

对于一个矩阵$$A \in R^{m \times n}$$，$$m \geq n$$而且$$rank(A)=p$$，那么我们有

$$A^{T}A \in R^{n \times n}$$

是对称矩阵而且是positive semi-definite的。所以说它可以被进行eigenvalue decomposition，并且它的eigenvalues是非负的，表示为：$$\sigma_1^2 \geq \cdots \geq \sigma_n^2$$，以及相对应的orthonormal eigenvectors $$v_1, \cdots, v_n$$。这些非负的$$\sigma_i$$就叫做singular values。

因为

$$ker(A^TA) = ker(A)$$

和

$$range(A^TA) = range(A^T)$$

我们有$$span(v_{i_1},\cdots, v_{i_p}) = range(A^T)$$以及$$span(v_{i_{p+1}},\cdots,v_{i_{n}}) = ker(A)$$，这里$$i_1,\cdots,i_n$$是$$1,\cdots,n$$的一个排列，而且假设$$\sigma_{i_1} \geq \sigma_{i_2} \geq \cdots \geq \sigma_{i_p}$$。因为$$v_1,\cdots,v_n$$构成$$R^{n}$$空间的一组基，所以上述结果一定成立。从而，我们定义

$$u_k = \frac{1}{\sigma_k}Av_{i_k}$$

其中$$k=1,\cdots,p$$。也就是说，$$Av_{i_k} = \sigma_k u_k$$。因为$$v_{i_1},\cdots,v_{i_p}$$并不在$$ker(A)$$内，所以上述$$u_k$$都不是0。而且这些$$u_k \in R^{m}$$还是orthonormal的：

$$<u_i,u_j> = \frac{1}{\sigma_i \sigma_j}<Avi, Avj> = \frac{1}{\sigma_i \sigma_j}<v_i, A^TAv_j> = \delta_{ij}$$

将集合$$(u_1, \cdots, u_p)$$拓展到$$R^m$$的一个基$$(u_1, \cdots, u_m)$$。因为$$Av_i = \sigma_i u_i$$，所以我们有：

$$
\begin{pmatrix}
 & \sigma_{i_1} & 0 & 0 & \cdots & 0 \\
 & 0 & \sigma_{i_2} & 0 & \cdots & 0 \\
A(v_1, \cdots, v_n) = (u_1, \cdots, u_m) & 0 & \cdots & \sigma_{i_p} & \cdots & 0 \\
 & \cdots & \cdots & \cdots & \cdots & 0 \\
 & 0 & \cdots & \cdots & \cdots & 0 \\
\end{pmatrix}
$$

也就是$$A\widetilde V = \widetilde U \widetilde \Sigma$$，因此

$$A = \widetilde U \widetilde \Sigma \widetilde V^T$$

现在将$$\widetilde U$$对应乘以0 singular values的列和$$\widetilde V^T$$对应的乘以0 singular values的行去掉，我们就得到了最终的$$A = U \Sigma V^T$$，其中$$U \in R^{m \times p}$$并且$$V \in R^{n \times p}$$。


**Another Proof:**

矩阵$$A^TA$$和$$AA^T$$都是positive semidefinite的。因此，它们的eigenvalues都是非负的。

如果$$\lambda_i$$是矩阵$$A^TA$$的一个非零的eigenvalue，其对应的eigenvector是$$v_i$$，那么我们可以有$$A^TAv_i = \sigma_i^2 v_i$$，其中$$\sigma_i = \sqrt{\lambda_i}$$。

将上面的式子两边左乘$$v_i^T$$得到：

$$v_i^T A^T A v_i = \sigma_i^2 v_i^T v_i$$

因此

$$v_i^TA^TAv_i = (Av_i)^T(Av_i) = ||Av_i||^2 = \sigma_i^2v_i^Tv_i = \sigma_i^2$$

也就是说$$||Av_i|| = \sigma_i$$。

再次回到$$A^TAv_i = \sigma_i^2v_i$$，左乘$$A$$我们得到：

$$AA^TAv_i = \sigma_i^2Av_i$$

这个式子表明$$Av_i$$是矩阵$$AA^T$$的eigenvector，对应的eigenvalue是$$\sigma_i^2$$。

结合上面两点，我们知道$$Av_i / \sigma_i$$是矩阵$$AA^T$$的一个单位长度的eigenvector，而且$$Av_i = \sigma_i u_i$$。

将上述所有的这些不为零的特征值都做上面的操作，并结合起来，就可以得到矩阵$$A$$的SVD分解了。


#### 2.3 SVD的几何意义

#### 2.3.1 定义

eigenvalue和singular values都被用来描述矩阵作用于某些向量的标量，都是描述向量模长变化幅度的数值。它们的差异在于：

* eigen vector描述的是矩阵方向不变作用（invariant action）的向量。
* singular vector描述的是矩阵最大作用（maximum action）的向量。

这里的作用（action）所指的是矩阵与向量的乘积得到的一个新的向量，几何上相当于对向量进行了旋转和拉伸，就像是对向量施加了一个作用（action），或者叫变换。

如果有向量$$v$$能使得矩阵$$A$$与之的积$$Av=\lambda v$$，$$\lambda$$是标量，那么$$\lambda$$和$$v$$就分别是$$A$$的特征值和特征向量。

如果存在单位正交矩阵$$U$$和$$V$$，使得$$A = U \Sigma V^T$$，其中$$\Sigma$$为对角矩阵，对角线上的值被称为奇异值，$$U$$和$$V$$的列分别被称为$$A$$的左奇异向量和右奇异向量。

#### 2.3.2 特征值和特征向量

特征向量给出了方向不变作用的方向。当矩阵$$A$$作用于特征向量时，它只是将特征向量$$x$$乘了个标量$$\lambda$$，缩放了$$x$$，并没有改变它的方向。

矩阵

$$
\begin{pmatrix}
0 & 2 \\
2 & 0 \\
\end{pmatrix}
$$

其特征值为-2和2，对应的特征向量分别为$$\left[1, 1\right]$$和$$\left[-1. 1\right]$$。如下图所示。

![1]({{ '/assets/images/SVD-1.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}


对于矩阵

$$
\begin{pmatrix}
1 & \frac{1}{3} \\
\frac{4}{3} & 1 \\
\end{pmatrix}
$$

单位圆被变换为椭圆，特征向量是所有向量中经过变形变换后（乘以$$A$$后）方向不变的向量。如下图所示。

![2]({{ '/assets/images/SVD-2.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

#### 2.3.3 奇异值与奇异向量

特征向量不变的方向并不保证是拉伸效果最大的方向，而这是奇异值的方向。奇异值是非负实数，通常从大到小排列。

$$\sigma_1$$是矩阵$$A$$的最大奇异值，对应右奇异向量$$v$$。$$v$$是

$$argmax_{||x||=1}(||Ax||)$$

的解，换句话说，在$$A$$与所有单位向量$$x$$的乘积中，$$Av = \sigma_1$$是最大的；但不能保证乘积之后方向不变。在下图的变化中，单位圆变成椭圆，奇异向量对应椭圆的长半轴，不同于上例中同意矩阵的特征向量的方向。

![3]({{ '/assets/images/SVD-3.jpg' | relative_url }})
{: style="width: 800px; max-width: 100%;"}

>对于最大的特征值来说，其是$$argmax_{||x||=1} <x, Ax>$$的解，因为引入了内积，所以引入了方向。

方向不变和拉伸最大都是矩阵的内禀的性质，方向不变在马尔科夫随机场中很重要；而拉伸最大方向则是数据方差分布最大的方向，所以所含的信息量最大，是PCA等方法中的核心思想。









### 参考文献

* https://zhuanlan.zhihu.com/p/30658304
* https://zhuanlan.zhihu.com/p/353637184



---
