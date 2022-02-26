---
layout: post
comments: false
title: "PyTorch functions"
date: 2022-02-09 01:09:00

---

> This post of different PyTorch functions.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## torch.tensor

## torch.nn


### 1. scatter_ 和 scatter method

这个method是tensor所拥有的，而scartter和scatter_的用法是一样的，区别仅在于，scatter method会返回一个新的tensor，而scatter_仅改变原tensor，而不返回新的tensor。在PyTorch里，所有带一个
下划线结束的tensor的method都是类似的效果。所以我们仅介绍scatter method的用法即可。

scatter method的用法为：

```python
target.scatter(dim, index, src)
```
> * target：即目标张量，将在该张量上进行映射
> * src：即源张量，将把该张量上的元素逐个映射到目标张量上
> * dim：指定轴方向，定义了填充方式。对于二维张量，dim=0表示逐列进行行填充，而dim=1表示逐列进行行填充
> * index: 按照轴方向，在target张量中需要填充的位置

为了保证scatter填充的有效性，需要注意：
* （1）target张量在dim方向上的长度不小于source张量，且在其它轴方向的长度与source张量一般相同。这里的一般是指：scatter操作本身有broadcast机制。
* （2）index张量的shape一般与source相同，从而定义了每个source元素的填充位置。这里的一般是指broadcast机制下的例外情况。

例子1：
```python
a = torch.arange(10).reshape((2, 5)).float()
b = torch.zeros(3, 5)
b_ = b.scatter(0, index=torch.tensor([[1,2,1,1,2], [2,0,2,1,0]], dtype=torch.int64), a)
print(b_)

# torch.tensor([[0., 6., 0., 0., 9.],
                [0., 0., 2., 8., 0.],
                [5., 1., 7., 0., 4.]])
```

例子2：
scatter函数的一个典型应用就是在分类问题中，将目标标签转换为one-hot编码形式，如：
```python
labels = torch.tensor([1, 3], dtype=torch.int64)
targets = torch.zeros(2, 5)
targets_ = targets.scatter(1, labels.unsqueeze(-1), 1.0)
print(targets_)

# torch.tensor([[0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0.]])
```


                

---
