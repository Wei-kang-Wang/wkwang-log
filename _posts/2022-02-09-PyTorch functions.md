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

### scatter_ 和 scatter method

这个method是tensor所拥有的，而scartter和scatter_的用法是一样的，区别仅在于，scatter method会返回一个新的tensor，而scatter_仅改变原tensor，而不返回新的tensor。在PyTorch里，所有带一个
下划线结束的tensor的method都是类似的效果。所以我们仅介绍scatter method的用法即可。


---
