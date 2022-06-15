---
layout: post
comments: false
title: "Linux Commands"
date: 2022-06-04 01:09:00

---

> This post of common Linux commands.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

### 1. 实时监控GPU运行状态

```c

watch -d -n 0.5 nvidia-smi

```

其中-d会高亮显示变化的部分。-n 0.5表示每0.5秒更新一次。


### 2. 旋转显示屏

```c

xrandr -o left       /*向左旋转90度*/
xrandr -o right      /*向右旋转90度*/
xrandr -o inverted   /*上下翻转*/
xrandr -o normal     /*恢复正常*/

```







---
