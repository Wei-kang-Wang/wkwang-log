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

### 3. 关闭GPU进程

首先查看正在运行的GPU进程

```shell

watch -n l nvidia-smi

```

显示GPU进程PID：

![jc1]({{ '/assets/images/JC-1.PNG' | relative_url }})
{: style="width: 600px; max-width: 100%;"}

使用命令kill -9 PID关闭进程，多个进程则PID之间用空格隔开，例如：

```shell

kill -9 2874

```









---
