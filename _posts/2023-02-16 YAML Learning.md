---
layout: post
comments: false
title: "YAML tips"
date: 2021-12-14 01:09:00

---

> This post is a summary of YAML.


<!--more-->



---

1. 旧版本的YAML文件可能会用!!来表示注释，直接将其改成#就可以，否则python的pyyaml包就无法识别。
2. 旧版本的YAML文件里的dictionary结构，需要在每个key之后缩进一行，然后写上dictitems:，再缩进一行，再写items。但这种语法结构被淘汰了，现在的YAML文件直接在key之后缩进一行，写items就可以。老版本的YAML一样会使得pyyaml包报错。
