---
layout: post
comments: false
title: "The C Programming Language"
date: 2021-11-29 01:09:00
tags: book-reading
---

> 这是The C Programming Language(second edition)这本书的翻译版，加上了个人一些拙略的理解。


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---


## Preface

自从1978年 The C Programming Language 这本书出版以来，计算机领域经历了一场革命。大型计算机的功能越来越强大，而个人计算机的性能也可以与十多年前的大型机相媲美。在此期间，C语言也在悄悄地演进，其发展早已超出了它仅仅作为UNIX操作系统编程语言的初衷。

C语言的普及程度的逐渐增加以及该语言本身的发展，加之很多组织开发出了与其设计有所不同的编译器，所有的这一切都要求对C语言有一个比本书第一版更加精确、更适应其发展的定义。1983年，美国国家协会（ANSI）成立了一个委员会，其目标是制定一个“无歧义性的且与具体机器无关的C语言定义”，而同时又要保持C语言原有的“精神”。结果产生了C语言的ANSI标准。

ANSI标准规范了一些在本书第1版中提及但没有具体描述的结构，特别是解构赋值和枚举。该标准还提供了一种新的函数声明形式，允许在使用过程中对函数的定义进行交叉检查。标准中还详细说明了一个具有标准输入/输出、内存管理和字符串操作等扩展函数集的标准库。它精确地说明了在C语言原始定义中并不明晰的某些特性的行为，同时还明确了C语言中与具体机器相关的一些特性。

本书第2版介绍的是ANSI标准定义的C语言。尽管我们已经注意到了该语言中已经变化了的地方，但我们还是决定在这里只列出它们的新形式。最重要的原因是，新旧形式之间并没有太大的差别；最明显的变化是函数的声明和定义。目前的编译器已经能够支持该标准的大部分特性。

我们将尽力保持本书第1版的简洁性。C语言并不是一种大型语言，也不需要用一本很厚的书来描述。我们在讲解一些关键特性（比如指针）时做了改进，它是C语言程序设计的核心。我们重新对以前的例子进行了精炼，并在某些章节里增加了一些新例子。例如，我们通过实例程序对复杂的声明进行处理，以将复杂的声明转换为描述性的说明或反之。

## Preface for the first edition

C语言是一种通用的程序设计语言，其特点包括简洁的表达式、流行的控制流和数据结构、丰富的运算符集等。C语言不是一种很“高级”的语言，也不“庞大”，并且不专用于某一特定的应用领域。但是，C语言的限制少，通用性强，这使得它比一些公认为更强大的语言使用更方便、效率更高。

C语言最初是由Dennis Ritchie为UNIX操作系统设计的，并在DEC PDP-11计算机上实现。UNIX操作系统、C编译器和几乎所有的UNIX应用程序都是用C语言写的。同时，还有一些适用于其他机器的编译器产品，比如IBM System/370、Honeywell 6000、Interdata 8/32等。但是，C语言不受限于任何特定的机器或系统，使用它可以很容易地编写出不经修改就可以运行在所有支持C语言的机器上的程序。

本书的目的是帮助读者学习如何用C语言编写程序。本书的开头有一个指南性的引言，目的是使新用户能尽快地开始学习；随后在不同的章节中介绍了C语言的各种主要特性；本书的附录中还包括一份参考手册。本书并不仅仅只是讲述了语言的一些规则，而是采用阅读别人的代码、自己编写代码、修改某些代码等不同的方式来指导读者进行学习。书中的大部分例子都可以完整的运行，而不只是孤立的程序段。

本书并不是一本有关程序设计的入门性手册，他要求读者熟悉基本的程序设计概念，如变量、赋值语句、循环和函数等。尽管如此，初级的程序员仍然可以阅读此书，并借此学会C语言。

C语言是一种令人愉快的、具有很强表达能力的通用的语言，适合于编写各种程序。它容易学习，并且随着使用经验的增加，使用者会越来越感到得心应手。


## Introduction

C语言是一种通用性的编程语言。C语言在开发的过程中和UNIX系统有着很紧密的联系，因为UNIX和绝大多数UNIX上的程序都是用C语言编写的。C语言，并不局限于某一特定的操作系统或者机器；而且尽管因为它是被用于编写操作系统和编译器而叫做“system programming language"，C语言其实在很多领域都有很大的作用。

很多C语言里的重要想法来源于Martin Richards开发的BCPL语言。BCPL语言对C语言的影响，通过B语言来间接实现，而B语言是由Ken Thompson在1970年在DEC PDP-7上为第一个UNIX系统开发的。

BCPL和B是typeless语言。相对来说，C提供了一系列不同的data types。基础的types包括characters，intergers以及不同的size的floating-point numbers。而且，还有由pointers，arrays，structures和unions构造的derived data types。Expressions是由operations和operands组成的；任何expression都可以是一个statement（包括assignment，function call等等）。pointers提供和机器无关的地址计算。

C语言为良好的结构的programs提供了基础的control-flow constructions：statement grouping，decision making（if-else），selecting one of a set of possible cases（switch），looping with the termination test at the top（while，for）或者at the bottom（do），以及early loop exit（break）。

functions会返回基本data types，structures，unions或者pointers。任何function都可以被递归调用。



        
        


                

















































---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
