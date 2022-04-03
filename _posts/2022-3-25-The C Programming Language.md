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

functions会返回基本data types，structures，unions或者pointers。任何function都可以被递归调用。function的定义不可以是nested，但variables可以用block-strcutured的方式来声明。一个C program的functions可以在多个分开编译的source files里。variables可能是一个function的internal的，也可以是external但是只在一个source file里的，也可以是在整个program里都可见的。

C是一种相对来说low-level的语言。这个特征并不是贬低C语言；它只是表示C语言处理的objects和大多数computers所处理的objects是类似的，也就是characters，numbers，以及addresses。

C并没有提供operations直接处理复合objects比如character strings，sets，lists，arrays等。并没有能够治理一整个array或者string的operations。除了static definition和functions的local variable提供的stack discipline，C语言也并不定义任何storage allocation。而且C语言也没有heap或者garbage collection。C语言自身不提供input/output工具；并没有READ或者WRITE statements，以及并没有built-in的file access methods。所有的这些high-level机制都必须通过调用functions来提供。大多数的C语言的implementations都会包含一定数量的这样的functions。

相似的，C只提供直接的，single-thread的控制流：tests，loops，grouping和subprograms，并没有multiprogramming，parallel operations，synchronization，或者coroutines。

虽然说上述这些特征的缺失看似是很大的缺陷（比较两个character strings我居然还需要调用函数？），但是将一个语言的体量压缩到合适的大小有很大的好处。因为C语言本身很小，所以它可以很快的被学会。

很多年来，C的定义都是由第一版的the C programming language里的reference manual来定义的。1983年，ANSI建立了一个委员会来提供一个现代的，全面的C的定义。ANSI C作为C的定义，在1988年完成。这个定义的大多数特征都已经被现代编译器所支持。

ANSI C标准或者定义是基于原版的reference manual的。语言只有很少的改动；ANSI C的一个要求是之前已经存在了的programs需要仍然保持有效，或者说如果失效了，那么编译器也应该能够给出错误的提示。

对于大多数程序员来说，最重要的一个改变就是声明以及定义function的语法变了。现在一个function的声明可以包含这个function的arguments的描述；从而定义的语法改变了。这个新多出来的信息使得编译器能够更好的发现由不匹配的arguments导致的错误；在我们的使用经验里，这个改动是很有意义的。

还有一些小规模的改动。structure assignment和enumeration，现在正式成为了语言的一部分。floating-point computation可以在单一精度下完成了。算术的性质，特别是unsigned types，也被正式提出。preprocessor变得更加精细了。绝大部分的这些改变对于程序员来说基本没有影响。

ANSI C标准的第二个重要的贡献是提出了一个用来辅助C的library的定义。它将用来access operation system（比如，read或者write files）的functions，formatted input和output，memory allocation，string manipulation，like列入说明。一个standard headers的集合用来提供functions和data types声明的统一的access。用这个library来和操作系统进行交互的programs可以确保没有问题。这个library里的大部分内容也和UNIX系统里的标准I/O library兼容。

因为C提供的control structures和data types直接被绝大多数电脑支持，用来支持自带programs的run-time library十分小。标准library functions必须明确调用，这样才能避免在不需要的时候不会出现。这些标准库函数绝大多数用C来写（除去那些它们隐藏了的操作系统的细节），并且是可以跨平台移动的。

尽管C能够和绝大多数电脑兼容，它并不是为了任何特殊的电脑系统架构而设计的。只需要注意，我们就很容易写出可以移植的programs，也就是说，这些programs在任何硬件平台上都是一样的效果。ANSI C标准定义了这种可移植性，并且规定了一些常数来描述program在哪个机器上运行。

C并不是一个strongly-typed language，但是它进化了，它的type-checking被强化了。最初的C语言不喜欢，但是也不禁止，pointers和integers互相交换数据；但新的标准要求清晰的变量声明以及明确的转换定义。Compilers会警告所有的type errors，并且并不会对不兼容的数据类型进行转换，会直接报错。函数的声明也有类似的强化。但是，C语言还是保持了程序员自己知道自己在做什么事的原则；C语言只需要程序员清晰的把他们的意图说出来。

C语言，和任何其它语言一样，也有它的缺点。有一些operators有着错误的precedence；有一些部分的语法可以改得更好。但是，C语言已经被证明是一个极为高效且极具表达能力的语言，并且对于绝大多数应用领域都是如此。

这本书的结构如下。Chapter1是C核心部分的一个tutorial。目的是让读者尽快的开始，因为学习语言的最好方法就是尽快的用它来写program。这个tutorial认为读者对于编程有着最基本的认识；对于电脑，编译，或者类似n=n+1这种expression并没有任何的解释。这本书并不是为了变成一本数据结构或者算法的书，我们要集中注意力在语言本身。

Chpater2到Chapter6细节的讨论了C语言各个方面的内容，要比Chapter1更加详细，更加正式。Chapter2介绍基本的数据类型，operators和expressions。Chapter3介绍control flow：if-else，switch，while，for。Chapter4介绍了functions和program structure——external variables，scope规则，多个source files等——并且也接触到了preprocessor。Chapter5讨论了pointers和address算术。Chapter6介绍了structures和unions。

Chapter7介绍了标准库，其提供了和操作系统交互的interface。这个library是由ANSI标准定义的，并且应该能够支持所有可以支持C的机器，从而用它来操作input，output以及其它access操作系统的programs在任何机器上都应该是保持不变的，不需要随着机器的不同而更改代码。

Chapter8介绍了C programs和UNIX操作系统之间的一个interface，主要集中在input/output，file system和storage allocation。尽管这章的内容是为了UNIX系统设计的，用其他系统的程序员仍然可以在其中找到有用的内容，比如说一个标准库是如何被应用的。

Appendix A包含了一个语言reference manual。C语言语法和语义的官方文件应该是ANSI标准。然而ANSI标准是为了编译器设计的人写的。而此处的reference manual用更简洁的方式来介绍了语言的定义，舍去了那些标准化的内容。Appendix B是标准库的一个总结，也是为了程序员而不是编译器设计员而写的。Appendix C是现在这版C语言和原版的改动的总结。

ANSI标准拥有对C语言的最终解释权。


## Chapter 1: A Tutorial Introduction







        
        


                

















































---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
