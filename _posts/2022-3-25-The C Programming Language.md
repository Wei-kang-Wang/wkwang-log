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

让我们对C做一个快速的入门。我们的目的是在真实的programs显示这个语言的最核心的部分，而不要去管一些细枝末节的东西。此时，我们并不打算全面甚至不打算精确的介绍内容。我们只希望能尽快让读者写出有用的programs，为了做到这点，我们需要集中注意力在基础上：variables和constants，arithmetic，control flow，functions，以及input和output基础。我们这一章并不打算介绍写大型C程序所需要的内容，这些内容包括pointers，structures，C语言里丰富的operators，一些control flow statements，以及标准库。

我们上述这种介绍方式也有着缺点，最显著的就是C语言任何一部分的完整解释在这里都是没有的。而且这一章里的programs因为并没有用到C语言的全部特性（故意为之），所以并不一定是实现起来效率最高最简洁的。另一个缺点就是接下来的章节可能会重复这一章节的内容。

不管怎样，有经验的程序员能在这一章里学到有用的新思想，新手能够学会自己写一些有用的小的C程序。


### 1.1 Getting Started

学习一门新的编程语言的唯一方法就是用它来写程序。对所有语言来说，第一个程序都是print hello world!，在C语言里这样实现：

```c
# include <stdio.h>

main()
{
    printf("hello, world!\n");
}
```

如何运行这个C程序取决于其所在的系统。举例来说，如果在UNIX操作系统上，你需要创建一个.c的程序，比如说hello.c，然后用如下command来编译：

```c
cc hello.c
```

如果程序本身没有问题，那么编译的过程会安静的进行，然后生成一个executable文件，叫做a.out，用如下command来运行：

```c
a.out
```

他会输出hello world!。

在不同的操作系统里，上述过程会有所不同。

现在我们来对上面那段程序进行一些解释。一个C程序，不管它大小是多少，都有着functions和variables。一个function有着statements来指示需要做的计算，并且variables会记下计算过程中需要的值。C语言里的functions类似于Fortran语言里的subroutines和functions或者Pascal语言里的procedures和functions。在我们上面的程序里，我们有一个function叫做main。通常情况下，你可以随意命名你的程序，但是main是很特殊的：你的程序永远会从main function开始执行。这也表明任何C程序都必须在某个地方有一个main function。

main function经常会在内部调用其它的function来完成它的任务，有些是你自己定义的，有些是库里提供的。上面程序里的第一行#include <stdio.h>告诉编译器要将标准input/output库里的信息包含进来；这一行在很多C程序的开头都会出现。标准库会在Chapter7和Appendix B里被介绍。

一个在functions之间交换数据的方法就是一个function，提供一系列的values，叫做arguments，给被调用的另一个function。一个function的arguments放在这个function的名字后面，用括号括起来。在我们上面的程序里，main function并没有参数，于是在function名字后面是一对空的括号。

一个function里面的statements用花括号括起来，main function只有一句statement，printf("hello, world!\n")。一个function可以通过叫他的名字来调用，这个名字后面跟着用括号括起来的arguments，所以说这个statement里用到的function，printf，通过叫他的名字来调用，并且提供了括号括起来的arguments："hello, workd!\n"。printf function是一个库函数，用来print输出的。

一列用双引号包起来的characters，就像"hello, world!\n"，叫做一个character string或者string constant。

上面的character string里的\n是C语言里的newline character，表示换行。如果你用下面这个方式来写，C语言编译器会报错：

```c
printf("hello, world!
");
```

printf function并不会自动提供换行，所以说如果用多个printf function也是一样的效果：

```c
# include <stdio.h>

main()
{
    printf("hello, ");
    printf("world!");
    print("\n");
}
```

注意到，\n作为一个长度为2的character string，其仅仅表达一个character的意思（换行本身就是一个character），还有很多这样的，叫做escape sequence，这些是用来实现那些常用的但却很难打出来的或者不可见的characters，比如说\t表示制表，\b表示退一格，\"表示双引号本身，\\表示反斜杠本身，等等。在section2.3里有一个全面的表格来介绍escape sequence。

我们来看一下我们所学习的第一个C语言程序：

```c
#include <stdio.h>               # include information about standard library

main()                           # define a function named main that receives no argument values
{                                # statements of main are enclosed in braces
    printf("hello, world!\n");   # main calls library function printf to print this sequence of characters; \n represents the newline character
}
```


### 1.2 Variables and Arithmetic Expressions

我们下一个程序用公式$$C = (5/9)(F - 32)$$来输出下面的华氏度表格以及其摄氏度对应的温度：

```c
0    -17
20   -6
40   4
60   15
80   26
100  37
120  48
140  60
160  71
180  82
200  93
220  104
240  115
260  126
280  137
300  148
```

这个程序仍然只含有一个function，也就是main function，虽然比hello world那个程序长一些，但并不更加复杂。这个程序将会介绍一些新的内容，包括comment，declaration，variable，arithmetic expression，loop以及formatted output。

```c
# include <stdio.h>

/* print Fahrenheit-Celsius table
   for fahr = 0, 20, ..., 300 */

main()
{
    int fahr, celsius;
    int lower, upper, step;
    
    lower = 0;      /* lower limit of temperature table */
    upper = 300;    /* upper limit */
    step = 20;      /* step size */
    
    fahr = lower;
    while (fahr <= upper){
        celsius = 5 * (fahr-32) / 9;
        printf("%d\t%d\n", fahr, celsius);
        fahr = fahr + step;
    }
}
```
上面的两行
```c
/* print Fahrenheit-Celsius table
   for fahr = 0, 20, ..., 300 */
```
是comment，也就是简要的解释一下这个程序是做什么的。任何在/\*和\*/之间的内容都会被编译器忽略掉；它们可以让一个程序的可读性更强。comment可以出现在程序的任何位置。

在C语言里，所有的variables在使用之前都需要被声明，它们通常出现在function的开头，也就是在后面需要使用这些variables的statements之前。一个declaration会宣布variables的性质；它由一个type名字和一系列variables组成，比如：

```c
int fahr, celsius;
int lower, upper, step;
```

type int表明variables是integers（整数），而type float表明variables是浮点数，也就是带小数点的数。int和float类型的variables的范围都是由操作系统决定的。

C语言除了int和float type，还提供了很多其它基础的types：

```c
char    字符，一个字节
short   短整数
long    长整数
double  双精度浮点数
```

上述type的variables的大小仍然是由操作系统决定的。C语言里还有arrays，structures，unions，这些是由基础的types组合成的；还有pointers，它表示某个variable的地址；还有functions，会返回这些不同type的variables。

上述程序里，最开始是assignment：

```c
lower = 0;
upper = 300;
step = 20;
fahr = lower;
```

这会给variables初始值。每个statement都由分号结尾。

因为我们想要打印的表格每一列的数字都是由相同的计算公式算出来的，所以我们使用一个while循环来做：

```c
while (fahr <= upper){
    ...
}
```

while循环按照下面的逻辑来运行：首先括号内的条件要被判断。如果条件为真（fahr确实不比upper大），循环体会被执行。之后括号内的条件再次被判断。如果条件为真，那么循环体再次被执行。重复这两个步骤直到循环体内的条件不满足（fahr比upper大），此时循环停止，while循环后面紧接着的statement将会被执行。

while循环的循环体可以是一句或者多句statements，如果是多句statements，那么就一定要在花括号里面，如果是一句statement，那么可以不用花括号（也可以用）：

```c
while (i < j)
    i = 2 * i;
```

>C语言和python不一样，并不是形式型语言，也就是说缩进多少并不影响代码的正确性，但是为了好看，还是按照常用的缩进（4格）。同样的，每行只写一句statement也是为了代码好看，写多句也是可以的，但可阅读性会被降低，这也是一种习惯。还有个习惯就是对于operator，一般都会前后空一格，更加清晰，比如a + b。花括号的位置也不重要。

上述代码里最重要的部分就是celsius = 5 * (fahr - 32) / 9;这一句，之所以写成先乘以5再除以9而不是直接乘以5/9是因为在C语言里，如同很多其它语言一样，integer除法会将小数部分去掉，因为5和9都是整数，所以5/9的结果就变成了0，是不正确的。

在上述代码里，又用到了printf function。此处用到了printf function更多的功能。printf function本身是一个通用的output formatting function，我们将会在Chapter7里详细介绍。它的第一个argument是要打印的一个character string，用%来表示占位符，%后面的字母表示输出的格式，比如说%d表示按照整数来输出这个argument：

```c
printf("%d\t%d\n", fahr, celsius);
```

会以整数的格式输出两个arguments：fahr和celsius，并且中间隔一个制表符（也就是空四格）。

printf第一个argument里的每一个%都需要对应好后续的arguments，而且格式也需要注意。

但我们还需要注意，printf并不是C语言的一部分；C语言里并没有处理input或者output的工具。printf只是一个标准库里十分常见的function。printf function的行为是由ANSI标准规定的，所以任何按照ANSI标准编写的C程序如果使用printf function那么都是相同的。

为了集中注意力在C语言本身，我们直到Chapter7之前都不会过多深入input和output的工具。我们会在那里再说formatted input的内容。scanf function可以用来处理formatted input，其和printf相似，但是是处理input而不是output的。

对于上述代码，我们发现其还存在着某些问题。比如说，输出的结果并不是很好看，因为输出的数字并没有对齐。这很容易解决，我们只需要在printf function的%后面再加上一个数字表述输出宽度，那么这个占位符的输出就按照指定的宽度按右对齐输出，比如说：

```c
printf("%3d %6d\n", fahr, celsius);
```

那么结果将会是：

```c
  0    -17
 20     -6
 40      4
 60     15
 80     26
100     37
120     48
140     60
160     71
180     82
200     93
220    104
240    115
260    126
280    137
300    148
```

比格式问题更复杂的问题是，我们上面使用的都是整数运算，不够精确，比如0华氏度实际上是-17.8摄氏度，而不是-17摄氏度。为了获得更精确的结果，我们需要用浮点数而不是整数来计算。这需要对代码进行一些调整：

```c
#include <stdio.h>

/* print Fahrenheit-Celsius table
    for fahr = 0, 20, ..., 300; floating-point version */
main()
{
    float fahr, celsius;
    int lower, upper, step;
    
    lower = 0;           /* lower limit of temperature table */
    upper = 300;         /* upper limit */
    step = 20;           /* step size */
    
    fahr = lower;
    while (fahr <= upper) {
        celsius = (5.0 / 9.0) * (fahr - 32.0);
        printf("%3.0f %6.1f\n", fahr, celsius);
        fahr = fahr + step;
    }
}
```

上述代码和前面那个版本的差不多，除了fahr和celsius被声明为float，并且while循环里计算的公式写的更加自然。我们并不能使用(5 / 9)因为其算的结果是0。而带有小数点，即使是.0仍然说明其是float而不是integer，所以说5.0 / 9.0的结果并不会将小数点后的部分截掉，因为它是两个浮点数的计算结果。

如果一个arithmetic operator的operands是integers，那么这个operation就是一个integer operation，比如说5 / 9。如果一个arithmetic operator只要有一个floating-point operand，那么即使其它的还是integer operands，这些integers会被转换为floating-point数，然后再进行operation的计算，比如说5.0 / 9。在上面的代码里，即使我们写成fahr - 32，其中32是一个integer，因为fahr在声明的时候是float，所以在这个arithmetic operation，减法，里，32被转换为floating-point数。但为了观感，我们还是写成了farh - 32.0。

合适integers会被转换为floating point数的完整的细节的规定在Chapter2里会详细说明。对于现在来说，注意这个assignment：

```c
fahr = lower;
```

以及这个：

```c
while (farh <= upper)
```

它两也在operator里遇到了type不同的variables，而在这两种情况下，int仍然被转换为了float来进行operation。

在printf function里，%3.0f说明按照floating-point的形式输出，其宽度至少是3个字符，但不输出小数点以及小数点之后的内容。$6.1f表明按照floating-point的形式输出，输出宽度至少是6个字符，小数点后保持一位。上述代码执行之后的结果是：

```c
  0  -17.8
 20   -6.7
 40    4.4
 ...
```

我们还可以用更省略的方式来表示，%6f表示按照floating-point方式输出，宽度至少为6，不带小数点，%.2f表示按floating-point的方式输出，小数点后保留两位，但是宽度不做限制，%f只是表示按照floating-point的方式输出。

> %d     按照十进制整数的形式输出
> %6d    按照十进制整数的形式输出，至少6个字符宽
> %f     按照浮点数输出
> %6f    按照浮点数输出，至少6个字符宽
> %.2f   按照浮点数输出，小数点后保留两位
> %6.2f  按照浮点数输出，至少6个字符宽，小数点后保留两位

printf里还会用%o表示八进制输出，%x表示十六进制输出，%c表示字符，%s表示字符串，%%表示%本身。


### 1.3 The For Statement


















        
        


                


















































---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
