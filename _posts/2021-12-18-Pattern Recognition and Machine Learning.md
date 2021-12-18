---
layout: post
comments: false
title: "Pattern Recognition and Machine Learning"
date: 2021-12-17 01:09:00
tags: book-reading
---

> This post is a summary of book *PATTERN RECOGNITION AND MACHINE LEARNING*.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

---

# Introduction

The field of pattern recognition is concerned with the automatic discovery of regularities in data through the use of computer algorithms and with the use of these regularities 
to take actions such as classifying the data into different categories.

Consider the example of recognizing handwritten digits. Each digit corresponds to a $$28 \times 28$$ pixel image and so can be represented by a vector **x** comprising $$784$$ real numbers. The goal is to build a machine that will take such a vector **x** as input and that will produce the identity of $$0,...,9$$ as the output. This is a nontrivial problem due to the wide variability of handwriting. It could be tackled using handcrafted rules or heuristics for distinguishing the digits based on the shapes of the strokes, but in practice such an approach leads to proliferation of rules and of exceptions to the rules and so on, and invariably gives poor results.

Far better results can be obtained by adopting a machine learning approach in which a large set of $$N$$ digits $$\left\{ **x_1,...,x_N** \right\}$$ called a *training set* is used to tune the parameters of an adaptive model. The categories of the digits in the training set are known in advance, typically by inspecting them individually and hand-labelling them. We can express the category of a digit using *target vector* **t**, which represents the identity of the corresponding digit. Note that there is one such target vector **t** for each digit image **x**.

The result of running the machine learning algorithm can be expressed as a function **y(x)** which takes a new digit image **x** as input and that generates an output vector **y**, encoded in the same way as the target vectors. The precise form of the function **y(x)** is determined during the *training phase*, also known as the learning phase, on the basis of the training data. Once the model is trained, it can then determine the identity of new digit images, which are said to comprise a *test set*. The ability to categorize correctly new examples that differ from those used for training is known as *generalization*. In practical applications, the variability of the input vectors will be such that the training data can comprise only a tiny fraction of all possible input vectors, and so generalization is a central goal in pattern recognition.

For most practical applications, the original input variables are typically *preprocessed* to transform them into some new space of variables where, it is hoped, the pattern recognition problem will be easier to solve. This pre-processing stage is sometimes also called *feature extraction*. Note that new test data must be pre=processed using the same steps as the training data. Pre-processing might also be performed in order to speed up computation. Care must be taken during pre-processing because often information is discarded, and if this information is important to the solution of the problem then the overall accuracy of the system can suffer.

Applications in which the training data comprises examples of the input vectors along with their corresponding target vectors are known as *supervised learning* problems. Cases such as the digit recognition example, in which the aim is to assign each input vector to one of finite number of discrete categories, are called *classification* problems. If the desired output consists of one or more continuous variables, then the task is called *regression*. An example of a regression problem would be the prediction of the yield in a chemical manufacturing process in which the inputs consist of the concentrations of reactants, the temperature, and the pressure.

In other pattern recognition problems, the training data consists of a set of input vector **x** without any corresponding target values. The goal in such *unsupervised learning* problems may be to discover groups of similar examples within the data, where it is called *clustering*, or to determine the distribution of data within the input space, known as *density estimation*, or to project the data from a high-dimensional space down to two or three dimensions for the purpose of *visualization*.

## Example: Polynomial Curve Fitting


---

*If you notice mistakes and errors in this post, don't hesitate to contact me at* **wkwang0916 at outlook dot com** *and I would be super happy to correct them right away!*
