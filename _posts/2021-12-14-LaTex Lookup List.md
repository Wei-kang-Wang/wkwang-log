---
layout: post
comments: false
title: "LaTex Lookup List"
date: 2021-12-14 01:09:00

---

> This post is a summary of common used LaTex grammer.


<!--more-->



---


## Latex括号


| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 圆括号，小括号 | \left( \frac{a}{b} \right) | $$\left( \frac{a}{b} \right)$$ |
| 方括号，中括号 | \left[ \frac{a}{b} \right] | $$\left[ \frac{a}{b} \right]$$ |
| 花括号，大括号 | \left\{ \frac{a}{b} \right\} | $$\left\{ \frac{a}{b} \right\}$$ |
| 尖括号 | \left\langle \frac{a}{b} \right\rangle | $$\left\langle \frac{a}{b} \right\rangle$$ |
| 双竖线，范式 | \left \| \frac{a}{b} \right\| | $$\left \| \frac{a}{b} \right\|$$ |
| 取整符号 | \left\lfloor \frac{a}{b} \right\rfloor | $$\left\lfloor \frac{a}{b} \right\rfloor$$ |
| 取顶符号 | \left\lceil \frac{a}{b} \right\rceil | $$\left\lceil \frac{a}{b} \right\rceil$$ |
| 混合符号 | \left( 0,1 \right] | $$\left( 0,1 \right]$$ |
| 单侧括号 | \left .\frac{a}{b} \right\} | $$\left .\frac{a}{b} \right\}$$ |


## Latex字体
| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 显示直立文字 | \textup{文本} | $$\textup{文本}$$ |
| 意大利斜体 | \textit{文本} | $$\textit{文本}$$ |
| slanted斜体 | \textsl{文本} | $$\textsl{文本}$$ |
| 显示小体大写文本 | \textsc{文本} | $$\textsc{文本}$$ |
| 中等字体 | \textmd{文本} | $$\textmd{文本}$$ |
| 加粗 | \textbf{文本} | $$\textbf{文本}$$ |
| 默认值 | \textnormal{文本} | $$\textnormal{文本}$$ |
| 斜体 | \emph{文本} | $$\emph{文本}$$ |
| 细体字 | \textlf{文本} | $$\textlf{文本}$$ |
| 使用等宽字体 | \texttt{文本} | $$\texttt{文本}$$ |
| 使用无衬线字体 | \textsf{文本} | $$\textsf{文本}$$ |
| 所有字母大写 | \uppercase{文本} | $$\uppercase{文本}$$ |
| 空心字母 | 先导入包\usepackage{amsfonts,amssymb}，\mathbb{文本} | $$\mathbb{文本}$$ |


## Latex集合
| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 集合的大括号 | \{ ... \} | $$\{ ... \}$$ |
| 集合中的| | \mid | $$\mid$$ |
| 属于 | \in | $$\in$$ |
| 不属于 | \not\in或者\notin | $$\not\in$$ |
| A包含于B | A\subset B | $$A\subset B$$ |
| A真包含于B | A\subsetneqq B | $$A\subsetneqq B$$ |
| A包含B | A\supset B | $$A\supset B$$ |
| A真包含B | A\supsetneqq B | $$A\supsetneqq B$$ |
| A不包含于B | A\not\subset B | $$A\not\subset B$$ |
| A交B | A\cap B | $$A\cap B$$ |
| A并B | A\cup B | $$A\cup B$$ |
| A大交 | \bigcap A | $$\bigcap A$$ |
| A大并 | \bigcup A | $$\bigcup A$$ |
| A的闭包/补集 | \overline{A} | $$\overline{A}$$ |
| A减去B | A\setminus B | $$A\setminus B$$ |
| 空集 | \emptyset | $$\emptyset$$ |


## Latex逻辑
| 功能 | 语法 | 显示 |
| :------: | :------: | :------: |
| 存在 | \exists | $$\exists$$ |
| 存在且仅存在一个 | \exists! | $$\exists!$$ |
| 不存在 | \nexists | $$\nexists$$ |
| 所有，任意 | \forall | $$\forall$$ |
| 否 | \neg | $$\neg$$ |
| 或，析取 | \lor | $$\lor$$ |
| 与，合取 | \land | $$\land$$ |
| 除 | \div | $$\div$$ |
| 蕴含 | \implies | $$\implies$$ |
| 当且仅当 | \iff | $$\iff$$ |
| 等价 | \Leftrightarrow | $$\Leftrightarrow$$ |
| 小于 | \le | $$\le$$ |
| 大于 | \ge | $$\ge$$ |
| 小于等于 | \leq | $$\leq$$ |
| 大于等于 | \geq | $$\geq$$ |
| 不等于 | \neq | $$\neq$$　|


