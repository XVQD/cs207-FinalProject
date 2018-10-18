# Milestone 1

**Due: Thursday, October 18th at 11:59 PM**

You must clearly outline your software design for the project.  Here are some possible sections to
include in your document along with some prompts that you may want to address.

## Introduction

*Describe problem the software solves and why it's important to solve that problem*

Automatic differentiation (AD) is implemented in this software. More specifically, it can automatically differentiate of a python function up to machine precision and it can take derivatives of derivatives. Obtaining derivatives accurately is important because it is the key part of gradient-based optimization, which is the foundation of many machine learning algorithms. As a matter of fact, most machine learning problems can be divided into following steps:

1. define a function connecting some input $X$ with some output $Y$ with a set of parameters $\beta$ as $Y = f(X,\beta)$;
2. define a loss function to check how good the model is $L(X,Y,\beta)$;
3. find the parameter set $\beta$ that minimize the loss function: argmin$_\beta L(X,Y,\beta)$.

Generally the power or performance of a machine learning algorithm is limited by the third step which is can be handled by gradient-based optimization. Therefore, our software can be used in various machine learning packages and boost their performances.

Furthermore, AD can also be applied to solve differential equations in various physical systems. Such as, diffusion equations, wave equations, Navier¨CStokes equations and other non-linear equations which cannot be solved analytically. Traditional numerical method using difference method possess error much larger than machine error. Therefore, applying AD will possibly increase the accuracy of the solvers of those differential equations. 

## Background

*Describe (briefly) the mathematical background and concepts as you see fit.  You **do not** need to
give a treatise on automatic differentiation or dual numbers.  Just give the essential ideas (e.g.
the chain rule, the graph structure of calculations, elementary functions, etc).*



#### What is AD?

AD is a set of techniques to numerically evaluate the derivative of a function specified by a computer program based on the fact that every computer program execute a sequence of elementary arithmetic operations and elementary functions. Using the chain rule, the derivative of each sub-expression can be calculated recursively to obtain the final derivatives. Depending on the sequence of calculating those sub-expressions, there are two major method of doing AD: **forward accumulation** and **reverse accumulation**. 

#### Why AD?

Traditionally, there are two ways of doing differentiation, i.e., symbolic differentiation (SD) and numerical differentiation (ND). SD gives exact expression of the derivatives and produce differentiation up to machine precision, while SD is very inefficient since the expression could become very during differentiation. ND on the other hand, suffers from round-off errors (or truncate error), which leads to bad precision. Moreover, both ND and SD have problems with calculating higher derivatives and they are slow for vector inputs with large size. AD solves all of these problems nicely.

#### How to do AD?

Considering a simple function:
$$z = \cos(x)\sin(y) + \frac{x}{y}$$
In AD, its computational graph for forward accumulation method looks like:
<img src="Fig1.png" width="400">
Accoring to the graph, the simple function can be rewritten as

\begin{align}
z = \cos(x)\sin(y) + \frac{x}{y}=\cos(w_1) \sin(w_2) + \frac{w_1}{w_2}=w_3 w_4+w_6=w_5 + w_6=w_7
\end{align}
The derivates with respect to $x$ and $y$ can be calcualted according to chain rule as:

\begin{align}
\frac{\partial z}{\partial x}&=\frac{\partial z}{\partial w_7}\left(\frac{\partial w_7}{\partial w_5}\frac{\partial w_5}{\partial w_3}\frac{\partial w_3}{\partial w_1}+\frac{\partial w_7}{\partial w_6}\frac{\partial w_6}{\partial w_1}\right)\frac{\partial w_1}{\partial x}\\
\frac{\partial z}{\partial y}&=\frac{\partial z}{\partial w_7}\left(\frac{\partial w_7}{\partial w_5}\frac{\partial w_5}{\partial w_4}\frac{\partial w_4}{\partial w_2}+\frac{\partial w_7}{\partial w_6}\frac{\partial w_6}{\partial w_2}\right)\frac{\partial w_2}{\partial y}\\
\end{align}

Therefore $\frac{\partial z}{\partial x}$ and $\frac{\partial z}{\partial y}$ are just the combinations of derivatives of elementary functions, which can be calculated analytically. In forward accumulation, the chain rule are applied from inside to outside. Computationally, the values of $w_i$ and their derivatives are store along the chain accumulatively.

## How to Use *AutoDiff*

*How do you envision that a user will interact with your package?  What should they import?  How can they instantiate AD objects?*

The user will first import our package.
```
import autodiff as ad
```
Then depending on the type of expressions they have, they will employ one of the following methods.

#### Scalar functions of scalar values
Say the user wants to get the gradient of the expression $f = alpha * x + 3$.
The user will first create a variable x and then define the symbolic expression for `f`.
```
x = ad.Variable(name = "x")
f = alpha * x + 3
```
Then when they want to evaluate the gradients of f with respect to x, they will do
```
grad_x = ad.gradients([f], [x])
```
grad_x will then contain the gradient of f with respect to x.

#### Scalar functions of vectors
Say the user wants to get the gradient of the expression $f = x_1 x_2 + x_1$. 

The user will first create two variables `x1` and `x2` and then define the symbolic expression for `f`.
```
x1 = ad.Variable(name = 'x1')
x2 = ad.Variable(name = 'x2')
f = x1 * x2 + x_1
```
Then when they want to evaluate the gradients of f with respect to x1 and x2, they will do
```
grad_x1, grad_x2 = ad.gradients([f], [x1, x2])
```
grad_x1, grad_x2 will then contain the gradient of f with respect to x1 and x2.

#### Vector functions of vectors
Say the user wants to get the gradients of the system of functions 
$$f_1 = x_1 x_2 + x_1$$
$$f_2 = \frac{x_1}{x_2}$$
The user will first create two variables `x1` and `x2` and then define the symbolic expression for `f`.
```
x1 = ad.Variable(name = 'x1')
x2 = ad.Variable(name = 'x2')
f1 = x1 * x2 + x_1
f2 = x1 / x2
```
Then when they want to evaluate the gradients of f with respect to x1 and x2, they will do
```
grad_f1, grad_f2 = ad.gradients([f1, f2], [x1, x2])
```
grad_f1, grad_f1 will contain the gradients of f1 and f2 with respect to x1 and x2.

## Software Organization

*Discuss how you plan on organizing your software package.*
#### Directory structure
*What will the directory structure look like?*  
```
/myproj
    /myproj
        __init__.py
        autodiff.py
        tests/
            __init__.py
            tests.py
    README.md
    setup.py
    LICENSE
```
#### Modules
*What modules do you plan on including?  What is their basic functionality?*
#### Test
*Where will your test suite live?  Will you use `TravisCI`? `Coveralls`?*

The test suite will live on a tests.py file in tests folder. We will use both `TravisCI` and `Coveralls`.

#### Package
*How will you distribute your package (e.g. `PyPI`)?*

We will use PyPI to distribute our package.

## Implementation
*Discuss how you plan on implementing the forward mode of automatic differentiation.*

#### Data structures
*What are the core data structures?*

* Computation graph and Node
* Overloaded operators such as add, mul, etc.
* Lists to hold function expressions and variables

Since we already hold our function expressions and variables in lists, we are able to accommodate the cases of scalar function of vectors and vector functions of vectors without additional data structures.

#### Classes
*What classes will you implement?*

a DualNumbers/Variable class to declare the variables, perform single expression derivative evaluation and include overloaded operators
 
*What method and name attributes will your classes have?*

* DualNumbers/Variable class
    * Methods
        * \__add\__
        * \__radd\__
        * \__sub\__
        * \__rsub\__
        * \__mul\__
        * \__rmul\__
        * \__pow\__
        * \__rpow\__
        * \__eq\__
        * \__lt\__
        * \__le\__
        * sin
        * cos
        * exp
    * Attributes
        * variable name

#### External dependecies
*What external dependencies will you rely on?*
    
* Numpy?
* Math?

#### Elementary functions
*How will you deal with elementary functions like `sin` and `exp`?*

* Include them as methods in DualNumbers/Variable class