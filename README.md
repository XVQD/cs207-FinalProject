# Automatic Differentiation

[![Build Status](https://travis-ci.org/XVQD/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/XVQD/cs207-FinalProject.svg?branch=master)

[![Coverage Status](https://coveralls.io/repos/github/XVQD/cs207-FinalProject/badge.svg?branch=master)](https://coveralls.io/github/XVQD/cs207-FinalProject?branch=master)

## Introduction

Automatic differentiation (AD) is implemented in this software. More specifically, it can automatically differentiate of a python function up to machine precision and it can take derivatives of derivatives. Obtaining derivatives accurately is important because it is the key part of gradient-based optimization, which is the foundation of many machine learning algorithms. As a matter of fact, most machine learning problems can be divided into following steps:

1. define a function connecting some input $X$ with some output $Y$ with a set of parameters $\beta$ as $Y = f(X,\beta)$;
2. define a loss function to check how good the model is $L(X,Y,\beta)$;
3. find the parameter set $\beta$ that minimize the loss function: argmin$_\beta L(X,Y,\beta)$.

Generally the power or performance of a machine learning algorithm is limited by the third step which is can be handled by gradient-based optimization. Therefore, our software can be used in various machine learning packages and boost their performances.

Furthermore, AD can also be applied to solve differential equations in various physical systems. Such as, diffusion equations, wave equations, Navier–Stokes equations and other non-linear equations which cannot be solved analytically. Traditional numerical method using difference method possess error much larger than machine error. Therefore, applying AD will possibly increase the accuracy of the solvers of those differential equations. 

## Background

#### What is AD?

AD is a set of techniques to numerically evaluate the derivative of a function specified by a computer program based on the fact that every computer program execute a sequence of elementary arithmetic operations and elementary functions. Using the chain rule, the derivative of each sub-expression can be calculated recursively to obtain the final derivatives. Depending on the sequence of calculating those sub-expressions, there are two major method of doing AD: **forward accumulation** and **reverse accumulation**. 

#### Why AD?

Traditionally, there are two ways of doing differentiation, i.e., symbolic differentiation (SD) and numerical differentiation (ND). SD gives exact expression of the derivatives and produce differentiation up to machine precision, while SD is very inefficient since the expression could become very during differentiation. ND on the other hand, suffers from round-off errors (or truncate error), which leads to bad precision. Moreover, both ND and SD have problems with calculating higher derivatives and they are slow for vector inputs with large size. AD solves all of these problems nicely.

#### How to do AD?

Considering a simple function:
$$z = \cos(x)\sin(y) + \frac{x}{y}$$
In AD, its computational graph for forward accumulation method looks like:
<img src="figs/Fig1.png" width="400">
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

## Operator overloading

Operator overloading is applied in this package to realize AD. Values and derivatives of functions are updated and passed at each operation. That means if we have two functions $\omega_1 (x_1,x_2,...,x_n)$ and $\omega_2 (x_1,x_2,...,x_n)$ with their derivatives ${\partial\omega_1}/{\partial x_i}$ and ${\partial\omega_1}/{\partial x_i}$ and second derivatives ${\partial^2\omega_1}/{\partial x_i}{\partial x_j}$ and ${\partial^2\omega_2}/{\partial x_i}{\partial x_j}$, $\forall i,j$. For a new variable $\omega_3 = \omega_1*\omega_2$, where $*$ is some operator (addition, multiplication, division, power, etc). Accoring to specific types of the operator we can write down analytical expression of the values as well the derivatives of $\omega_3$ as functions of $\omega_1$ and $\omega_2$ and their derivatives. The following expression shows these analytical expressions:

| operator 	| &nbsp; &nbsp; &nbsp;value $\omega_3$&nbsp; | &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; first derivative $\frac{\partial \omega_3}{\partial x_i}$  &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp; &nbsp;| second derivative $\frac{\partial^2 \omega_3}{\partial x_i \partial x_j }$                                                                  	|   	|
|:---:|:---:|:---:|:---:|---	|
|    $+$   	|         $\omega_1 + \omega_2$         	|$\frac{\partial \omega_1}{\partial x_i} +\frac{\partial \omega_2}{\partial x_i}$                                    	|                                                                                                                                                                                                                                                                                     $\frac{\partial^2 \omega_1}{\partial x_i \partial x_j } +\frac{\partial^2 \omega_2}{\partial x_i \partial x_j }$                                                                                                                                                                                                                                                                                    	|   	|
|    $-$   	|    $\omega_1 - \omega_2$   	|                                    $\frac{\partial \omega_1}{\partial x_i} - \frac{\partial \omega_2}{\partial x_i}$                                   	|                                                                                                                                                                                                                                                                                     $ \frac{\partial^2 \omega_1}{\partial x_i \partial x_j } -\frac{\partial^2 \omega_2}{\partial x_i \partial x_j }$                                                                                                                                                                                                                                                                                    	|   	|
|    $\times$   	| $\omega_1 \times \omega_2$ 	|                           $ \omega_2 \frac{\partial \omega_1}{\partial x_i} +\omega_1\frac{\partial \omega_2}{\partial x_i}$                           	|                                                                                                                                                                                             $ \frac{\partial \omega_1}{\partial x_i}\frac{\partial \omega_2}{\partial x_j} + \frac{\partial \omega_1}{\partial x_j}\frac{\partial \omega_2}{\partial x_i}+ \omega_2 \frac{\partial^2 \omega_1}{\partial x_i \partial x_j } + \omega_1 \frac{\partial^2 \omega_2}{\partial x_i \partial x_j }$                                                                                                                                                                                            	|   	|
|    $/$   	|    $\omega_1 / \omega_2$   	|            $\frac{1}{\omega_2} \frac{\partial \omega_1}{\partial x_i} - \frac{\omega_1}{\omega_2^2} \frac{\partial \omega_2}{\partial x_i}$            	|                                                                                                        $ -\frac{1}{\omega_2^2}\left(\frac{\partial \omega_1}{\partial x_i}\frac{\partial \omega_2}{\partial x_j} + \frac{\partial \omega_1}{\partial x_j}\frac{\partial \omega_2}{\partial x_i}\right) + \frac{2\omega_1}{\omega_2^3}\frac{\partial \omega_2}{\partial x_i}\frac{\partial \omega_2}{\partial x_j}+\frac{1}{\omega_2} \frac{\partial^2 \omega_1}{\partial x_i \partial x_j } - \frac{\omega_1}{\omega_2^2} \frac{\partial^2 \omega_2}{\partial x_i \partial x_j }$                                                                                                        	|   	|
| ^     | $\omega_1^{\omega_2}$      	| $\omega_1 ^{\omega_2-1} \omega_2\frac{\partial \omega_1}{\partial x_i}+\ln(\omega_1) \omega_1^{\omega_2} \frac{\partial \omega_2}{\partial x_i}$ 	| $\omega_1^{\omega_2-2}(\omega_2^2 - \omega_2) \frac{\partial \omega_1}{\partial x_i}\frac{\partial \omega_1}{\partial x_j}+ (\omega_1^{\omega_2-1}+\omega_2\ln(\omega_1)\omega_1^{\omega_2-1})\left( \frac{\partial \omega_1}{\partial x_i}\frac{\partial \omega_2}{\partial x_j} +\frac{\partial \omega_1}{\partial x_j}\frac{\partial \omega_2}{\partial x_i}\right)+\ln (\omega_1)^2\omega_1^{\omega_2}\frac{\partial \omega_2}{\partial x_i}\frac{\partial \omega_2}{\partial x_j} + \omega_2 \omega_1 ^{\omega_2-1}\frac{\partial^2 \omega_1}{\partial x_i \partial x_j } + \ln(\omega_1) \omega_1^{\omega_2}\frac{\partial^2 \omega_2}{\partial x_i \partial x_j }$ 	|   	|

## How to Use *AutoDiff*

There are two ways to install our package.

### Method 1: User installation via ```pip```
Our package is currently published on PyPI, as ```AutoDiff-XVQD```. With the forward mode implementation complete, users can currently install our package via ```pip```. After creating a new directory for their project, the user should go into this directory and enter in the following commands:

Within the directory the user just created, they create a virtual environment and call it `env`.
```bash
virtualenv env
```

Then they activate the virtual environment and install the package.
```bash
source env/bin/activate
pip install AutoDiff-XVQD
```

After these steps, the user can open a Python interpreter on the virtual environment they just created and now they will be able to import the module.

```python
>>> import AutoDiff.AutoDiff as ad
```

### Method 2: Installation via Github (for developers and users)
Since our package is not on PyPI yet, the user needs to create a virtual environment and manually install the package. First, the user needs to download the package from GitHub. Then they create a directory and unpack the project into that directory. 
```bash
git clone https://github.com/XVQD/cs207-FinalProject.git
```
Within the directory the user just created, they create a virtual environment and call it `env`.
```bash
virtualenv env
```

Then they activate the virtual environment and install the dependencies.
```bash
source env/bin/activate
pip install -r requirements.txt
```

After these steps, the user can open a Python interpreter on the virtual environment they just created and now they will be able to import the module.

```python
>>> import AutoDiff.AutoDiff as ad
```

### Introduction to basic usage of the package

After successful installation, the user will first import our package.
```python
>>> import AutoDiff.AutoDiff as ad
```
Then depending on the type of expressions they have, they will employ one of the following methods.

#### Scalar functions of scalar values
Say the user wants to get the gradient of the expression $f(x) = alpha * x + 3$.
The user will first create a variable x and then define the symbolic expression for `f`.
```python
>>> x = ad.Variable(2, name='x')
>>> f = 2 * x + 3
```
Note: If the user wants to include special functions like sin and exp, they need to do the following:
```python
>>> f = 2 * ad.sin(x) + 3
```
Then when they want to evaluate the gradients of f with respect to x, they will do
```python
>>> print(f.val, f.der)
```
f.val and f.der will then contain the value and gradient of f with respect to x.

Then if they want to evaluate the second derivatives of f with respect to x, they will do
```python
>>> print(f.der2)
```
f.der2 will then contain the second derivative of f with respect to x.

#### Scalar functions of vectors - Type 1
Say the user wants to get the gradient of the expression $f(x_1,x_2) = x_1 x_2 + x_1$. 

The user will first create two variables `x1` and `x2` and then define the symbolic expression for `f`.
```python
>>> x1 = ad.Variable(2,name='x1')
>>> x2 = ad.Variable(3,name='x2')
>>> f = x1 * x2 + x1
```
Then when they want to get the values and gradients of f with respect to x1 and x2, they will do
```python
>>> print(f.val, f.der)
```
f.val and f.der will then contain dictionaries of values and gradients of f with respect to x1 and x2.

Then if they want to get the second derivatives of f with respect to x1 and x2, they will do
```python
>>> print(f.der2)
```
f.der2 will then contain dictionaries of values and gradients of f with respect to x1 and x2, i.e., $\frac{\partial^2 f}{\partial x_1^2}$, $\frac{\partial^2 f}{\partial x_2^2}$, $\frac{\partial^2 f}{\partial x_1 \partial x_2}$ and $\frac{\partial^2 f}{\partial x_2 \partial x_1}$ as a dictionary with keys `'x1x1'`, `'x2x2'`, `'x1x2'` and `'x2x1'` respectively.

#### Scalar functions of vectors - Type 2
Our package is also able to get the gradient of the expression $f(x_1, x_2) = (x_1 - x_2)^2$ where $x_1$ and $x_2$ are vectors themselves. 

The user will first create two variables `x1` and `x2`, and then the symbolic expression for `f`.
```python
>>> x1 = ad.Variable([2, 3, 4], name='x1')
>>> x2 = ad.Variable([3, 2, 1], name='x2')
>>> f = (x1 - x2)**2
```
Then when they want to get the values and gradients of f with respect to $x_1$ and $x_2$, they will do
```python
>>> print(f.val, f.der, f.der2)
```
#### Vector functions of vectors
Say the user wants to get the gradients of the system of functions 
$$f_1 = x_1 x_2 + x_1$$
$$f_2 = \frac{x_1}{x_2}$$

i.e.
$$\mathbf{f}(x1,x2)=(f_1(x_1,x_2),f_2(x_1,x_2))$$
The user will first create two variables `x1` and `x2` and then define the symbolic expression for `f`.
```python
>>> x1 = ad.Variable(3, name = 'x1')
>>> x2 = ad.Variable(2, name = 'x2')
>>> f1 = x1 * x2 + x1
>>> f2 = x1 / x2
```
Then when they want to evaluate the gradients of f with respect to x1 and x2, they will do
```python
>>> print(f1.val, f2.val, f1.der, f2.der)
```
The Jacobian $\mathbf{J}(\mathbf{f})$ =(f1', f2') = (f1.der, f2.der)

They can also obtain second derivatives (Hessian matrix) by doing
```python
>>> print(f1.der2, f2.der2)
```
### Demo

The following demo shows a simple case of using the AutoDiff package to calculate the derivatives. This can be done in the directory the user creates AFTER they install the package as shown in 'installation' section.

First the user imports the package `AutoDiff`
```python
>>> import AutoDiff.AutoDiff as ad
```

Then they define the variables. For example, if they have variables $x_1$ and $x_2$, which they want to evaluate at 3 and 2 respectively, they will define them as the following. The user needs to give a name as a parameter when they define the variables.
```python
>>> x1 = ad.Variable(3, name='x1')
>>> x2 = ad.Variable(2, name='x2')
```

Then the user writes down the expression/function they want to evaluate. Most of the expression would be normal arithmetic expressions, except for the special functions such as exp, sin, cos, etc., the user needs to use ad.exp(), ad.sin(), ad.cos(), etc.
```python
>>> f = ad.sin(x1) + x2**2
```

Finally, the user can get print the value and derivatives of their expression with respect to $x_1$ and $x_2$. This will output the value as a scalar and the partial derivatives as a dictionary where the keys are the variable names and the values are the derivatives with respect to that variable.
```python
>>> print(f.val, f.der)
4.141120008059867 {'x2': 4, 'x1': -0.9899924966004454}
```
```python
>>> print(f.der2)
4.141120008059867 {'x2': 4, 'x1': -0.9899924966004454}
```

## Software Organization

#### Directory structure 
```
/cs207-FinalProject
    /docs
        Final.ipynb
        milestone1.ipynb
        milestone2.ipynb
    /AutoDiff
        __init__.py
        AutoDiff.py
        GradDesc.py
        NewtonOpt.py
        gmres.py
    /tests
        __init__.py
        test_operator.py
        test_elementary_functions.py
        test_gmres.py
        test_graddesc.py
        test_hessian.py
        test_newton.py
        test_second_derivatives.py
    /demos
        gmres_demo.py
        graddesc_demo.py
        NewtonOpt_demo.py
    README.md
    requirements.txt
    LICENSE.md
```
#### Modules

- `__init__.py`:  initialize the package by importing necessary functions from other modules

- `AutoDiff.py`:  main module of the package which implements basic data structure and algorithms of the forward automatic differentiation, including overloaded operators and special functions such as sin and trig

- `NewtonOpt.py`: module for optimization with Newton's method using automatic differentiation to calculate derivatives 

- `GradDesc.py`: module for optimization with gradient descent method using automatic differentiation to calculate derivatives 

- `GMRes.py`: module for root finding with Generalized minimal residual method using automatic differentiation to calculate matrix-vector product

#### Test

The test suite include the following files:

* `test_operator.py` - tests the overloaded operators
* `test_elementary_functions.py` - tests the elementary functions such as exp, cos, sin, etc.
* `test_second_derivatives.py` - tests the second derivatives
* `test_hessian.py` - test Hessian matrix
* `test_newton.py` - test Newton's method for optimization using AD 
* `test_graddesc.py` - test gradient descent for optimization using AD
* `test_gmres.py` - test GMRes for root finding using AD

We automate our testing using continuous integration. Every time we commit and push to GitHub, our code is automatically tested by `Travis CI` and `Coveralls` for code coverage. 

#### Package Installation

Eventually we will use PyPI to distribute our package. At this point, the user needs to download and manually install the package as following.

First, the user needs to download the package from GitHub. Then they create a directory and unpack the project into that directory.
```bash
git clone https://github.com/XVQD/cs207-FinalProject.git
```

Within the directory the user just created, they create a virtual environment and call it env.
```bash
cd yourdir
virtualenv env
```

Then they activate the virtual environment and install the dependencies.
```bash
source env/bin/activate
pip install -r requirements.txt
```

After these steps, the user can open a Python interpreter on the virtual environment they just created and now they will be able to import the package.
```python
>>> import AutoDiff.AutoDiff as ad
```

## Implementation

### Current Implementation
#### Data structures
*What are the core data structures?*

* dictionary: we use dictionaries to keep track of the partial derivatives. The keys are the variables we differentiate with respect to and the values are the actual derivatives.
* overloaded operators such as \__add\__ and \__mul\__ to add or multiply two auto-differentiation objects.

#### Classes
*What are the core classes?*

* class Variable - an auto-differentiation class with the overloaded operators 
    * attributes
        * val: np.array of the value(s) at which we want to evaluate the variable
        * name: (optional) name of variable; suggested that the name should match the name of the Variable created. If name is not supplied, then a new variable has not been created, but rather a combination of other variables has occurred
        * der: (optional) dict of str keys and np.array of float/int values str keys correspond to variable names, and values are floats/ints corresponding to the partial derivative with respect to the name key
        * der2: (optional) dict of str keys and np.array of float/int values str keys correspond to variable names, and values are floats/ints corresponding to the partial second derivative with respect to the name key
    
    * Methods
        * \__pos\__
        * \__neg\__
        * \__add\__
        * \__radd\__
        * \__sub\__
        * \__rsub\__
        * \__mul\__
        * \__rmul\__
        * \__itruediv\__
        * \__rtruediv\__
        * \__pow\__
        * \__rpow\__


* method exp()
    * input 
        * Variable object
    * output 
        * Variable object after taking exponential


* method log()
    * input 
        * Variable object
    * output 
        * Variable object after taking log


* method sin()
    * input 
        * Variable object
    * output 
        * Variable object after taking sine


* method cos()
    * input 
        * Variable
    * output 
        * Variable object after taking cosine


* method tan()
    * input 
        * Variable
    * output 
        * Variable object after taking tangent
        
* method sinh()
    * input 
        * Variable
    * output 
        * Variable object after taking sinh
      
      
* method cosh()
    * input 
        * Variable
    * output 
        * Variable object after taking cosh


* method tanh()
    * input 
        * Variable
    * output 
        * Variable object after taking tanh
        
        
* method arcsin()
    * input 
        * Variable
    * output 
        * Variable object after taking arcsin


* method arccos()
    * input 
        * Variable
    * output 
        * Variable object after taking arccos


* method arctan()
    * input 
        * Variable
    * output 
        * Variable object after taking arctan
        
* method sqrt()
    * input
        * Variable
    * output
        * Variable object after taking square root

* method sigmoid()
    * input
        * Variable
    * output
        * Variable object after taking logistic function

#### External dependecies

* numpy==1.15.4
* scipy==1.1.0

#### Elementary functions

Our elementary functions include the following: 
* exp
* log
* sin
* cos
* tan
* sinh
* cosh
* tanh
* arcsin
* arccos
* arctan

## Additional Features

### Second Derivative (Hessian)
Besides the first order derivatives we implemented as a basic version of AD, we implemented the second derivatives as well. The specific implementation of the higher derivatives does not change the exsiting code structure. The primary challenge was to define the overloaded operators and elementary functions for higher derivatives.

The second derivatives can be accessed in the following way: 
```python
>>> import AutoDiff.AutoDiff as ad
>>> x1 = ad.Variable(3, name='x1')
>>> x2 = ad.Variable(2, name='x2')
>>> f = x1 + x1*x2
>>> print(f.der2)
{'x2x2': array([0.]), 'x1x1': array([0.]), 'x2x1': array([1.]), 'x1x2': array([1.])}
```

Here `x1x1` corresponds to the entry in row 1, column 1 of the Hessian matrix, i.e. $\frac{\delta f}{x_1 x_1}$, `x1x2` corresponds to the entry in row 1, column 2 of the Hessian matrix, i.e. $\frac{\delta f}{x_1 x_2}$, and so on.

### Newton's optimizer

Newton's optimization algorithm is a second-order optimization algorithm. With proper choice of nintial points, Newton's method will converge in very few steps. Our package implemented this feature.

Function call: 

	AutoDiff.NewtonOpt.NewtonOpt(func, init, precision = 0.00001, max_iters = 10000, message=True)

	INPUT
	=========
	f       : function
	            the objective function that returns AutmoDiff.Variable class;
	            has 1 argument -- a list of numbers that correspond to values at which
	            variable are evaluated

	init     : dictionary
			    variable string key:int/float value pair that represents the variables
			    at which they are evaluated.
			    assumes length matches the number of unique variables in f
                
	precision  : int or float
	            numerical threshold at which to stop the descent (default 0.00001)
                
	max_iters  : int
	            maximum amount of iterations of descent 
	            if precision not met (default 10000)
                
	message   : Boolean
	            Prints summary of number of iterations and the point at which there is local
	            optimum (default True)   
```python
>>> import pytest
>>> import numpy as np
>>> import AutoDiff.AutoDiff as ad 
>>> from AutoDiff.NewtonOpt 
>>> import  NewtonOpt 
>>> def f(value):
... """
...  f takes in a dictionary of independent variables
... """
...     X=ad.Variable(value['x'],name='x')
>>>     Y=ad.Variable(value['y'],name='y')
...     Z1 = ad.exp(-X**2 - Y**2)
...     Z2 = ad.exp(-(X - 1)**2 - (Y - 1)**2)
...     return (Z1 - Z2) * 2
>>> initp ={'x':.8,'y':1.4}
>>> result= NewtonOpt(f,initp,message=False)
>>> print(result)
{'point': {'x': 1.099839320128867, 'y': 1.099839320128867}, 'iters': 5}
```

### Newton solver
In engineering as well as science, we always need to solve sets of nonlinear equations, i.e., find the solution of 

$$\mathbf{F}(\mathbf{X})=\mathbf{0}$$

where $\mathbf{X}$ and $\mathbf{F}$ are high dimensional vectors. Typically Newton's method is used to iteratively solve this equation,

$$\mathbf{X_{n+1}} = \mathbf{X_{n}} - \left(\mathbf{F'}(\mathbf{X_n})\right)^{-1} \mathbf{F}(\mathbf{X_n})$$

where $\mathbf{F'}(\mathbf{X_n}$ is the Jacobian of $\mathbf{F}$. The iteration ends when $\|\mathbf{F}(\mathbf{X_n})\|$ is sufficiently closed to zero.

The accuracy and robustness of Newton's method highly depend on the tolerence of the calculated Jocobian, which can be calculated using our AD package. As for implementation, we define function $\mathbf{F_0}$ with some initial guess of solution $\mathbf{X_0}$, then use Newton's method to find $\mathbf{X_1}$, then update the function to $\mathbf{F_1}$, then move on and on until $\mathbf{F_n}(\mathbf{X_n})$ is sufficiently closed to zero. 

### Gradient Descent

The method of gradient descent to iteratively find the local minima of a function uses the first derivative of the function evaluated at some initial starting point and updating this point by subtracting some proportion of the derivative. Gradient descent continues to find the derivative at the updated points until the difference in the updated point and the previous point becomes negligible or less than some precision value. Our Automatic Differentiation module comes in handy because, given a function created by Variable instances, we can automatically calculate the derivative by passing a point into this function, which is also a Variable instance.

This feature can be used in the following way:
```python
>>> import AutoDiff.AutoDiff as ad
>>> import AutoDiff.GradDesc as gd 
>>> import numpy as np
>>> # define objective function 
>>> def f(values):
    ... x1 = ad.Variable(values['x1'], name='x1')
    ... x2 = ad.Variable(values['x2'], name='x2')
    ... f = 2 * (x1 ** 2) + ad.sin(x2)
    ... return f
>>> # define initial starting point that matches dictionary keys of objective function
>>> x = {'x1':-23, 'x2':23}
>>> # store gradient descent results
>>> grad_desc_f = gd.grad_desc(f = f, init = x, gamma = 0.001, message = False)
>>> # can call dictionary keys for number of iterations and point at which gradient descent stops
>>> print (grad_desc_f['iters'])
>>> print (grad_desc_f['point'])
4055
{'x1': array([-2.01069012e-06]), 'x2': array([23.55195857])}
```

### GMRES - Generalized Minimal Residual Method
GMRES is an iterative method to solve system of linear equations which approximates the solution by the vector in a Krylov subspace with minimal residual. There are two ways to solve a linear system $Ax = b$ using `scipy.sparse.linalg.gmres`. One way is to pass in the matrix A and vector b directly to the gmres function as `scipy.sparse.linalg.gmres(A, b)`. The other way is to pass in an action and the vector b to the gmres function as `scipy.sparse.linalg.gmres(action, b)` where action is a LinearOperator with a user-defined 'action' passed in argument to the `matvec` parameter. This is possible because GMRES only requires the matrix-vector product for it to work, instead of the entire matrix A. This property enables us to use automatic differentiation to churn out the 'action' to pass in to the gmres function, since the result of automatic differentiation is already the matrix-vector product we want for GMRES!

This feature can be used in the following way: 
```python
>>> import AutoDiff.AutoDiff as ad
>>> from AutoDiff.GMRes import gmres_autodiff
>>> b = [1, 2, 3]
>>> x1 = ad.Variable(1, name='x1')
>>> x2 = ad.Variable(1, name='x2')
>>> x3 = ad.Variable(1, name='x3')
>>> f1 = 2*x1+3*x2+2*x3
>>> f2 = 3*x1+2*x2+1*x3
>>> f3 = 3*x1+3*x2+3*x3
>>> F = [f1, f2, f3]
>>> x = gmres_autodiff(F, b)
array([ 1., -1.,  1.])
```

In this way, the user never even needs to start a full matrix A in memory!

## Future Work

We hope that you find our package useful! Some additional features we wish to add in the future are:

1. Back propagation in neural network
2. Additional ptimization methods with AD to machine learning
3. Applications to high-accuracy scientific & engineering simulations, e.g., turbulence in fluid dynamics