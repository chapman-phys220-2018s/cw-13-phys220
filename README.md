# PHYS220 CW 13 

**Author(s):** **Daniel Chang**

[![Build Status](https://travis-ci.com/chapman-phys220-2018s/cw-13-phys220.svg?branch=master)](https://travis-ci.com/chapman-phys220-2018s/cw-13-phys220)

## Specification

In this assignment, we will explore the exciting new development in python of [```numba```](http://numba.pydata.org/), which is a just-in-time (JIT) compiler optimized for python's ```numpy``` library. In short, ```numba``` interprets python code and converts it into machine code before executing it. This means that code that is run repeatedly (such as vectorized code) will receive a dramatic speedup over raw python code. 

* Write a notebook ```cw13-juliasets.ipynb``` that includes the following bit of code.

```python
import numpy as np
#import numba as nb  #uncomment for numba
import matplotlib.pyplot as plt
%matplotlib inline

def julia(c):
    @np.vectorize  #comment for numba
    #@nb.vectorize #uncomment for numba
    def j(z):
        for n in range(100):
            z = z**2 + c
            if abs(z) > 2:
                return n
        return 0
    return j

j = julia(0.345 + 0.45j)

#@nb.jit  #uncomment for numba
def cplane(min=-1.5, max=1.5, points=10000):
    r = np.linspace(-1.5, 1.5, 10000)
    x, y = np.meshgrid(r,r)
    z = x + y * 1j
    return z

%time z = cplane()
%time jset = j(z)

plt.figure(1, (20,15))
plt.imshow(jset, cmap=plt.cm.bone)
plt.xticks([])
plt.yticks([])
plt.title("Julia Set : c = 0.345 + 0.45j")
plt.show()
```

As written, this will use ```numpy``` in a vectorized way as we have been using in class so far. Run the code and note how long the timed portions take. Then uncomment the ```@nb``` lines to enable ```numba```. Rerun the code and compare the results of the timings. (The ```@nb``` lines are called "decorators" and modify the subsequent function definitions to add functionality, in this case just-in-time compilation via ```numba```.)

* Copy your CW12 into this repository and modify your code to use ```numba```. Are you able to speed up the run time of your Runge-Kutta integration? Speculate why or why not.

## Assessment

Analyze in this section what you found useful about this assignment in your own words. Include any lingering questions or comments that you may have.

**Cool tool to learn at this point because a lot of the code we are running can take quite awhile to work now. I am having some issues with it working for my final, but this is a neat tool. -Daniel**

## Honor Pledge

I pledge that all the work in this repository is my own with only the following exceptions:

* Content of starter files supplied by the instructor;
* Code borrowed from another source, documented with correct attribution in the code and summarized here.

Signed,

**Daniel Chang**
