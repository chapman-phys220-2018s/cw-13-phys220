#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
#Daniel Chang
#2260161
#Chang254@mail.chapman.edu
#Myranda Hoggatt
#2285495
#hogga102@mail.chapman.edu
#Devon Ball
#2313078
#dball@chapman.edu
#PHYS220 Spring 2018
#CW12
###

import numpy as np
import matplotlib.pyplot as plt
import numba as nb

@nb.jit
def runge4(x0,y0,F,N):
    dt = .001
    t = 0
    x = np.zeros(N)
    x[0] = x0
    y = np.zeros(N)
    y[0] = y0
    for i in range(1,N):
        t += dt
        tmid = t - (dt/2)
        tforw = t + dt
        xk1 = dt*(y[i-1])
        yk1 = dt*(-.25*y[i-1]+x[i-1]-(x[i-1]**3)+F*np.cos(t-dt))
        xmid = x[i-1] + xk1/2
        ymid = y[i-1] + yk1/2
        xk2 = dt*(ymid)
        yk2 = dt*(-.25*ymid+xmid-(xmid**3)+F*np.cos(tmid))
        xmid = x[i-1] + xk2/2
        ymid = y[i-1] + yk2/2
        xk3 = dt*(ymid)
        yk3 = dt*(-.25*ymid+xmid-(xmid**3)+F*np.cos(tmid))
        xforw = x[i-1] + xk3
        yforw = y[i-1] + yk3
        xk4 = dt*(yforw)
        yk4 = dt*(-.25*yforw+xforw-(xforw**3)+F*np.cos(t))
        x[i] = x[i-1] + (xk1 + 2*xk2 + 2*xk3 +xk4)/6
        y[i] = y[i-1] + (yk1 + 2*yk2 + 2*yk3 +yk4)/6
    return (x,y)

def plot_sombrero():
    plt.figure(figsize=(10,10))
    plt.title("Sombrero Plot")
    plt.xlabel("x")
    plt.ylabel("v")
    x = np.linspace(-1.5,1.5,10000)
    v = (x**4/2)-(x**2/2)
    plt.plot(x,v)

def plot_x(x0,y0,F,N):
    plt.figure(figsize=(10,10))
    plt.title("x(t) vs t")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    x = runge4(x0,y0,F,N)[0]
    t = np.linspace(0,.001*N,N)
    plt.plot(t,x)

def plot_parametric(x0,y0,F,N):
    plt.figure(figsize=(10,10))
    plt.title("y(t) vs x(t)")
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    xfull = runge4(x0,y0,F,N*6283)[0]
    yfull = runge4(x0,y0,F,N*6283)[1]
    t = np.linspace(0,N*2*np.pi,N+1)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    x[0] = xfull[0]
    y[0] = yfull[0]
    for i in range(1,N):
        x[i] = xfull[i*6283]
        y[i] = yfull[i*6283]
    plt.scatter(x,y)

    