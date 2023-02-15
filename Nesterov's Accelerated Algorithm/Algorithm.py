# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:23:38 2022

@author: sean
"""


import numpy as np
import networkx as nx
import random as rd
import matplotlib.pyplot as plt
#import algorithm0426 as al

#def gradient_fun(x,i):
#    out = 20+10*(i-1)-(2200-sum(x)-x[i])
#    return out
#
#
#def proj_fun(x):
#    if x<0:
#        x = 0
#    elif x>200:
#        x = 200
#    else:
#        pass
#    return x

#def gradient_fun(x,i):
#    out = 10+4*(i-1)-(600-sum([np.square(x[i]) for i in range(N)])-2*np.square(x[i]))
#    return out
#
#
#def proj_fun(x):
#    if x<0:
#        x = 0
#    elif x>20:
#        x = 20
#    else:
#        pass
#    return x

def gradient_fun(x,i):
    if i==0:
        mid1=x[i]*x[i]+x[i]*x[i+1]
        mid2=4*x[i]-2*x[i+1]-1
    elif i==499:
        mid1=x[i]*x[i]+x[i-1]*x[i-1]+x[i-1]*x[i]
        mid2=4*x[i]+x[i-1]-1
    else:
        mid1=x[i]*x[i]+x[i-1]*x[i-1]+x[i-1]*x[i]+x[i]*x[i+1]
        mid2=4*x[i]+x[i-1]-2*x[i+1]-1
    out=mid1+mid2
    return out


def proj_fun(x):
    if x<0:
        x = 0
    else:
        pass
    return x

def gradient_play(N,x,lambd,T):
    x_list = [x]
    for iter_ in range(T):
        x_tplus = []
        x_t = x_list[-1]
        for i in range(N):
            x_tplus.append(proj_fun(x_t[i]-lambd*gradient_fun(x_t,i)))
        x_list.append(x_tplus)
    return x_list


def extra_gradient_play(N,x,lambd,T):
    x_list = [x]
    for iter_ in range(T):
        x_tplus = []
        x_t = x_list[-1]
        y=[]
        for i in range(N):
            y.append(x_t[i]-lambd*gradient_fun(x_t,i))
        for i in range(N):
            x_tplus.append(proj_fun(x_t[i]-lambd*gradient_fun(y,i)))
        x_list.append(x_tplus)
    return x_list

def Neterov_gradient_play(N,x,lambd,beta,T):
    x_list = [x]
    x_tplus = []
    x_t = x_list[-1]
    for i in range(N):
        x_tplus.append(proj_fun(x_t[i]-lambd*gradient_fun(x_t,i)))
    x_list.append(x_tplus)
    for iter_ in range(T):
        x_tplus = []
        x_t = x_list[-1]
        x_t2=x_list[-2]
        y=[]
        for i in range(N):
            y.append(x_t[i]+beta*(x_t[i]-x_t2[i]))
        for i in range(N):
            x_tplus.append(proj_fun(y[i]-lambd*gradient_fun(y,i)))
        x_list.append(x_tplus)
    return x_list
    

def Reflected_gradient_play(N,x,lambd,T):
    x_list = [x]
    x_tplus = []
    x_t = x_list[-1]
    for i in range(N):
        x_tplus.append(proj_fun(x_t[i]-lambd*gradient_fun(x_t,i)))
    x_list.append(x_tplus)
    for iter_ in range(T):
        x_tplus = []
        x_t = x_list[-1]
        x_t2=x_list[-2]
        y=[]
        for i in range(N):
            y.append(x_t[i]+(x_t[i]-x_t2[i]))
        for i in range(N):
            x_tplus.append(proj_fun(x_t[i]-lambd*gradient_fun(y,i)))
        x_list.append(x_tplus)
    return x_list

def Golden_ratio_play(N,x,lambd,T):
    x_list = [x]
    y_list=[x]
    x_t = x_list[-1]
    for iter_ in range(T):
        x_tplus = []
        x_t = x_list[-1]
        y_t = y_list[-1]
        y=[]
        for i in range(N):
            y.append((1-(np.sqrt(5)-1)/2)*x_t[i]+(np.sqrt(5)-1)/2*y_t[i])
        for i in range(N):
            x_tplus.append(proj_fun(y[i]-lambd*gradient_fun(x_t,i)))
        y_list.append(y)
        x_list.append(x_tplus)
    return x_list

def gradient_error(x,N):
    err=[]
    N_0=len(x)
    for i in range(N_0):
        temp=0
        for j in range(N):
            temp+=gradient_fun(x[i],j)*gradient_fun(x[i],j)
        err.append(temp)
    return err

def gradient_error_2(x,lambd,N):
    err=[]
    N_0=len(x)
    for i in range(N_0):
        temp=0
        for j in range(N):
            temp+=(x[i][j]-proj_fun(x[i][j]-lambd*gradient_fun(x[i],j)))*(x[i][j]-proj_fun(x[i][j]-lambd*gradient_fun(x[i],j)))
        err.append(temp)
    return err

N=500
x_0 = np.zeros((500,1))

lambd=0.005
beta=0.7

T=10000
x_list_1 = gradient_play(N,x_0,lambd,T)
x_list_1=np.array(x_list_1)
e_10=gradient_error_2(x_list_1,lambd,N)

#
##
x_list_2 = Neterov_gradient_play(N,x_0,lambd,beta,T)
x_list_2=np.array(x_list_2)
e_20=gradient_error_2(x_list_2,lambd,N)
#
##
x_list_3 = Reflected_gradient_play(N,x_0,lambd,T)
x_list_3=np.array(x_list_3)
e_30=gradient_error_2(x_list_3,lambd,N)


x_list_4 = Golden_ratio_play(N,x_0,lambd,T)
x_list_4=np.array(x_list_4)
e_40=gradient_error_2(x_list_4,lambd,N)
#
beta=0.5
x_list_5 = Neterov_gradient_play(N,x_0,lambd,beta,T)
x_list_5=np.array(x_list_5)
e_20_1=gradient_error_2(x_list_5,lambd,N)
#
beta=0.3
x_list_6 = Neterov_gradient_play(N,x_0,lambd,beta,T)
x_list_6=np.array(x_list_6)
e_20_2=gradient_error_2(x_list_6,lambd,N)
#
plt.figure(2)
##
plt.loglog(range(1,len(e_10)+1),e_10,'r--')
plt.loglog(range(1,len(e_20)+1),e_20,'b--')
plt.loglog(range(1,len(e_30)+1),e_30,'g--')
plt.loglog(range(1,len(e_40)+1),e_40,'c--')
plt.loglog(range(1,len(e_20_1)+1),e_20_1,'k--')
plt.loglog(range(1,len(e_20_2)+1),e_20_2,'m--')

#plt.figure(4)
#for i in range(N):
#    plt.plot(x_list_4[:,i])
    