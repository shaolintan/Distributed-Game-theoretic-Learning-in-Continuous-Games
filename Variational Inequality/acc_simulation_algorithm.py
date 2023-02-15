# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:11:27 2022

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

def gradient_fun(x,i):
    out = 10+4*(i-1)-(600-sum([np.square(x[i]) for i in range(N)])-np.square(x[i]))
    return out


def proj_fun(x):
    if x<0:
        x = 0
    elif x>20:
        x = 20
    else:
        pass
    return x

def Di_cycle(n):
    g=nx.DiGraph()
    for i in range(n-1):
        g.add_edge(i,i+1)
    g.add_edge(n-1,0)
    return g

def graph_to_mixing_matrix(g,r):
    ### W ###
    A = nx.adjacency_matrix(g).todense().A
    L = np.diag(np.sum(A,0))-A
    eig_val,_ = np.linalg.eig(L)
    N=nx.number_of_nodes(g)
#    if r<=0.5*np.max(eig_val):
#        print('r <= 0.5*lambd_max')
    W = np.eye(N)-(L)/r   
    return W

def graph_to_mixing_matrix2(g):
    ### W ###
    A = nx.adjacency_matrix(g).todense().A
    b=np.sum(A,axis=0)
    N=nx.number_of_nodes(g)
    W=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if A[i,j]!=0:
                W[i,j]=A[i,j]/max(b[i]+1,b[j]+1)
    for i in range(N):
        W[i,i]=1-np.sum(W[:,i])
    return W


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

def Neterov_gradient_play(N,x,lambd,T):
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
            y.append(x_t[i]+lambd*(x_t[i]-x_t2[i]))
        for i in range(N):
            x_tplus.append(proj_fun(y[i]-lambd*gradient_fun(y,i)))
        x_list.append(x_tplus)
    return x_list



def distributed_gradient_play_1(Z,N,lambd,W,T):
    z_list = [Z.copy()]
    t=0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i]=proj_fun(sum([W[i,k]*z_a[k,i] for k in range(N)])-lambd*gradient_fun(z_a[i,:],i))
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
        z_list.append(mid_z)
    return z_list

def augmented_game_approach(Z,N,thet,lambd,W,T):
    z_list = [Z.copy()]
    t=0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i] = proj_fun(z_a[i,i]-lambd*(z_a[i,i]-sum([W[i,k]*z_a[k,i] for k in range(N)])+thet*gradient_fun(z_a[i,:],i)))
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=z_a[i,j]-lambd*(z_a[i,j]-sum([W[i,k]*z_a[k,j] for k in range(N)]))
        z_list.append(mid_z)
    return z_list

#z[i,i]=proj_gradient(z[i,:])$ 采用
def distributed_gradient_play_2(Z,N,lambd,W,T):
    z_list = [Z.copy()]
    t=0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i]=proj_fun(z_a[i,i]-lambd*gradient_fun(z_a[i,:],i))
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
        z_list.append(mid_z)
    return z_list

#z[i,j]=z[j,j] if (i,j)=1$
def distributed_gradient_play_3(Z,N,lambd,W,T):
    z_list = [Z.copy()]
    t=0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i]=proj_fun(z_a[i,i]-lambd*gradient_fun(z_a[i,:],i))
            for j in range(N):
                if j!=i:
                    if W[i,j]>0:
                        mid_z[i,j]=z_a[j,j]
                    else:
                        mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
        z_list.append(mid_z)
    return z_list

#z[i,i]=proj_gradient(z[i,:])$
def distributed_gradient_play_4(Z,N,lambd,W,T,d):
    z_list = [Z.copy()]
    t=0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        mid_z=z_list[-1].copy()
        for i in range(N):
            d1=0
            while d1<=d:
                d1=d1+1
                mid_z[i,i]=proj_fun(mid_z[i,i]-lambd*gradient_fun(mid_z[i,:],i))                
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
        z_list.append(mid_z)
    return z_list

def distributed_gradient_play_5(Z,N,lambd,W,beta,T):
    z_list = [Z.copy()]
    z_a = z_list[-1]
    mid_z=z_list[-1].copy()
    for i in range(N):
        for j in range(N):
            mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
    z_list.append(mid_z)
    t=0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        z_a2=z_list[-2]
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i]=proj_fun(z_a[i,i]-lambd*gradient_fun(z_a[i,:],i))
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=sum([W[i,k]*((1+beta)*z_a[k,j]-beta*z_a2[k,j]) for k in range(N)])
        z_list.append(mid_z)
    return z_list

def distributed_gradient_play_6(Z,N,lambd,W,beta,T):
    z_list = [Z.copy()]
    z_a = z_list[-1]
    mid_z=z_list[-1].copy()
    for i in range(N):
        for j in range(N):
            mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
    z_list.append(mid_z)
    t=0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        z_a2=z_list[-2]
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i]=proj_fun(sum([W[i,k]*((1+beta)*z_a[k,i]-beta*z_a2[k,i]) for k in range(N)])-lambd*gradient_fun(z_a[i,:],i))
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=sum([W[i,k]*((1+beta)*z_a[k,j]-beta*z_a2[k,j]) for k in range(N)])
        z_list.append(mid_z)
    return z_list

#采用
def distributed_gradient_play_7(Z,N,lambd,W,beta,T):
    z_list = [Z.copy()]
    z_a = z_list[-1]
    mid_z=z_list[-1].copy()
    for i in range(N):
        for j in range(N):
            mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
    z_list.append(mid_z)
    t=0
    while t<T:
        t=t+1
        z_a = np.array(z_list[-1])
        z_a2=np.array(z_list[-2])
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i]=proj_fun((1+beta)*z_a[i,i]-beta*z_a2[i,i]-lambd*gradient_fun((1+beta)*z_a[i,:]-beta*z_a2[i,:],i))
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=sum([W[i,k]*((1+beta)*z_a[k,j]-beta*z_a2[k,j]) for k in range(N)])
        z_list.append(mid_z)
    return z_list

def distributed_gradient_play_8(Z,N,lambd,W,beta,T):
    z_list = [Z.copy()]
    z_a = z_list[-1]
    mid_z=z_list[-1].copy()
    for i in range(N):
        for j in range(N):
            mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
    z_list.append(mid_z)
    t=0
    while t<T:
        t=t+1
        z_a = np.array(z_list[-1])
        z_a2=np.array(z_list[-2])
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i]=proj_fun((1-beta)*z_a[i,i]+beta*z_a2[i,i]-lambd*gradient_fun((1+beta)*z_a[i,:]-beta*z_a2[i,:],i))
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=sum([W[i,k]*((1-beta)*z_a[k,j]+beta*z_a2[k,j]) for k in range(N)])
        z_list.append(mid_z)
    return z_list

def distributed_gradient_play_9(Z,N,lambd,W,beta,T):
    z_list = [Z.copy()]
    z_a = z_list[-1]
    mid_z=z_list[-1].copy()
    for i in range(N):
        for j in range(N):
            mid_z[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
    z_list.append(mid_z)
    t=0
    while t<T:
        t=t+1
        z_a = np.array(z_list[-1])
        z_a2=np.array(z_list[-2])
        y_a=(1+beta)*z_a-beta*z_a2
        mid_z=z_list[-1].copy()
        for i in range(N):
            mid_z[i,i]=proj_fun(z_a[i,i]-lambd*gradient_fun(y_a[i,:],i))
            for j in range(N):
                if j!=i:
                    mid_z[i,j]=z_a[i,j]-y_a[i,j]+sum([W[i,k]*(y_a[k,j]) for k in range(N)])
        z_list.append(mid_z)
    return z_list

# 与算法2收敛速度相同
def distributed_extra_gradient_play(Z,N,lambd,W,T):
    z_list = [Z.copy()]
    t=0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        mid_z_1=z_list[-1].copy()
        mid_z_2=z_list[-1].copy()
        for i in range(N):
            mid_z_1[i,i]=z_a[i,i]-lambd*gradient_fun(z_a[i,:],i)
            for j in range(N):
                if j!=i:
                    mid_z_1[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
        for i in range(N):
            mid_z_2[i,i]=proj_fun(z_a[i,i]-lambd*gradient_fun(mid_z_1[i,:],i))
            for j in range(N):
                if j!=i:
                    mid_z_2[i,j]=sum([W[i,k]*z_a[k,j] for k in range(N)])
        z_list.append(mid_z_2)
    return z_list  

def Nash_error(x,z):
    N_0=np.size(x)
    N_1=np.size(z,0)
    err=[]
    for i in range(N_1):
        z1=[z[i][j,j] for j in range(N_0)]
        temp=np.linalg.norm(x-z1)
        err.append(temp)
    return err

def Nash_error_1(x,z):
    N_0=len(z)
    err=[]
    for i in range(N_0):
        temp=np.linalg.norm(x-z[i])
        err.append(temp)
    return err

N=8
g1=nx.cycle_graph(8)
g2=Di_cycle(8)
g3=nx.path_graph(8)
g4=nx.star_graph(7)
g5=nx.wheel_graph(8)
W1=graph_to_mixing_matrix2(g1)
W2=graph_to_mixing_matrix2(g2)
W3=graph_to_mixing_matrix2(g3)
W4=graph_to_mixing_matrix2(g4)
W5=graph_to_mixing_matrix2(g5)

#W=graph_to_mixing_matrix(g,3)
#
#
#W2=graph_to_mixing_matrix(g,2)
z_0 = np.random.random((N,N))*20
x_0 = [rd.random()*20 for i in range(N)]
lambd=0.001
beta=0.5
#thet=0.2
T=10000
x_list_1 = gradient_play(N,x_0,lambd,T)
x_list_1=np.array(x_list_1)
#x_list_2 =extra_gradient_play(N,x_0,lambd,T)
#x_list_2=np.array(x_list_2)
#x_list_3=Neterov_gradient_play(N,x_0,lambd,T)
#x_list_3=np.array(x_list_3)
e_0=Nash_error_1(x_list_1[-1],x_list_1)
#e5=Nash_error_1(x_list_1[-1],x_list_2)
#e6=Nash_error_1(x_list_3[-1],x_list_3)
#z_list_1=distributed_gradient_play_1(z_0,N,lambd,W,T)
#z_list_2=distributed_gradient_play_2(z_0,N,lambd,W,T)
#
#z_list_3=augmented_game_approach(z_0,N,thet,lambd,W,T)
#z_list_4=distributed_gradient_play_4(z_0,N,lambd,W,T,d=5)

#z_list_5=distributed_gradient_play_2(z_0,N,lambd,W2,T)
#z_list_6=distributed_gradient_play_5(z_0,N,lambd,W,beta,T)
#z_list_7=distributed_gradient_play_7(z_0,N,lambd,W,beta,T)
#beta1=0.1
#z_list_9_1=distributed_gradient_play_7(z_0,N,lambd,W,beta1,T)
#beta2=0.3
#z_list_9_2=distributed_gradient_play_7(z_0,N,lambd,W,beta2,T)
#beta3=0.5
#z_list_9_3=distributed_gradient_play_7(z_0,N,lambd,W,beta3,T)
#beta4=0.7
#z_list_9_4=distributed_gradient_play_7(z_0,N,lambd,W,beta4,T)
#beta5=0.8
#z_list_9_5=distributed_gradient_play_7(z_0,N,lambd,W,beta5,T)
#e9_1=Nash_error(x_list_1[-1],z_list_9_1)
#e9_2=Nash_error(x_list_1[-1],z_list_9_2)
#e9_3=Nash_error(x_list_1[-1],z_list_9_3)
#e9_4=Nash_error(x_list_1[-1],z_list_9_4)
#e9_5=Nash_error(x_list_1[-1],z_list_9_5)

#z_list_6_1=distributed_gradient_play_7(z_0,N,lambd,W1,beta,T)
#z_list_6_2=distributed_gradient_play_7(z_0,N,lambd,W2,beta,T)
#z_list_6_3=distributed_gradient_play_7(z_0,N,lambd,W3,beta,T)
#z_list_6_4=distributed_gradient_play_7(z_0,N,lambd,W4,beta,T)
#z_list_6_5=distributed_gradient_play_7(z_0,N,lambd,W5,beta,T)
#e6_1=Nash_error(x_list_1[-1],z_list_6_1)
#e6_2=Nash_error(x_list_1[-1],z_list_6_2)
#e6_3=Nash_error(x_list_1[-1],z_list_6_3)
#e6_4=Nash_error(x_list_1[-1],z_list_6_4)
#e6_5=Nash_error(x_list_1[-1],z_list_6_5)
#
#plt.loglog(range(1,T+3),e6_1/e6_1[0],'k--')
#plt.loglog(range(1,T+3),e6_2/e6_2[0],'r--')
#plt.loglog(range(1,T+3),e6_3/e6_3[0],'b--')
#plt.loglog(range(1,T+3),e6_4/e6_4[0],'g--')
#plt.loglog(range(1,T+3),e6_5/e6_5[0],'c--')

plt.loglog(range(1,T+2),e_0/e_0[0],'y--')

z_list_5_1=distributed_gradient_play_2(z_0,N,lambd,W1,T)
z_list_5_2=distributed_gradient_play_2(z_0,N,lambd,W2,T)
z_list_5_3=distributed_gradient_play_2(z_0,N,lambd,W3,T)
z_list_5_4=distributed_gradient_play_2(z_0,N,lambd,W4,T)
z_list_5_5=distributed_gradient_play_2(z_0,N,lambd,W5,T)
e5_1=Nash_error(x_list_1[-1],z_list_5_1)
e5_2=Nash_error(x_list_1[-1],z_list_5_2)
e5_3=Nash_error(x_list_1[-1],z_list_5_3)
e5_4=Nash_error(x_list_1[-1],z_list_5_4)
e5_5=Nash_error(x_list_1[-1],z_list_5_5)

plt.loglog(range(1,T+2),e5_1/e5_1[0],'k--')
plt.loglog(range(1,T+2),e5_2/e5_2[0],'r--')
plt.loglog(range(1,T+2),e5_3/e5_3[0],'b--')
plt.loglog(range(1,T+2),e5_4/e5_4[0],'g--')
plt.loglog(range(1,T+2),e5_5/e5_5[0],'c--')

#z_list_9=distributed_gradient_play_9(z_0,N,lambd,W,beta,T)
#e_1=Nash_error(x_list_1[-1],z_list_1)
#e_2=Nash_error(x_list_1[-1],z_list_2)
#e_3=Nash_error(x_list_1[-1],z_list_3)
#e4=Nash_error(x_list_1[-1],z_list_4)
#e5=Nash_error(x_list_1[-1],z_list_5)
#e6=Nash_error(x_list_1[-1],z_list_6)
#e_7=Nash_error(x_list_1[-1],z_list_7)
#e8=Nash_error(x_list_1[-1],z_list_8)
#e9=Nash_error(x_list_1[-1],z_list_9)
#plt.figure(1)
#plt.loglog(range(1,T+2),e_0/e_0[0],'b--')
#plt.loglog(range(1,T+2),e_1/e_1[0],'m--')
#plt.loglog(range(1,T+2),e_2/e_2[0],'r--')
#plt.loglog(range(1,T+2),e_3/e_3[0],'c--')
#plt.loglog(range(1,T+2),e4/e4[0],'k--')
#plt.loglog(range(1,T+3),e6/e6[0],'k--')
#plt.loglog(range(1,T+3),e_7/e_7[0],'k--')
#plt.loglog(range(1,T+2),e2/e2[0],'r--')
#plt.loglog(range(1,T+3),e9_1/e9_1[0],'k--')
#plt.loglog(range(1,T+3),e9_2/e9_2[0],'r--')
#plt.loglog(range(1,T+3),e9_3/e9_3[0],'b--')
#plt.loglog(range(1,T+3),e9_4/e9_4[0],'g--')
#plt.loglog(range(1,T+3),e9_5/e9_5[0],'m--')