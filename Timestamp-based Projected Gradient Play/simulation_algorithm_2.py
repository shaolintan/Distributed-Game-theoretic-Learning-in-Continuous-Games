# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 10:59:56 2022

@author: sean
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:38:19 2022

@author: sean
"""

import numpy as np
import networkx as nx
import random as rd
import matplotlib.pyplot as plt
#import algorithm0426 as al

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

def step_fun(t,k):
    if k==0:
        out=1/t
    elif k<0:
        out=1/np.log(t+1)
    else:
        out=k
    return out




#input initial state z(0), initial timestamp tao(0), stepsize lambda, graph g, 
#gradient information grident_fun,proj_fun,iteration time T
    
def baseline(N,x,lambd,T):
    x_list = [x]
    for iter_ in range(T):
        x_tplus = []
        x_t = x_list[-1]
        for i in range(N):
            x_tplus.append(proj_fun(x_t[i]-lambd[i]*gradient_fun(x_t,i)) )
        x_list.append(x_tplus)
    return x_list

def baseline_2(N,x,T,k):
    x_list = [x]
    for iter_ in range(T):
        x_tplus = []
        x_t = x_list[-1]
        for i in range(N):
            x_tplus.append(proj_fun(x_t[i]-step_fun(iter_,k)*gradient_fun(x_t,i)) )
        x_list.append(x_tplus)
    return x_list
#
def timestamp_gradient(z_0,tao_0,g,T,k):
    z_list = [z_0.copy()]
#   tao_list=[tao_0.copy()]
    t=0
    N=nx.number_of_nodes(g)
    nodes=list(nx.nodes(g))   
    tao=tao_0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        mid_z=z_list[-1].copy()
        mid_t=tao.copy()
        for i in range(N):
            mid_z[i,i]=proj_fun(z_a[i,i]-step_fun(t,k)*gradient_fun(z_a[i,:],i))
            mid_t[i,i]=tao[i,i]+1
            for j in range(N):
                if j!=i:
                    neighbors = list(nx.neighbors(g,nodes[i]))
                    mid_t[i][j] = max([tao[nei][j] for nei in neighbors])
                    nei_tao_max = [nei for nei in neighbors if tao[nei][j]==mid_t[i][j]]
                    mid_z[i,j]=z_a[rd.choice(nei_tao_max),j]
        z_list.append(mid_z)
        tao=mid_t
    return z_list

def switching_timestamp_gradient(z_0,tao_0,g,T,k):
    z_list = [z_0.copy()]
#   tao_list=[tao_0.copy()]
    t=0
    g1=g[0]
    g2=g[1]
    N=nx.number_of_nodes(g1)
    nodes=list(nx.nodes(g1)) 
    tao=tao_0
    while t<T:
        t=t+1
        z_a = z_list[-1]
        mid_z=z_list[-1].copy()
        mid_t=tao.copy()
        for i in range(N):
            mid_z[i,i]=proj_fun(z_a[i,i]-step_fun(t,k)*gradient_fun(z_a[i,:],i))
            mid_t[i,i]=tao[i,i]+1
            for j in range(N):
                if j!=i:
                    if t%2==0:
                        neighbors = list(nx.neighbors(g1,nodes[i]))
                    else:
                        neighbors = list(nx.neighbors(g2,nodes[i]))
                    if neighbors:
                        mid_t[i][j] = max([tao[nei][j] for nei in neighbors])
                        nei_tao_max = [nei for nei in neighbors if tao[nei][j]==mid_t[i][j]]
                        mid_z[i,j]=z_a[rd.choice(nei_tao_max),j]
        z_list.append(mid_z)
        tao=mid_t
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

def Di_cycle(n):
    g=nx.DiGraph()
    for i in range(n-1):
        g.add_edge(i,i+1)
    g.add_edge(n-1,0)
    return g

def Switch_Di_cycle(n):
    g1=nx.DiGraph()
    g2=nx.DiGraph()
    for i in range(n-1):
        if i%2==0:
            g1.add_edge(i,i+1)
        else:
            g2.add_edge(i,i+1)
    if n%2==0:
        g2.add_edge(n-1,0)
    else:
        g1.add_edge(n-1,0)
    g=[g1,g2]
    return g


N = 8
z_0 = np.random.random((N,N))*20
tao_0=np.eye((N))
g=nx.cycle_graph(8)
g1=Di_cycle(8)
g2=Switch_Di_cycle(8)

T=3000
lambd = [0.001]*8
x_0 = [rd.random()*8 for i in range(N)]

x_list = baseline(N,x_0,lambd,T)
x_list=np.array(x_list)
plt.figure(1)
for i in range(8):
    plt.plot(x_list[:,i])

z_list=timestamp_gradient(z_0,tao_0,g,T,k=0)
N_1=np.size(z_list,0)
bar_x=[]
for i in range(N_1):
    bar_x.append([z_list[i][j,j] for j in range(8)])
bar_x=np.array(bar_x)
    
plt.figure(2)
for i in range(8):
    plt.plot(bar_x[:,i])

e_0=Nash_error(x_list[-1],z_list)
plt.figure(3)
plt.loglog(range(1,T+2),e_0/e_0[0],'r--')


z_list_1=timestamp_gradient(z_0,tao_0,g1,T,k=0)
e_1=Nash_error(x_list[-1],z_list_1)
plt.loglog(range(1,T+2),e_1/e_1[0],'b--')

z_list_2=switching_timestamp_gradient(z_0,tao_0,g2,T,k=0)
e_2=Nash_error(x_list[-1],z_list_2)
plt.loglog(range(1,T+2),e_2/e_2[0],'c--')

z_list_3=timestamp_gradient(z_0,tao_0,g,T,k=0.001)

z_list_4=timestamp_gradient(z_0,tao_0,g1,T,k=0.001)

z_list_5=switching_timestamp_gradient(z_0,tao_0,g2,T,k=0.001)

e_3=Nash_error(x_list[-1],z_list_3)
e_4=Nash_error(x_list[-1],z_list_4)
e_5=Nash_error(x_list[-1],z_list_5)
plt.loglog(range(1,T+2),e_3/e_3[0],'r-')
plt.loglog(range(1,T+2),e_4/e_4[0],'b-')
plt.loglog(range(1,T+2),e_5/e_5[0],'c-')

#z_list_6=timestamp_gradient(z_0,tao_0,g,T,k=-1)
#
#z_list_7=timestamp_gradient(z_0,tao_0,g1,T,k=-1)
#
#z_list_8=switching_timestamp_gradient(z_0,tao_0,g2,T,k=-1)
#
#e_6=Nash_error(x_list[-1],z_list_6)
#e_7=Nash_error(x_list[-1],z_list_7)
#e_8=Nash_error(x_list[-1],z_list_8)
#plt.loglog(range(1,T+2),e_6/e_6[0],'r:')
#plt.loglog(range(1,T+2),e_7/e_7[0],'b:')
#plt.loglog(range(1,T+2),e_8/e_8[0],'c:')