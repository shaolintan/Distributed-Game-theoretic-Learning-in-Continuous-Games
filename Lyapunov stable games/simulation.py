# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 15:16:22 2022

@author: sean
"""

import numpy as np
import networkx as nx
import random as rd
import matplotlib.pyplot as plt

def Nash_error_1(x,z):
    N_0=len(z)
    err=[]
    for i in range(N_0):
        temp=np.linalg.norm(x-z[i])
        err.append(temp)
    return err

def Nash_error_2(x,z):
    N_0=np.size(x)
    N_1=np.size(z,0)
    err=[]
    for i in range(N_1):
        z1=[z[i][j,j] for j in range(N_0)]
        temp=np.linalg.norm(x-z1)
        err.append(temp)
    return err

def proj_gradient(x,r,iterations):
    x_list = [x]
    for iter_ in range(iterations):
        x_t = x_list[-1]
        y=[0,0,0,0,0]
        y[0]=x_t[0]-r*(6*(x_t[0]-1)+2*(x_t[1]-2)+2*(x_t[2]-3)+8*(x_t[3]-4)+3*(x_t[4]-5))
        y[1]=x_t[1]-r*(2*(x_t[0]-1)+6*(x_t[1]-2)+1*(x_t[2]-3)+3*(x_t[3]-4)+2*(x_t[4]-5))
        y[2]=x_t[2]-r*(3*(x_t[0]-1)+1*(x_t[1]-2)+6*(x_t[2]-3)+1*(x_t[3]-4)+1*(x_t[4]-5))
        y[3]=x_t[3]-r*(4*(x_t[0]-1)+3*(x_t[1]-2)+1*(x_t[2]-3)+6*(x_t[3]-4)+1*(x_t[4]-5))
        y[4]=x_t[4]-r*(1*(x_t[0]-1)+3*(x_t[1]-2)+1*(x_t[2]-3)+4*(x_t[3]-4)+6*(x_t[4]-5))
        x_list.append(y)
    return x_list

x=[0,0,0,0,0]
x1=[1,2,3,4,5]
r=0.1
iterations1=1000

y_1=proj_gradient(x,r,iterations1)
y_1=np.array(y_1)
e_1=Nash_error_1(x1,y_1)
plt.figure(1)
for i in range(5):
    plt.plot(y_1[:,i])


def proj_gradient_2(x,iterations):
    x_list = [x]
    for iter_ in range(iterations):
        x_t = x_list[-1]
        y=[0,0,0,0,0]
        y[0]=x_t[0]-0.1/np.log(iter_+2)*(6*(x_t[0]-1)+2*(x_t[1]-2)+2*(x_t[2]-3)+8*(x_t[3]-4)+3*(x_t[4]-5))
        y[1]=x_t[1]-0.1/np.log(iter_+2)*(2*(x_t[0]-1)+6*(x_t[1]-2)+1*(x_t[2]-3)+3*(x_t[3]-4)+2*(x_t[4]-5))
        y[2]=x_t[2]-0.1/np.log(iter_+2)*(3*(x_t[0]-1)+1*(x_t[1]-2)+6*(x_t[2]-3)+1*(x_t[3]-4)+1*(x_t[4]-5))
        y[3]=x_t[3]-0.1/np.log(iter_+2)*(4*(x_t[0]-1)+3*(x_t[1]-2)+1*(x_t[2]-3)+6*(x_t[3]-4)+1*(x_t[4]-5))
        y[4]=x_t[4]-0.1/np.log(iter_+2)*(1*(x_t[0]-1)+3*(x_t[1]-2)+1*(x_t[2]-3)+4*(x_t[3]-4)+6*(x_t[4]-5))
        x_list.append(y)
    return x_list

iterations2=4000
y_2=proj_gradient_2(x,iterations2)
y_2=np.array(y_2)
e_2=Nash_error_1(x1,y_2)
plt.figure(2)
for i in range(5):
    plt.plot(y_2[:,i])
    
def initialization(N,g,r):
    ### W ###
    A = nx.adjacency_matrix(g).todense().A
    L = np.diag(np.sum(A,0))-A
    eig_val,_ = np.linalg.eig(L)
    if r<=0.5*np.max(eig_val):
        print('r <= 0.5*lambd_max')
    W = np.eye(N)-(L)/r   
    return W
#    
def dis_proj_gradient(Z,r,W,iterations):
    z_list = [Z.copy()]
    for iter_ in range(iterations):
        z_t = z_list[-1]
        z=np.zeros((5,5))
        t0=6*(z_t[0,0]-1)+2*(z_t[0,1]-2)+2*(z_t[0,2]-3)+8*(z_t[0,3]-4)+3*(z_t[0,4]-5)
        z[0,:]=sum([W[0,j]*z_t[j,:] for j in range(5)])-r*np.array([t0,0,0,0,0])
        t1=2*(z_t[1,0]-1)+6*(z_t[1,1]-2)+1*(z_t[1,2]-3)+3*(z_t[1,3]-4)+2*(z_t[1,4]-5)
        z[1,:]=sum([W[1,j]*z_t[j,:] for j in range(5)])-r*np.array([0,t1,0,0,0])
        t2=3*(z_t[2,0]-1)+1*(z_t[2,1]-2)+6*(z_t[2,2]-3)+1*(z_t[2,3]-4)+1*(z_t[2,4]-5)
        z[2,:]=sum([W[2,j]*z_t[j,:] for j in range(5)])-r*np.array([0,0,t2,0,0])
        t3=4*(z_t[3,0]-1)+3*(z_t[3,1]-2)+1*(z_t[3,2]-3)+6*(z_t[3,3]-4)+1*(z_t[3,4]-5)
        z[3,:]=sum([W[3,j]*z_t[j,:] for j in range(5)])-r*np.array([0,0,0,t3,0])
        t4=1*(z_t[4,0]-1)+3*(z_t[4,1]-2)+1*(z_t[4,2]-3)+4*(z_t[4,3]-4)+6*(z_t[4,4]-5)
        z[4,:]=sum([W[4,j]*z_t[j,:] for j in range(5)])-r*np.array([0,0,0,0,t4])       
        z_tplus = np.vstack(z)
        z_list.append(z_tplus)
    return z_list

g=nx.cycle_graph(5)
W=initialization(5,g,4)
Z=np.zeros((5,5))
r=0.1
iterations3=4000
z1=dis_proj_gradient(Z,r,W,iterations3)
z1=np.array(z1)
e_3=Nash_error_1(x1,z1)

plt.figure(3)
for i in range(5):
    for j in range(5):
        plt.plot(z1[:,i,j])


W2=np.zeros((5,5))
for i in range(5):
    W2[i,i]=0.5
W2[0,1]=W2[1,2]=W2[2,3]=W2[3,4]=W2[4,0]=0.5

z2=dis_proj_gradient(Z,r,W2,iterations3)
z2=np.array(z2)
e_4=Nash_error_1(x1,z2)

plt.figure(4)
for i in range(5):
    for j in range(5):
        plt.plot(z2[:,i,j])
        

plt.figure(5)
plt.loglog(range(1,iterations1+2),e_1/e_1[0],'r-')
plt.loglog(range(1,iterations2+2),e_2/e_2[0],'b-')
plt.loglog(range(1,iterations3+2),e_3/e_3[0],'k-')
plt.loglog(range(1,iterations3+2),e_4/e_4[0],'c-')


