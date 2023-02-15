# -*- coding: utf-8 -*-


import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt



def g_i(x,i):
    out = 20+10*(i-1)-(2200-sum(x)-x[i])
    return out


def func(x):
    if x<0:
        x = 0
    elif x>200:
        x = 200
    else:
        pass
    return x


def gi_bar(zi,i):
    out = np.zeros(np.size(zi))
    out[i] = g_i(zi,i)
    return out


def baseline(N,x,lambd,iterations):
    x_list = [x]
    for iter_ in range(iterations):
        x_tplus = []
        x_t = x_list[-1]
        for i in range(N):
            x_tplus.append( func(x_t[i]-lambd[i]*g_i(x_t,i)) )
        x_list.append(x_tplus)
    return x_list

def initialization(g,r):
    ### W ###
    A = nx.adjacency_matrix(g).todense().A
    L = np.diag(np.sum(A,0))-A
    eig_val,_ = np.linalg.eig(L)
    if r<=0.5*np.max(eig_val):
        print('r <= 0.5*lambd_max')
    W = np.eye(N)-(L)/r   
    return W


def initialization1(N,g,mu,l,r):
    ### W ###
    A = nx.adjacency_matrix(g).todense().A
    L = np.diag(np.sum(A,0))-A
    eig_val,_ = np.linalg.eig(L)
    if r<=0.5*np.max(eig_val):
        print('r <= 0.5*lambd_max')
    W = np.eye(N)-(L)/r
    
    ### lambd ###
    eig_val,_ = np.linalg.eig(W)
    eig_val = np.sort(eig_val)
    rho_w = eig_val[-2]
    lambd_max = (1-rho_w)*mu/(2*N*l*l)
    lambd_max_bar = random.uniform(0, lambd_max)
    while lambd_max_bar==0 or lambd_max_bar==lambd_max:
        lambd_max_bar = random.uniform(0, lambd_max)
    lambd_min = 2*N*l*l*lambd_max_bar*lambd_max_bar/((1-rho_w)*mu )
    lambd_max = lambd_max_bar
    lambd = []
    for i in range(N):
        tmp = random.uniform(lambd_min, lambd_max)
        while tmp==lambd_min or tmp==lambd_max:
            tmp = random.uniform(lambd_min, lambd_max)
        lambd.append(tmp)
    
    return W, lambd


def iter1_1(Z,N,lambd,W,iterations):
    z_list = [Z.copy()]
    for iter_ in range(iterations):
        z_tplus = []
        z_t = z_list[-1]
        for i in range(N):
            tmp = sum([W[i,j]*z_t[j,:] for j in range(N)])-lambd[i]*gi_bar(z_t[i,:],i)
            tmp[i] = func(tmp[i])
            z_tplus.append( tmp )
        z_tplus = np.vstack((z_tplus))
        z_list.append(z_tplus)
    return z_list


def iter1_2(Z,N,lambd,W,iterations):
    z_list = [Z.copy()]
    lambd_list = [lambd.copy()]
    for iter_ in range(iterations):
        z_tplus = []
        lambd_tplus = []
        z_t = z_list[-1]
        lambd = lambd_list[-1]
        for i in range(N):
            tmp = sum([W[i,j]*z_t[j,:] for j in range(N)])-lambd[i]*gi_bar(z_t[i,:],i)
            tmp[i] = func(tmp[i])
            z_tplus.append( tmp )
            
            tmp2 = sum([W[i,j]*lambd[j] for j in range(N)])
            lambd_tplus.append( tmp2 )
            
        z_tplus = np.vstack((z_tplus))
        z_list.append(z_tplus)
        lambd_list.append(lambd_tplus)
    return z_list,lambd_list


def iter1_3(Z,N,lambd,W,iterations):
    z_list = [Z.copy()]
    for iter_ in range(iterations):
        z_tplus = []
        z_t = z_list[-1]
        for i in range(N):
            tmp = sum([W[i,j]*z_t[j,:]-W[i,j]*lambd[j]*gi_bar(z_t[j,:],j) for j in range(N)])
            tmp[i] = func(tmp[i])
            z_tplus.append( tmp )            
        z_tplus = np.vstack((z_tplus))
        z_list.append(z_tplus)
    return z_list

def iter1_4(Z,N,lambd,W,iterations):
    z_list = [Z.copy()]
    for iter_ in range(iterations):
        z_tplus = []
        z_t = z_list[-1]
        for i in range(N):
            tmp = sum([W[i,j]*z_t[j,:]-W[i,j]*lambd[j]*gi_bar(z_t[j,:],j) for j in range(N)])-lambd[i]*gi_bar(z_t[i,:],i)
            tmp[i] = func(tmp[i])
            z_tplus.append( tmp )            
        z_tplus = np.vstack((z_tplus))
        z_list.append(z_tplus)
    return z_list


def initialization2(N,g,mu,l,r):
    ### W ###
    A = nx.adjacency_matrix(g).todense().A
    L = np.diag(np.sum(A,0))-A
    eig_val,_ = np.linalg.eig(L)
    if r<=0.5*np.max(eig_val):
        print('r <= 0.5*lambd_max')
    W = np.eye(N)-(L)/r
    
    ### thet ###
    eig_val,_ = np.linalg.eig(W)
    eig_val = np.sort(eig_val)
    rho_w = eig_val[-2]
    thet_max = (1-rho_w)/l
    thet_max_bar = random.uniform(0, thet_max)
    while thet_max_bar==0 or thet_max_bar==thet_max:
        thet_max_bar = random.uniform(0, thet_max)
    thet_min = N*l*l*thet_max_bar*thet_max_bar/(mu*(1-rho_w-l*thet_max_bar))
    thet_max = thet_max_bar
    thet = []
    print('error: thet_max < thet_min :\n1) thet_max :',thet_max,'; 2) thet_min :', thet_min)
    for i in range(N):
        tmp = random.uniform(thet_min, thet_max)
        while tmp==thet_min or tmp==thet_max:
            tmp = random.uniform(thet_min, thet_max)
        thet.append(tmp)
    
    ### lambd ###
    thet_min_bar = min(thet)
    thet_max_bar = max(thet)
    mat = np.array([[mu*thet_min_bar/N,-l*thet_max_bar],
                    [-l*thet_max_bar,1-rho_w-l*thet_max_bar] ])
    eig_val,_ = np.linalg.eig(mat)
    mu_bar = np.min(eig_val)
    l_bar = l*thet_max_bar+1
    lambd_max = 2*mu_bar/(l_bar*l_bar)
    lambd_max_bar = random.uniform(0, lambd_max)
    while lambd_max_bar==0 or lambd_max_bar==lambd_max:
        lambd_max_bar = random.uniform(0, lambd_max)
    lambd_min = lambd_max_bar*lambd_max_bar*l_bar*l_bar/(2*mu_bar*mu_bar)
    lambd = []
    for i in range(N):
        tmp = random.uniform(lambd_min, lambd_max)
        while tmp==lambd_min or tmp==lambd_max:
            tmp = random.uniform(lambd_min, lambd_max)
        lambd.append(tmp)
    
    return W, thet, lambd


def iter2_1(Z,N,thet,lambd,W,iterations):
    z_list = [Z.copy()]
    for iter_ in range(iterations):
        z_tplus = []
        z_t = z_list[-1]
        for i in range(N):
            tmp = z_t[i]-lambd[i]*(z_t[i]-sum([W[i,j]*z_t[j,:] for j in range(N)])+
                                    thet[i]*gi_bar(z_t[i],i))
            tmp[i] = func(tmp[i])
            z_tplus.append( tmp )
        z_tplus = np.vstack((z_tplus))
        z_list.append(z_tplus)
    return z_list


def iter2_2(Z,N,thet,lambd,W,iterations):
    z_list = [Z.copy()]
    lambd_list = [lambd.copy()]
    thet_list = [thet.copy()]
    for iter_ in range(iterations):
        z_tplus = []
        lambd_tplus = []
        thet_tplus = []
        z_t = z_list[-1]
        lambd = lambd_list[-1]
        thet = thet_list[-1]
        for i in range(N):
            tmp = z_t[i,:]-lambd[i]*(z_t[i,:]-sum([W[i,j]*z_t[j,:] for j in range(N)])+
                                      thet[i]*gi_bar(z_t[i],i))
            tmp[i] = func(tmp[i])
            z_tplus.append( tmp )
            
            tmp2 = sum([W[i,j]*lambd[j] for j in range(N)])
            lambd_tplus.append( tmp2 )
            
            tmp3 = sum([W[i,j]*thet[j] for j in range(N)])
            thet_tplus.append( tmp3 )
        z_tplus = np.vstack((z_tplus))
        z_list.append(z_tplus)
        lambd_list.append(lambd_tplus)
        thet_list.append(thet_tplus)
    return z_list, thet_list, lambd_list


def iter2_3(Z,N,thet,lambd,W,iterations):
    z_list = [Z.copy()]
    for iter_ in range(iterations):
        z_tplus = []
        z_t = z_list[-1]
        for i in range(N):
            tmp = z_t[i]-lambd[i]*(z_t[i]-sum([W[i,j]*z_t[j,:] for j in range(N)])+
                                    sum([W[i,j]*thet[j]*gi_bar(z_t[j],j) for j in range(N)]))
            tmp[i] = func(tmp[i])
            z_tplus.append( tmp )
        z_tplus = np.vstack((z_tplus))
        z_list.append(z_tplus)
    return z_list

def Nash_error(x,z):
    N_0=np.size(x)
    N_1=np.size(z,0)
    err=[]
    for i in range(N_1):
        z1=[z[i,j,j] for j in range(N_0)]
        temp=np.linalg.norm(x-z1)
        err.append(temp)
    return err


''' baseline '''
N = 20
x = [random.random()*200 for i in range(N)]
lambd = [0.01]*20
lambd2 = [0.1]*20
iterations = 2000
x_list = baseline(N,x,lambd,iterations)
x_list=np.array(x_list)
#for i in range(20):
#    plt.plot(x_list[:,i])

''' algorithm 1 '''

####  initialization  ####
N = 20
Z = np.random.random((N,N))*200

g = nx.Graph()
g.add_nodes_from([i for i in range(20)])
for i in range(N):
    g.add_edge(i,(i+1)%20)
    
#g2=nx.newman_watts_strogatz_graph(20,4,0)


#mu = 2
#l = 2
r = 4
iterations = 20000
W= initialization(g,r)
#W2= initialization(g2,r)
#Lambda=[random.uniform(0.05,0.15) for i in range(N)]
#
#Lambda2=[random.uniform(0.01,0.19) for i in range(N)]
#Lambda3=[random.uniform(0.001,0.199) for i in range(N)]
#

#####  1-1  ####
z_list_0 = iter1_1(Z,N,lambd2,W,iterations)
z_list_0=np.array(z_list_0)
e0=Nash_error(x_list[-1],z_list_0)
plt.figure(1)
plt.loglog(range(1,iterations+2),e0/e0[0],'r--')

z0_list_0 = iter1_1(Z,N,lambd2,W,iterations)
z0_list_0=np.array(z0_list_0)
e_0=Nash_error(x_list[-1],z0_list_0)
plt.loglog(range(1,iterations+2),e_0/e_0[0],'r-')

z_list = iter1_1(Z,N,Lambda,W,iterations)
z_list=np.array(z_list)
e1=Nash_error(x_list[-1],z_list)
plt.loglog(range(1,iterations+2),e1/e1[0],'b--')

z2_list = iter1_1(Z,N,Lambda2,W,iterations)
z2_list=np.array(z2_list)
e_1_1=Nash_error(x_list[-1],z2_list)
plt.loglog(range(1,iterations+2),e_1_1/e_1_1[0],'b-')

z3_list = iter1_1(Z,N,Lambda3,W,iterations)
z3_list=np.array(z3_list)
e_1_3=Nash_error(x_list[-1],z3_list)
plt.loglog(range(1,iterations+2),e_1_3/e_1_3[0],'c-')

#z1_list = iter1_1(Z,N,Lambda,W2,iterations)
#z1_list=np.array(z1_list)
#e_1=Nash_error(x_list[-1],z1_list)
#plt.loglog(range(1,iterations+2),e_1/e_1[0],'b-')
#
#####  1-2  ####
#z_list_2,lambd_list = iter1_2(Z,N,Lambda,W,iterations)
#z_list_2=np.array(z_list_2)
#e2=Nash_error(x_list[-1],z_list_2)
#plt.loglog(range(1,iterations+2),e2/e2[0],'c--')
#
#z2_list_2,lambd2_list = iter1_2(Z,N,Lambda,W2,iterations)
#z2_list_2=np.array(z2_list_2)
#e_2=Nash_error(x_list[-1],z2_list_2)
#plt.loglog(range(1,iterations+2),e_2/e_2[0],'c-')
#
#
#####  1-3  ####
#z_list_3 = iter1_3(Z,N,Lambda,W,iterations)
#z_list_3=np.array(z_list_3)
#e3=Nash_error(x_list[-1],z_list_3)
#e3=e3/e3[0]
#plt.loglog(range(1,iterations+2),e3,'k--')
#
#z3_list_3 = iter1_3(Z,N,Lambda,W2,iterations)
#z3_list_3=np.array(z3_list_3)
#e_3=Nash_error(x_list[-1],z3_list_3)
#e_3=e_3/e_3[0]
#plt.loglog(range(1,iterations+2),e_3,'k-')


####  1-4  ####
#z_list_4 = iter1_3(Z,N,Lambda,W,iterations)
#z_list_4=np.array(z_list_4)
#e4=Nash_error(x_list[-1],z_list_4)
#plt.loglog(e4/e4[0],'r-')

''' algorithm 2 '''

####  initialization  ####
# N = 20
# Z = np.random.random((N,N))*200

# g = nx.Graph()
# g.add_nodes_from([i for i in range(20)])
# for i in range(N):
#     g.add_edge(i,(i+1)%20)

# mu = 2
# l = 2
# r = 4
#iterations = 100000
## W, thet, lambd = initialization2(N,g,mu,l,r)
#thet=[random.uniform(0.1,0.3) for i in range(N)]
#thet0=[0.2]*20
#
#####  2-1  ####
#z_list_5 = iter2_1(Z,N,thet,Lambda,W,iterations)
#z_list_5=np.array(z_list_5)
#e5=Nash_error(x_list[-1],z_list_5)
#plt.figure(2)
#plt.loglog(range(1,iterations+2),e5/e5[0],'b--')
#
#z1_list_5 = iter2_1(Z,N,thet,Lambda,W2,iterations)
#z1_list_5=np.array(z1_list_5)
#e_5=Nash_error(x_list[-1],z1_list_5)
#plt.loglog(range(1,iterations+2),e_5/e_5[0],'b-')


#z_list_8 = iter2_1(Z,N,thet0,lambd2,W,iterations)
#z_list_8=np.array(z_list_8)
#e8=Nash_error(x_list[-1],z_list_8)
#plt.loglog(range(1,iterations+2),e8/e8[0],'r--')
#
#z1_list_8 = iter2_1(Z,N,thet0,lambd2,W2,iterations)
#z1_list_8=np.array(z1_list_8)
#e_8=Nash_error(x_list[-1],z1_list_8)
#plt.loglog(range(1,iterations+2),e_8/e_8[0],'r-')
##
######  2-2  ####
#z_list_6, thet_list, lambd_list = iter2_2(Z,N,thet,Lambda,W,iterations)
#z_list_6=np.array(z_list_6)
#e6=Nash_error(x_list[-1],z_list_6)
#plt.loglog(range(1,iterations+2),e6/e6[0],'c--')
#
#z1_list_6, thet1_list, lambd1_list = iter2_2(Z,N,thet,Lambda,W2,iterations)
#z1_list_6=np.array(z1_list_6)
#e_6=Nash_error(x_list[-1],z1_list_6)
#plt.loglog(range(1,iterations+2),e_6/e_6[0],'c-')
##
######  2-3  ####
#z_list_7 = iter2_3(Z,N,thet,Lambda,W,iterations)
#z_list_7=np.array(z_list_7)
#e7=Nash_error(x_list[-1],z_list_7)
#plt.loglog(range(1,iterations+2),e7/e7[0],'k--')
#
#z1_list_7 = iter2_3(Z,N,thet,Lambda,W2,iterations)
#z1_list_7=np.array(z1_list_7)
#e_7=Nash_error(x_list[-1],z1_list_7)
#plt.loglog(range(1,iterations+2),e_7/e_7[0],'k-')