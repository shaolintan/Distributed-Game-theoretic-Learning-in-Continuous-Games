# -*- coding: utf-8 -*-



from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from math import exp,sqrt,ceil
import random
from matplotlib.colors import LogNorm
from scipy.optimize import minimize
import networkx as nx
import disperse_function



def V_S(x,y,R,integ,density_center=[(0.35,0.65),(0.35,0.35),(0.65,0.65),(0.65,0.35)]):#目标概率分布
    #return 1/R**2
    #return ((x-1)**2+(y-1)**2)*3/2/R**4
    res = 0
    res = res+5*exp(-20*((x-0.7)**2+(y-0.8)**2))
    res = res+exp(-50*((x-0.3)**2+(y-0.2)**2))
    
# =============================================================================
#     for i,j in density_center:
#         res = res+5*exp(-10*((x-i)**2+(y-j)**2))
# =============================================================================
    res = res/integ
    return res


def Pi_sa(xs,ys,xi,yi,beta):
    return exp(-beta*sqrt((xs-xi)**2+(ys-yi)**2))


def P_sa(xs,ys,a,p_i):
    p_all = 1
    for i in range(len(a)):
        if len(a[i])==2:
            xi = a[i][0]
            yi = a[i][1]
            p_all = p_all*(1-p_i[i](xs,ys,xi,yi))
    return p_all
# =============================================================================
# def P_sa(xs,ys,a,p_i):
#     p_all = 1
#     for i in range(len(a)):
#         xi = a[i][0]
#         yi = a[i][1]
#         p_all = p_all*(1-p_i[i](xs,ys,xi,yi))
#     return p_all
# =============================================================================

def V_i(f_v,f_p_all,f_p_i,a,xs,ys,R,xi,yi):
    result = f_v(xs,ys)*f_p_all(xs,ys,a)*f_p_i(xs,ys,xi,yi)
    return result


def integrate_s(f_v,f_p_all,f_p_i,a,xs,ys,R,xi,yi):
    result = 0
    for i_xs in range(np.size(xs)):
        for i_ys in range(np.size(ys)):
            result = result+V_i(f_v,f_p_all,f_p_i,a,xs[i_xs],ys[i_ys],R,xi,yi)
    return result

def integrate_s2(f_v,beta,a,xs,ys,xj,yj,j):
    result = 0
    ###删除第j个，比如index==[0,1,2,3,5],j=4###
    index = [i for i in range(len(a))]
    index.remove(j)
    
    for i_xs in range(np.size(xs)):
        for i_ys in range(np.size(ys)):
            ###V(S)###
            vs = f_v(xs[i_xs],ys[i_ys])
            ###∏(1-p_i(s,a_i))，除了第j个，连乘###
            p_all = 1
            for i in index:
                #p_all = p_all*(1-f_p_i[i](xs[i_xs],ys[i_ys],a[i][0],a[i][1]))
                p_all = p_all*(1-Pi_sa(xs[i_xs],ys[i_ys],a[i][0],a[i][1],beta[i]))#a[i][0],a[i][1]表示第i个的坐标(xi,yi)
            ###1-p_j(s,a_j)，第j个的五个方向之一的坐标###
            #p_j = f_p_i[j](xs[i_xs],ys[i_ys],xj,yj)
            p_j = Pi_sa(xs[i_xs],ys[i_ys],xj,yj,beta[j])
            ###累加，xs,ys为网格横坐标纵坐标，比如[0,0.01,……,0.99]###
            result = result+vs*p_all*p_j
    return result



def compute_a2(n,n_iter,num_grid,R,x,y):
    a = [0]*n
    for j in range(n):
        a_temp = [0]*n
        for k in range(n):
            a_temp[k] = [x[(k+1)%n][n_iter-1]*R/num_grid, y[(k+1)%n][n_iter-1]*R/num_grid]
        a[j] = a_temp.copy()
    return a


def compute_a(n,n_iter,num_grid,R,x,y):
    a = [0]*n
    for k in range(n):
        a[k] = [x[(k+1)%n][n_iter-1]*R/num_grid, y[(k+1)%n][n_iter-1]*R/num_grid]
    return a


def Phi_a(a,f_v,f_p_all,xs,ys):
    Phi = 0
    for i in range(np.size(xs)):
        for j in range(np.size(ys)):
            Phi = Phi+f_v(xs[i],ys[j])*(1-f_p_all(xs[i],ys[j],a))
    return Phi

def Phi_a2(a,f_v,f_p_i,xs,ys):
    Phi = 0
    len_a = len(a)
    for i_xs in range(np.size(xs)):
        for i_ys in range(np.size(ys)):
            ###P(S,a)=∏(1-p_i(s,a_i))###
            p_all = 1
            for i in range(len_a):
                p_all = p_all*(1-f_p_i[i](xs[i_xs],ys[i_ys],a[i][0],a[i][1]))
            ###Phi=∑[V(S)*(1-∏(1-p_i(s,a_i)))]###
            Phi = Phi+f_v(xs[i_xs],ys[i_ys])*(1-p_all)
    return Phi

def Phi_a3(a,f_v,beta,xs,ys):
    Phi = 0
    len_a = len(a)
    for i_xs in range(np.size(xs)):
        for i_ys in range(np.size(ys)):
            ###P(S,a)=∏(1-p_i(s,a_i))###
            p_all = 1
            for i in range(len_a):
                #p_all = p_all*(1-f_p_i[i](xs[i_xs],ys[i_ys],a[i][0],a[i][1]))
                p_all = p_all*(1-Pi_sa(xs[i_xs],ys[i_ys],a[i][0],a[i][1],beta[i]))
            ###Phi=∑[V(S)*(1-∏(1-p_i(s,a_i)))]###
            Phi = Phi+f_v(xs[i_xs],ys[i_ys])*(1-p_all)
    return Phi


def computing_obstacle_grid(R,num_grid,obstacle):
    obstacle_grid = np.zeros((num_grid,num_grid))
    for i in range(len(obstacle)):
        xo1 = obstacle[i][0]
        xo2 = obstacle[i][0]+obstacle[i][2]
        yo1 = obstacle[i][1]
        yo2 = obstacle[i][1]+obstacle[i][3]
        xo1_grid = ceil(xo1/R*num_grid)
        xo2_grid = int(xo2/R*num_grid)+1
        yo1_grid = ceil(yo1/R*num_grid)
        yo2_grid = int(yo2/R*num_grid)+1
        for j in range(xo1_grid,xo2_grid):
            for k in range(yo1_grid,yo2_grid):
                obstacle_grid[j][k] = 1
    return obstacle_grid


def is_obstacle(R,num_grid,n,n_iter,x,y,obstacle,obstacle_grid):
    sign = [[0]*4]*n
    for i in range(n):
        ####第一步判断是否超出边界####
        if y[i][n_iter-1] == num_grid-1:
            sign[i][0] = 1
        if y[i][n_iter-1] == 0:
            sign[i][1] = 1
        if x[i][n_iter-1] == 0:
            sign[i][2] = 1
        if x[i][n_iter-1] == num_grid-1:
            sign[i][3] = 1
        ####第二步判断每个探测器4个方向是否碰到障碍物####
        ##上下左右4个方向的x和y坐标##
        xn = (x[i][n_iter-1],x[i][n_iter-1],(x[i][n_iter-1]-1),(x[i][n_iter-1]+1))
        yn = ((y[i][n_iter-1]+1),(y[i][n_iter-1]-1),y[i][n_iter-1],y[i][n_iter-1])
        ##每个探测器，判断四个方向坐标是否在障碍物内##
        for j in range(len(obstacle)):
            temp = [ii for ii in np.where(np.array(sign[i])==0)[0]]#对于未到边界和在0~len(obstacle)障碍判断可通行的方向
            if len(temp)>0:
                for k in temp:
                    if obstacle_grid[int(xn[k])][int(yn[k])]==1:
                        sign[i][k] = 1
    return sign