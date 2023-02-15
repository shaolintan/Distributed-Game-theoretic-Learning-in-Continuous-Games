# -*- coding: utf-8 -*-


import sympy
from sympy import lambdify,diff,integrate
import scipy.integrate
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import LogNorm
import networkx as nx
import continuous_function as cf

def V_S(x,y,R,integ,density_center=[(0.35,0.65),(0.35,0.35),(0.65,0.65),(0.65,0.35)]):#目标概率分布
    #return 1/R**2
    res = 0
    res = res+5*sympy.exp(-20*((x-0.7)**2+(y-0.7)**2))
    res = res+sympy.exp(-50*((x-0.2)**2+(y-0.2)**2))
# =============================================================================
#     res = 0
#     for i,j in density_center:
#         res = res+sympy.exp(-10*((x-i)**2+(y-j)**2))
# =============================================================================
    res = res/integ
    return res


def Pi_sa(xs,ys,xi,yi,beta):#第i个探测器的探测概率Pi(s,ai)
    return sympy.exp(-beta*sympy.sqrt((xs-xi)**2+(ys-yi)**2+np.spacing(1)))


def P_sa(xs,ys,a,p_i):#多个探测器的探测概率P(s,a)
    p_all = 1
    for i in range(len(a)):
        if 0 not in a[i]:
            xi = a[i][0]
            yi = a[i][1]
            p_all = p_all*(1-p_i[i](xs,ys,xi,yi))
    return p_all


def diff_VP(f_v,f_p_all,f_p_i,a,i):#第i个探测器积分内式子关于xi和yi的导数
    xs = sympy.symbols("xs")
    ys = sympy.symbols("ys")
    xi = a[i][0]
    yi = a[i][1]
    a2 = a.copy()
    a2[i] = [0]
    V_i = f_v(xs,ys)*f_p_all(xs,ys,a2)*f_p_i[i](xs,ys,xi,yi)
    dvi_dxi = diff(V_i,xi)
    dvi_dyi = diff(V_i,yi)
    return dvi_dxi, dvi_dyi


def diff_VP2(xs,ys,f_v,a,j,beta):#第i个探测器积分内式子关于xj和yj的导数
    xj = a[j][0]
    yj = a[j][1]
    index = [i for i in range(len(a))]
    index.remove(j)
    p_all = 1
    for i in index:
        xi = a[i][0]
        yi = a[i][1]
        p_all = p_all*(1-Pi_sa(xs,ys,xi,yi,beta[i]))  
    V_i = f_v(xs,ys)*p_all*Pi_sa(xs,ys,xj,yj,beta[j])
    dvi_dxi = diff(V_i,xj)
    dvi_dyi = diff(V_i,yj)
    return dvi_dxi, dvi_dyi


def Phi_a(f_v,f_p_all,a):#优化目标，Phi(a)
    xs = sympy.symbols("xs")
    ys = sympy.symbols("ys")
    Phi = f_v(xs,ys)*(1-f_p_all(xs,ys,a))
    return Phi

def Phi_a2(xs,ys,f_v,a,beta):#优化目标，Phi(a)
    p_all = 1
    for i in range(len(a)):
        xi = a[i][0]
        yi = a[i][1]
        p_all = p_all*(1-Pi_sa(xs,ys,xi,yi,beta[i]))
    Phi = f_v(xs,ys)*(1-p_all)
    return Phi


def is_obstacle(n,xi,yi,xn,yn,obstacle):
    sign1 = [0]*n
    num_obstacle = len(obstacle)
    sign2 = [0]*n
    for i in range(n):
        sign2[i] = (xn[i],yn[i])
    for i_obs in range(num_obstacle):
        xo1 = obstacle[i_obs][0]
        xo2 = obstacle[i_obs][0]+obstacle[i_obs][2]
        yo1 = obstacle[i_obs][1]
        yo2 = obstacle[i_obs][1]+obstacle[i_obs][3]
        for i_v in range(n):
            ##############判断是否在边界##############
            if xi[i_v]==xo1 or xi[i_v]==xo2:
                if yo1<yi[i_v]<yo2:
                    sign1[i_v] = i_obs+1
                    sign2[i_v] = (xi[i_v],yn[i_v])#在左右两边
                    break
            if yi[i_v]==yo1 or yi[i_v]==yo2:
                if xo1<xi[i_v]<xo2:
                    sign1[i_v] = i_obs+1
                    sign2[i_v] = (xn[i_v],yi[i_v])#在上下两边
                    break
            ##############判断是否存在交点##############
            #左边#
            if xi[i_v]<xo1:
                if xn[i_v]>xo1:
                    y2 = (xo1-xi[i_v])*(yn[i_v]-yi[i_v])/(xn[i_v]-xi[i_v])+yi[i_v]
                    if yo1<y2<yo2:
                        sign1[i_v] = i_obs+1
                        sign2[i_v] = (xo1,y2)
                        break
            #右边#
            if xi[i_v]>xo2:
                if xn[i_v]<xo2:
                    y2 = (xo2-xn[i_v])*(yn[i_v]-yi[i_v])/(xn[i_v]-xi[i_v])+yn[i_v]
                    if yo1<y2<yo2:
                        sign1[i_v] = i_obs+1
                        sign2[i_v] = (xo2,y2)
                        break
            #上边#
            if yi[i_v]>yo2:
                if yn[i_v]<yo2:
                    x2 = (yo2-yn[i_v])*(xn[i_v]-xi[i_v])/(yn[i_v]-yi[i_v])+xn[i_v]
                    if xo1<x2<xo2:
                        sign1[i_v] = i_obs+1
                        sign2[i_v] = (x2,yo2)
                        break
            #下边#
            if yi[i_v]<yo1:
                if yn[i_v]>yo2:
                    x2 = (yo1-yi[i_v])*(xn[i_v]-xi[i_v])/(yn[i_v]-yi[i_v])+xi[i_v]
                    if xo1<x2<xo2:
                        sign1[i_v] = i_obs+1
                        sign2[i_v] = (x2,yo1)
                        break
    return sign1,sign2


def is_in_obstacle(xi,yi,obstacle):
    n = len(xi)
    sign = [0]*n
    num_obstacle = len(obstacle)
    for i_obs in range(num_obstacle):
        xo1 = obstacle[i_obs][0]
        xo2 = obstacle[i_obs][0]+obstacle[i_obs][2]
        yo1 = obstacle[i_obs][1]
        yo2 = obstacle[i_obs][1]+obstacle[i_obs][3]
        for i_v in range(n):
            if xo1<xi[i_v]<xo2:
                if yo1<yi[i_v]<yo2:
                    sign[i_v] = 1
    return sign