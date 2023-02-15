# -*- coding: utf-8 -*-

import sympy
from sympy import lambdify,diff,integrate
import scipy.integrate
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.colors import LogNorm
import continuous_function as cf


class definition():
    def __init__(self,lambada=1,R=1,n=6,beta=5,iter_max=200,initialization='random',
                 x0=None,y0=None,epsabs=1.49e-02,epsrel=1.49e-02,grids_hot=1000,
                 density_center=[(0.35,0.65),(0.35,0.35),(0.65,0.65),(0.65,0.35)],
                 obstacle=[]):
        self.R = R
        self.n = n
        self.obstacle = obstacle
        if not isinstance(beta,list):
            beta = [beta]*n
        if not isinstance(lambada,list):
            lambada = [lambada]*n
        self.lambada = lambada
        self.beta = beta
        self.iter_max = iter_max
        self.f_p_i = [partial(cf.Pi_sa,beta = b) for b in beta]
        self.f_p_all = partial(cf.P_sa,p_i = self.f_p_i)
        x = np.zeros((n,iter_max))
        y = np.zeros((n,iter_max))
        if x0 is None:
            if initialization=='zero':
                for i in range(n):
                    x[i][0] = 0
                    y[i][0] = 0
            elif initialization=='random':
                for i in range(n):
                    x[i][0] = random.uniform(0, 1)
                    y[i][0] = random.uniform(0, 1)
        else:
            for i in range(n):
                x[i][0] = x0[i]
                y[i][0] = y0[i]
        ####如果初始点在障碍物内，则随机初始化初始点####
        sign = cf.is_in_obstacle(x[:,0],y[:,0],obstacle)
        for i in range(n):
            while sign[i]>0:
                x[i][0] = random.uniform(0, 1)
                y[i][0] = random.uniform(0, 1)
                sign = cf.is_in_obstacle(x[:,0],y[:,0],obstacle)
        self.x = x
        self.y = y
        self.xs = sympy.symbols("xs")
        self.ys = sympy.symbols("ys")
        zt = cf.V_S(self.xs,self.ys,R,integ=1)
        ft = lambdify([self.xs,self.ys],zt)
        integ,_ = scipy.integrate.dblquad(ft,0,R,0,R)
        self.density_center = density_center
        self.f_v = partial(cf.V_S,R=R,integ=integ,density_center=density_center)
        a = [0]*n
        for n_i in range(n):
            exec('x'+str(n_i+1)+' = sympy.symbols('+"'x"+str(n_i+1)+"'"+')')
            exec('y'+str(n_i+1)+' = sympy.symbols('+"'y"+str(n_i+1)+"'"+')')
            a[n_i] = eval('(x'+str(n_i+1)+','+'y'+str(n_i+1)+')')
        self.a = a
        self.symbol_phi = cf.Phi_a2(self.xs,self.ys,self.f_v,self.a,self.beta)
        self.epsabs=epsabs
        self.epsrel=epsrel
        self.Phi = np.zeros(iter_max)
        self.grids = grids_hot
        
    def dynamics_iter(self):
        dvi_dxi = [0]*self.n
        dvi_dyi = [0]*self.n
        for i in range(self.n):
            j = i
            dvi_dxi[i],dvi_dyi[i] = cf.diff_VP2(self.xs,self.ys,self.f_v,self.a,j,self.beta)
        x_next = [0]*self.n
        y_next = [0]*self.n
        for n_iter in range(1,self.iter_max):
            temp = {}
            for i in range(self.n):
                temp[self.a[i][0]]=self.x[i][n_iter-1]
                temp[self.a[i][1]]=self.y[i][n_iter-1]
            ##########不考虑障碍物相关信息确定下一步的位置##########
            for i in range(self.n):
                ####x坐标微分####
                gradient_xi = dvi_dxi[i].subs(temp)
                ##二重积分数值解##
                fx = lambdify([self.xs,self.ys],gradient_xi)
                px,err= scipy.integrate.dblquad(fx,0,self.R, 0,self.R, epsabs=self.epsabs, epsrel=self.epsrel)
                ####y坐标微分####
                gradient_yi = dvi_dyi[i].subs(temp)
                ##二重积分数值解##
                fy = lambdify([self.xs,self.ys],gradient_yi)
                py,err= scipy.integrate.dblquad(fy,0,self.R,0,self.R, epsabs=self.epsabs, epsrel=self.epsrel)
                
                x_next[i] = self.x[i][n_iter-1]+self.lambada[i]*px
                y_next[i] = self.y[i][n_iter-1]+self.lambada[i]*py
            ##########考虑障碍物相关信息确定下一步的位置##########
            xi = [i for i in self.x[:,n_iter-1]]
            yi = [i for i in self.y[:,n_iter-1]]
            sign1,sign2 = cf.is_obstacle(self.n,xi,yi,x_next,y_next,self.obstacle)
            for i in range(self.n):
                self.x[i][n_iter] = sign2[i][0]
                self.y[i][n_iter] = sign2[i][1]
            ##########计算每一步的Phi(a)##########
            symbol_phi = self.symbol_phi.subs(temp)
            phi_iter = lambdify([self.xs,self.ys],symbol_phi)
            self.Phi[n_iter-1],_ = scipy.integrate.dblquad(phi_iter,0,self.R,0,self.R, epsabs=self.epsabs, epsrel=self.epsrel)
        ##########计算最后步的Phi(a)##########
        temp = {}
        for i in range(self.n):
            temp[self.a[i][0]]=self.x[i][-1]
            temp[self.a[i][1]]=self.y[i][-1]
        symbol_phi = self.symbol_phi.subs(temp)
        phi_iter = lambdify([self.xs,self.ys],symbol_phi)
        self.Phi[-1],_ = scipy.integrate.dblquad(phi_iter,0,self.R,0,self.R, epsabs=self.epsabs, epsrel=self.epsrel)
        return self.x,self.y,self.Phi
    
    def draw(self):
        plt.figure()
        zt = self.f_v(self.xs,self.ys)
        ft = lambdify([self.xs,self.ys],zt)
        hot = np.zeros((self.grids,self.grids))
        for i in range(self.grids):
            for j in range(self.grids):
                hot[i][j] = ft(self.R/self.grids*i,self.R/self.grids*j)
        plt.imshow(hot, extent=(0, self.R, 0, self.R),
                   cmap=plt.cm.hot, norm=LogNorm())
        plt.colorbar()
        for i in range(self.n):
            plt.plot(self.x[i],self.y[i])
            plt.scatter(self.x[i][-1],self.y[i][-1],marker='^')
        x_t = np.zeros(len(self.density_center))
        y_t = np.zeros(len(self.density_center))
        for i in range(len(self.density_center)):
            x_t[i] = self.density_center[i][0]
            y_t[i] = self.density_center[i][1]
        for i in range(len(self.obstacle)):
            xo1 = self.obstacle[i][0]
            xo2 = self.obstacle[i][0]+self.obstacle[i][2]
            yo1 = self.obstacle[i][1]
            yo2 = self.obstacle[i][1]+self.obstacle[i][3]
            plt.vlines(xo1, yo1, yo2, colors = "k", linestyles = "dashed")
            plt.vlines(xo2, yo1, yo2, colors = "k", linestyles = "dashed")
            plt.hlines(yo1, xo1, xo2, colors = "k", linestyles = "dashed")
            plt.hlines(yo2, xo1, xo2, colors = "k", linestyles = "dashed")
        plt.scatter(x_t,y_t,marker='o')
        plt.xlim(0,self.R)
        plt.ylim(0,self.R)
        plt.show()
    
    def draw_without_hot(self):
        plt.figure()
        for i in range(self.n):
            plt.plot(self.x[i],self.y[i])
            plt.scatter(self.x[i][-1],self.y[i][-1],marker='^')
        x_t = np.zeros(len(self.density_center))
        y_t = np.zeros(len(self.density_center))
        for i in range(len(self.density_center)):
            x_t[i] = self.density_center[i][0]
            y_t[i] = self.density_center[i][1]
        for i in range(len(self.obstacle)):
            xo1 = self.obstacle[i][0]
            xo2 = self.obstacle[i][0]+self.obstacle[i][2]
            yo1 = self.obstacle[i][1]
            yo2 = self.obstacle[i][1]+self.obstacle[i][3]
            plt.vlines(xo1, yo1, yo2, colors = "k", linestyles = "dashed")
            plt.vlines(xo2, yo1, yo2, colors = "k", linestyles = "dashed")
            plt.hlines(yo1, xo1, xo2, colors = "k", linestyles = "dashed")
            plt.hlines(yo2, xo1, xo2, colors = "k", linestyles = "dashed")
        plt.scatter(x_t,y_t,marker='o')
        plt.xlim(0,self.R)
        plt.ylim(0,self.R)
        plt.show()


