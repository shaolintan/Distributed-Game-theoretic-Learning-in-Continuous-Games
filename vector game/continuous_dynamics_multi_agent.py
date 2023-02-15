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


class definition():
    def __init__(self,lambada=1,R=1,n=6,beta=5,iter_max=200,initialization='random',
                 x0=None,y0=None,epsabs=1.49e-02,epsrel=1.49e-02,grids_hot=1000,
                 density_center=[(0.35,0.65),(0.35,0.35),(0.65,0.65),(0.65,0.35)],
                 obstacle=[],G=nx.Graph(),alpha=0.5):
        self.R = R
        self.n = n
        self.obstacle = obstacle
        self.alpha = alpha
        if not isinstance(lambada,list):
            lambada = [lambada]*n
        self.lambada = lambada
        if G.number_of_nodes()!=n:
            G = nx.Graph()
            G.add_nodes_from([str(i+1) for i in range(n)])
            for i in range(n-1):
                G.add_edge(str(i+1),str(i+2))
            G.add_edge(str(n),str(1))
        self.G = G.copy()
        if not isinstance(beta,list):
            beta = [beta]*n
        self.beta = beta
        self.iter_max = iter_max
        self.f_p_i = [partial(cf.Pi_sa,beta = b) for b in beta]
        self.f_p_all = partial(cf.P_sa,p_i = self.f_p_i)
        x = np.zeros((n,n,iter_max))
        y = np.zeros((n,n,iter_max))
        if x0 is None:
            if initialization=='zero':
                for i in range(n):
                    for j in range(n):
                        x[i][j][0] = 0
                        y[i][j][0] = 0
            elif initialization=='random':
                for i in range(n):
                    for j in range(n):
                        x[i][j][0] = random.uniform(0, 1)
                        y[i][j][0] = random.uniform(0, 1)
        else:
            for i in range(n):
                for j in range(n):
                    x[i][j][0] = x0[i][j]
                    y[i][j][0] = y0[i][j]
        ####如果初始点在障碍物内，则随机初始化初始点####
        for i in range(n):
            sign = cf.is_in_obstacle(x[i,:,0],y[i,:,0],obstacle)
            for j in range(n):
                while sign[j]>0:
                    x[i][j][0] = random.uniform(0, 1)
                    y[i][j][0] = random.uniform(0, 1)
                    sign = cf.is_in_obstacle(x[i,:,0],y[i,:,0],obstacle)
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
        self.symbol_phi = cf.Phi_a(self.f_v,self.f_p_all,a)
        self.epsabs=epsabs
        self.epsrel=epsrel
        self.Phi = np.zeros((n,iter_max))
        self.Phi_all = np.zeros(iter_max)
        self.grids = grids_hot
        
    def dynamics_iter(self):
        n = self.n
        dvi_dxi = [0]*self.n
        dvi_dyi = [0]*self.n
        for i in range(self.n):
            j = i
            #dvi_dxi[i],dvi_dyi[i] = cf.diff_VP(self.f_v,self.f_p_all,self.f_p_i,self.a,j)
            dvi_dxi[i],dvi_dyi[i] = cf.diff_VP2(self.xs,self.ys,self.f_v,self.a,j,self.beta)
        nodes = list(self.G.nodes())
        Neighbor = [0]*self.n
        for i in range(self.n):
            Neighbor[i] = [nodes.index(j) for j in list(self.G[nodes[i]])]
        x_next = np.zeros((n,n))
        y_next = np.zeros((n,n))
        dis_x = np.zeros((n,n))
        dis_y = np.zeros((n,n))
        for n_iter in range(1,self.iter_max):
            ##########不考虑障碍物相关信息确定下一步的位置##########
            for i in range(self.n):
                temp = {}
                for j in range(self.n):
                    temp[self.a[j][0]]=self.x[i][j][n_iter-1]
                    temp[self.a[j][1]]=self.y[i][j][n_iter-1]
                for j in range(self.n):
                    ####x坐标微分####
                    gradient_xi = dvi_dxi[j].subs(temp)
                    ##二重积分数值解##
                    fx = lambdify([self.xs,self.ys],gradient_xi)
                    px,err= scipy.integrate.dblquad(fx,0,self.R, 0,self.R, epsabs=self.epsabs, epsrel=self.epsrel)
                    ####y坐标微分####
                    gradient_yi = dvi_dyi[j].subs(temp)
                    ##二重积分数值解##
                    fy = lambdify([self.xs,self.ys],gradient_yi)
                    py,err= scipy.integrate.dblquad(fy,0,self.R,0,self.R, epsabs=self.epsabs, epsrel=self.epsrel)
                    ####个体之间的距离####
                    dis_x[i][j] = sum([2*(self.x[i][j][n_iter-1]-self.x[k][j][n_iter-1]) for k in Neighbor[i]])
                    dis_y[i][j] = sum([2*(self.y[i][j][n_iter-1]-self.y[k][j][n_iter-1]) for k in Neighbor[i]])
                    ####下一步的位置####
                    x_next[i][j] = self.x[i][j][n_iter-1]+self.lambada[j]*(self.alpha*px-(1-self.alpha)*dis_x[i][j])
                    y_next[i][j] = self.y[i][j][n_iter-1]+self.lambada[j]*(self.alpha*py-(1-self.alpha)*dis_y[i][j])
                    
                ####计算每个iter每个探测器的Phi(a)####
                symbol_phi = self.symbol_phi.subs(temp)
                phi_iter = lambdify([self.xs,self.ys],symbol_phi)
                self.Phi[i][n_iter-1],_ = scipy.integrate.dblquad(phi_iter,0,self.R,0,self.R, epsabs=self.epsabs, epsrel=self.epsrel)
            ####计算每个iter所有探测器的Phi(a)####
            self.Phi_all[n_iter-1] = np.sum(self.Phi[:,n_iter-1])-np.sum(dis_x)-np.sum(dis_y)
            ##########考虑障碍物相关信息确定下一步的位置##########
            for i in range(n):
                sign1,sign2 = cf.is_obstacle(n,self.x[i,:,n_iter-1],self.y[i,:,n_iter-1],x_next[i],y_next[i],self.obstacle)
                for j in range(n):
                    self.x[i][j][n_iter] = sign2[j][0]
                    self.y[i][j][n_iter] = sign2[j][1]
        ##########计算最后一步的Phi(a)##########
        ####计算每个个体的Phi(a)####
        for i in range(self.n):
            temp = {}
            for j in range(n):
                temp[self.a[j][0]]=self.x[i][j][-1]
                temp[self.a[j][1]]=self.y[i][j][-1]
            symbol_phi = self.symbol_phi.subs(temp)
            phi_iter = lambdify([self.xs,self.ys],symbol_phi)
            self.Phi[i][-1],_ = scipy.integrate.dblquad(phi_iter,0,self.R,0,self.R, epsabs=self.epsabs, epsrel=self.epsrel)
        ####计算所有个体的Phi(a)####
        for i in range(n):
            for j in range(n):
                dis_x[i][j] = sum([2*(self.x[i][j][n_iter-1]-self.x[k][j][n_iter-1]) for k in Neighbor[i]])
                dis_y[i][j] = sum([2*(self.y[i][j][n_iter-1]-self.y[k][j][n_iter-1]) for k in Neighbor[i]])
        self.Phi_all[-1] = np.sum(self.Phi[:,-1])-np.sum(dis_x)-np.sum(dis_y)
        
        return self.x,self.y,self.Phi,self.Phi_all
    
    def draw(self):
        plt.figure()
        ####画热图####
        zt = self.f_v(self.xs,self.ys)
        ft = lambdify([self.xs,self.ys],zt)
        hot = np.zeros((self.grids,self.grids))
        for i in range(self.grids):
            for j in range(self.grids):
                hot[i][j] = ft(self.R/self.grids*i,self.R/self.grids*j)
        plt.imshow(hot, extent=(0, self.R, 0, self.R),
                   cmap=plt.cm.hot, norm=LogNorm())
        plt.colorbar()
        ####画个体轨迹####
        for i in range(self.n):
            plt.plot(self.x[i][i],self.y[i][i])
            plt.scatter(self.x[i][i][-1],self.y[i][i][-1],marker='^')
        ####画密度中心####
        x_t = np.zeros(len(self.density_center))
        y_t = np.zeros(len(self.density_center))
        for i in range(len(self.density_center)):
            x_t[i] = self.density_center[i][0]
            y_t[i] = self.density_center[i][1]
        plt.scatter(x_t,y_t,marker='o')
        ####画障碍物####
        for i in range(len(self.obstacle)):
            xo1 = self.obstacle[i][0]
            xo2 = self.obstacle[i][0]+self.obstacle[i][2]
            yo1 = self.obstacle[i][1]
            yo2 = self.obstacle[i][1]+self.obstacle[i][3]
            plt.vlines(xo1, yo1, yo2, colors = "k", linestyles = "dashed")
            plt.vlines(xo2, yo1, yo2, colors = "k", linestyles = "dashed")
            plt.hlines(yo1, xo1, xo2, colors = "k", linestyles = "dashed")
            plt.hlines(yo2, xo1, xo2, colors = "k", linestyles = "dashed")
        plt.xlim(0,self.R)
        plt.ylim(0,self.R)
        plt.show()
    
    def draw_without_hot(self):
        plt.figure()
        ####画个体轨迹####
        for i in range(self.n):
            plt.plot(self.x[i][i],self.y[i][i])
            plt.scatter(self.x[i][i][-1],self.y[i][i][-1],marker='^')
        ####画密度中心####
        x_t = np.zeros(len(self.density_center))
        y_t = np.zeros(len(self.density_center))
        for i in range(len(self.density_center)):
            x_t[i] = self.density_center[i][0]
            y_t[i] = self.density_center[i][1]
        plt.scatter(x_t,y_t,marker='o')
        ####画障碍物####
        for i in range(len(self.obstacle)):
            xo1 = self.obstacle[i][0]
            xo2 = self.obstacle[i][0]+self.obstacle[i][2]
            yo1 = self.obstacle[i][1]
            yo2 = self.obstacle[i][1]+self.obstacle[i][3]
            plt.vlines(xo1, yo1, yo2, colors = "k", linestyles = "dashed")
            plt.vlines(xo2, yo1, yo2, colors = "k", linestyles = "dashed")
            plt.hlines(yo1, xo1, xo2, colors = "k", linestyles = "dashed")
            plt.hlines(yo2, xo1, xo2, colors = "k", linestyles = "dashed")
        plt.xlim(0,self.R)
        plt.ylim(0,self.R)
        plt.show()
