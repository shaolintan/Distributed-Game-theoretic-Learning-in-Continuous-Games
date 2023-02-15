# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from math import exp,sqrt,ceil
import random
from matplotlib.colors import LogNorm
from scipy.optimize import minimize
import networkx as nx
import disperse_function as df


class definition():
    def __init__(self,R=1,n=6,beta=5,iter_max=200,initialization='random',
                 x0=None,y0=None,num_grid=100,beta_exp=4000,tol=1e-6,
                 density_center=[(0.35,0.65),(0.35,0.35),(0.65,0.65),(0.65,0.35)],
                 obstacle=[],alpha=0,G=nx.Graph()):
        self.R = R
        self.n = n
        self.beta_exp = beta_exp
        self.num_grid = num_grid
        self.tol = tol
        self.obstacle = obstacle
        self.alpha = alpha
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
        self.f_p_i = [partial(df.Pi_sa,beta = b) for b in beta]
        self.f_p_all = partial(df.P_sa,p_i = self.f_p_i)
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
                        x[i][j][0] = random.randint(0,num_grid-1)
                        y[i][j][0] = random.randint(0,num_grid-1)
        else:
            for i in range(n):
                for j in range(n):
                    x[i][j][0] = x0[i][j]
                    y[i][j][0] = y0[i][j]
        ####如果初始点在障碍物内，则随机初始化初始点####
        self.obstacle_grid = df.computing_obstacle_grid(self.R,self.num_grid,self.obstacle)
        for i in range(n):
            for j in range(n):
                while self.obstacle_grid[int(x[i][j][0])][int(y[i][j][0])]==1:
                    x[i][j][0] = random.randint(0,num_grid-1)
                    y[i][j][0] = random.randint(0,num_grid-1)
        self.x = x
        self.y = y
        self.xs = np.array([i*R/num_grid for i in range(1,int(num_grid+1))])
        self.ys = np.array([i*R/num_grid for i in range(1,int(num_grid+1))])
        zt = 0
        for xs_grid in range(num_grid):
            for ys_grid in range(num_grid):
                zt = zt+df.V_S(self.xs[xs_grid],self.ys[ys_grid],R=R,integ=1)
        self.f_v = partial(df.V_S,R=R,integ=zt,density_center=density_center)
        self.Phi = np.zeros((n,iter_max))
        self.Phi_all = np.zeros(iter_max)
        self.density_center = density_center
        
    
    def dynamics_iter(self):
        n = self.n
        R = self.R
        num_grid = self.num_grid
        alpha = self.alpha
        nodes = list(self.G.nodes())
        Neighbor = [0]*self.n
        for i in range(self.n):
            Neighbor[i] = [nodes.index(j) for j in list(self.G[nodes[i]])]
        ##########计算第一步的Phi(a)##########
        dis_x = np.zeros((n,n))
        dis_y = np.zeros((n,n))
        for i in range(n):
            a_temp = [0]*n
            for j in range(n):
                a_temp[j] = [self.x[i][j][0]*R/num_grid, self.y[i][j][0]*R/num_grid]
                dis_x[i][j] = sum([(self.x[i][j][0]*R/num_grid-self.x[k][j][0]*R/num_grid)**2 for k in Neighbor[j]])
                dis_y[i][j] = sum([(self.y[i][j][0]*R/num_grid-self.y[k][j][0]*R/num_grid)**2 for k in Neighbor[j]])
            self.Phi[i][0] = df.Phi_a3(a_temp,self.f_v,self.beta,self.xs,self.ys)
            
        self.Phi_all[0] = np.sum(self.Phi[:,0])-np.sum(dis_x)-np.sum(dis_y)
        ##########每次迭代##########
        for n_iter in range(1,self.iter_max):
            beta2 = self.beta_exp
            n = self.n
            j_random = [i for i in range(n)]
            random.shuffle(j_random)
            for i,j in enumerate(j_random):
                ####抽取第i给个体的第j个####
                xt = self.x[i][j][n_iter-1]*R/num_grid
                yt = self.y[i][j][n_iter-1]*R/num_grid
                a_all2 = [0]*n
                for k in range(n):
                    a_all2[k] = [self.x[i][k][n_iter-1]*R/num_grid,self.y[i][k][n_iter-1]*R/num_grid]
                sign = df.is_obstacle(self.R,self.num_grid,1,n_iter,self.x[i][j:j+1],self.y[i][j:j+1],self.obstacle,self.obstacle_grid)
                
                a_all = a_all2.copy()
                a_all[j] = [0]
                #p = np.zeros(5)
                p2 = np.zeros(5)
                ####计算保持不变的Vi p[0]####
                #result = alpha*df.integrate_s(self.f_v,self.f_p_all,self.f_p_i[j],a_all,self.xs,self.ys,R,xt,yt)
                #p[0] = result-(1-alpha)*sum([(xt-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                result = alpha*df.integrate_s2(self.f_v,self.beta,a_all2,self.xs,self.ys,xt,yt,j)
                p2[0] = result-(1-alpha)*sum([(xt-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                ####计算向上的Vi p[1]####
                if sign[0][0]==0:
                    #result = alpha*df.integrate_s(self.f_v,self.f_p_all,self.f_p_i[j],a_all,self.xs,self.ys,R,xt,yt+R/num_grid)
                    #p[1] = result-(1-alpha)*sum([(xt-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt+R/num_grid-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                    result = alpha*df.integrate_s2(self.f_v,self.beta,a_all2,self.xs,self.ys,xt,yt+R/num_grid,j)
                    p2[1] = result-(1-alpha)*sum([(xt-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt+R/num_grid-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                ####计算向下的Vi p[2]####
                if sign[0][1]==0:
                    #result = alpha*df.integrate_s(self.f_v,self.f_p_all,self.f_p_i[j],a_all,self.xs,self.ys,R,xt,yt-R/num_grid)
                    #p[2] = result-(1-alpha)*sum([(xt-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt-R/num_grid-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                    result = alpha*df.integrate_s2(self.f_v,self.beta,a_all2,self.xs,self.ys,xt,yt-R/num_grid,j)
                    p2[2] = result-(1-alpha)*sum([(xt-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt-R/num_grid-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                ####计算向左的Vi p[3]####
                if sign[0][2]==0:
                    #result = alpha*df.integrate_s(self.f_v,self.f_p_all,self.f_p_i[j],a_all,self.xs,self.ys,R,xt-R/num_grid,yt)
                    #p[3] = result-(1-alpha)*sum([(xt-R/num_grid-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                    result = alpha*df.integrate_s2(self.f_v,self.beta,a_all2,self.xs,self.ys,xt-R/num_grid,yt,j)
                    p2[3] = result-(1-alpha)*sum([(xt-R/num_grid-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                ####计算向右的Vi p[4]####
                if sign[0][3]==0:#右
                    #result = alpha*df.integrate_s(self.f_v,self.f_p_all,self.f_p_i[j],a_all,self.xs,self.ys,R,xt+R/num_grid,yt)
                    #p[4] = result-(1-alpha)*sum([(xt+R/num_grid-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                    result = alpha*df.integrate_s2(self.f_v,self.beta,a_all2,self.xs,self.ys,xt+R/num_grid,yt,j)
                    p2[4] = result-(1-alpha)*sum([(xt+R/num_grid-self.x[k][j][n_iter-1]*R/num_grid)**2+(yt-self.y[k][j][n_iter-1]*R/num_grid)**2 for k in Neighbor[j]])
                ####概率归一化####
                p = p2.copy()
                try:
                    temp = exp(beta2*np.max(p))
                except:
                    print('error1')
                    p = p-np.max(p)+709/beta2
                if exp(beta2*np.max(p))==0:
                    print('error2')
                    p = p-(744+np.min(p))/beta2
# =============================================================================
#                 p = np.exp(beta2*p)
#                 p = p*(1-np.array([0]+sign[0]))
#                 p = p/np.sum(p)
# =============================================================================
                
                p = np.exp(beta2*p)
                p = p*(1-np.array([0]+sign[0]))
                p = p/np.sum(p)
                
                #print(p)
                
                ####以p概率随机选择一个方向####
                sum_p = 0
                rand = random.uniform(0, 1)
                for p_i in range(5):
                    sum_p = sum_p+p[p_i]
                    if sum_p>rand:
                        if p_i==0:
                            self.x[i][j][n_iter],self.y[i][j][n_iter] = self.x[i][j][n_iter-1],self.y[i][j][n_iter-1]
                        elif p_i==1:
                            self.x[i][j][n_iter],self.y[i][j][n_iter] = self.x[i][j][n_iter-1],self.y[i][j][n_iter-1]+1
                        elif p_i==2:
                            self.x[i][j][n_iter],self.y[i][j][n_iter] = self.x[i][j][n_iter-1],self.y[i][j][n_iter-1]-1
                        elif p_i==3:
                            self.x[i][j][n_iter],self.y[i][j][n_iter] = self.x[i][j][n_iter-1]-1,self.y[i][j][n_iter-1]
                        elif p_i==4:
                            self.x[i][j][n_iter],self.y[i][j][n_iter] = self.x[i][j][n_iter-1]+1,self.y[i][j][n_iter-1]
                        break
                ####第i个个体中未被选中的，仍保持原地不动####
                for k in range(n):
                    if k!=j:
                        self.x[i][k][n_iter] = self.x[i][k][n_iter-1]
                        self.y[i][k][n_iter] = self.y[i][k][n_iter-1]
            ####计算每次迭代的phi(a)####
            dis_x = np.zeros((n,n))
            dis_y = np.zeros((n,n))
            for i in range(n):
                a_temp = [0]*n
                for j in range(n):
                    a_temp[j] = [self.x[i][j][n_iter]*R/num_grid, self.y[i][j][n_iter]*R/num_grid]
                    dis_x[i][j] = sum([(self.x[i][j][n_iter]*R/num_grid-self.x[k][j][n_iter]*R/num_grid)**2 for k in Neighbor[j]])
                    dis_y[i][j] = sum([(self.y[i][j][n_iter]*R/num_grid-self.y[k][j][n_iter]*R/num_grid)**2 for k in Neighbor[j]])
                self.Phi[i][n_iter] = df.Phi_a3(a_temp,self.f_v,self.beta,self.xs,self.ys)
            self.Phi_all[n_iter] = np.sum(self.Phi[:,n_iter-1])-np.sum(dis_x)-np.sum(dis_y)
        return self.x,self.y,self.Phi,self.Phi_all
    
    def draw(self):
        len_xs,len_ys = np.size(self.xs),np.size(self.ys)
        hot = np.zeros((len_xs,len_ys))
        for i in range(len_xs):
            for j in range(len_ys):
                hot[i][j] = self.f_v(1/len_xs*i,1/len_ys*j)
        plt.figure()
        plt.imshow(hot, extent=(0, self.R, 0, self.R),
            cmap=plt.cm.hot, norm=LogNorm())#cm.hot
        plt.colorbar()
        for i in range(self.n):
            plt.plot(self.x[i][i]*self.R/self.num_grid,self.y[i][i]*self.R/self.num_grid)
            plt.scatter(self.x[i][i][-1]*self.R/self.num_grid,self.y[i][i][-1]*self.R/self.num_grid,marker='^')
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
            plt.plot(self.x[i][i]*self.R/self.num_grid,self.y[i][i]*self.R/self.num_grid)
            plt.scatter(self.x[i][i][-1]*self.R/self.num_grid,self.y[i][i][-1]*self.R/self.num_grid,marker='^')
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
    
    def optimization_phi(self):
        ####优化工具箱计算phi(a)最优解####
        R = self.R
        n = self.n
        phi_iter = partial(df.Phi_a,f_v=self.f_v,f_p_all=self.f_p_all,xs=self.xs,ys=self.ys)
        def phi_f(x):
            len_x = int(len(x)/2)
            a = [0]*len_x
            for i in range(len_x):
                a[i] = (x[2*i],x[2*i+1])
            res = -phi_iter(a)
            return res
        x0 = [0]*2*n
        for i in range(n):
            x0[2*i] = random.uniform(0, R)
            x0[2*i+1] = random.uniform(0, R)
        str_bnds = '(0,'+str(R)+'),(0,'+str(R)+')'
        for i in range(n-1):
            str_bnds = str_bnds+',(0,'+str(R)+'),(0,'+str(R)+')'
        bnds = eval('('+str_bnds+')')
        self.opt_phi = minimize(phi_f, x0,bounds=bnds, tol=self.tol)
        ans_opt = np.zeros((n,2))
        for i in range(n):
            ans_opt[i][0] = self.opt_phi.x[2*i]
            ans_opt[i][1] = self.opt_phi.x[2*i+1]
        return ans_opt


