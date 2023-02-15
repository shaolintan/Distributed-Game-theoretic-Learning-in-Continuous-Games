# -*- coding: utf-8 -*-


import numpy as np
import continuous_dynamics as cd
import continuous_dynamics_multi_agent as cdma
import disperse_dynamics as dd
import disperse_dynamics_multi_agent as ddma

from sympy import lambdify
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import colors
import random

# =============================================================================
# '''连续，单个个体'''
# x0 = [0]*6
# y0 = [0]*6
# for i in range(6):
#     x0[i] = random.uniform(0, 0.05)
#     y0[i] = random.uniform(0, 0.05)
# mas = cd.definition(iter_max=500,n=6,beta=[5,5,5,5,5,5],x0=x0,y0=y0,lambada=[0.4,0.35,0.3,0.5,0.45,0.4])#,x0=[0,0.9,0.9,0.1,0.5,0.5],y0=[0,0.9,0.1,0.9,0,0.9]
# x1,y1,phi1 = mas.dynamics_iter()
# mas.draw_without_hot()
# '''连续，多个个体'''
# x0=np.zeros((6,6))
# y0=np.zeros((6,6))
# for i in range(6):
#     for j in range(6):
#         x0[i][j] = random.uniform(0, 0.05)
#         y0[i][j] = random.uniform(0, 0.05)
# mas2 = cdma.definition(lambada=[0.4,0.35,0.3,0.5,0.45,0.4],iter_max=500,n=6,alpha=0.9,x0=x0,y0=y0)
# x2,y2,phi2,phiall2 = mas2.dynamics_iter()
# mas2.draw_without_hot()
# =============================================================================
#'''离散，单个个体'''
#mas3 = dd.definition(iter_max=2000,n=6,initialization='zero',beta_exp = 5000)#,beta=[5,6,7,8,9,10]
#x3,y3,phi3 = mas3.dynamics_iter()
# =============================================================================
# mas3.draw()
# ans_opt = mas3.optimization_phi()
# =============================================================================

# =============================================================================
# '''离散，多个个体'''
mas4 = ddma.definition(iter_max=2000,n=6,alpha=0.9,initialization='zero',beta_exp=5000,obstacle=[
         (0.2,0.2,0.1,0.1)])
x4,y4,Phi4,Phi_all4 = mas4.dynamics_iter()
# =============================================================================
# =============================================================================
# mas4.draw_without_hot()
# mas4.draw()
# ans = mas4.optimization_phi()
# =============================================================================





# =============================================================================
# sign1 = 1
# sign2 = 0
# t = 132
# color=['Accent','Accent_r','Blues','Blues_r','BrBG','BrBG_r','BuGn','BuGn_r','BuPu','BuPu_r','CMRmap','CMRmap_r',\
# 'Dark2','Dark2_r','GnBu','GnBu_r','Greens','Greens_r','Greys','Greys_r','OrRd','OrRd_r','Oranges','Oranges_r',\
# 'PRGn','PRGn_r','Paired','Paired_r','Pastel1','Pastel1_r','Pastel2','Pastel2_r','PiYG','PiYG_r','PuBu','PuBuGn',\
# 'PuBuGn_r','PuBu_r','PuOr','PuOr_r','PuRd','PuRd_r','Purples','Purples_r','RdBu','RdBu_r','RdGy','RdGy_r','RdPu',\
# 'RdPu_r','RdYlBu','RdYlBu_r','RdYlGn','RdYlGn_r','Reds','Reds_r','Set1','Set1_r','Set2','Set2_r','Set3','Set3_r',\
# 'Spectral','Spectral_r','Wistia','Wistia_r','YlGn','YlGnBu','YlGnBu_r','YlGn_r','YlOrBr','YlOrBr_r','YlOrRd','YlOrRd_r',\
# 'afmhot','afmhot_r','autumn','autumn_r','binary','binary_r','bone','bone_r','brg','brg_r','bwr','bwr_r','cividis',\
# 'cividis_r','cool','cool_r','coolwarm','coolwarm_r','copper','copper_r','cubehelix','cubehelix_r','flag','flag_r',\
# 'gist_earth','gist_earth_r','gist_gray','gist_gray_r','gist_heat','gist_heat_r','gist_ncar','gist_ncar_r','gist_rainbow',\
# 'gist_rainbow_r','gist_stern','gist_stern_r','gist_yarg','gist_yarg_r','gnuplot','gnuplot2','gnuplot2_r','gnuplot_r',\
# 'gray','gray_r','hot','hot_r','hsv','hsv_r','icefire','icefire_r','inferno','inferno_r','jet','jet_r','magma',\
# 'magma_r','mako','mako_r','nipy_spectral','nipy_spectral_r','ocean','ocean_r','pink','pink_r','plasma','plasma_r',\
# 'prism','prism_r','rainbow','rainbow_r','rocket','rocket_r','seismic','seismic_r','spring','spring_r','summer',\
# 'summer_r','tab10','tab10_r','tab20','tab20_r','tab20b','tab20b_r','tab20c','tab20c_r','terrain','terrain_r',\
# 'twilight','twilight_r','twilight_shifted','twilight_shifted_r','viridis','viridis_r','vlag','vlag_r','winter','winter_r']
# if sign1==1:
#     x0 = [0]*6
#     y0 = [0]*6
#     for i in range(6):
#         x0[i] = random.uniform(0, 0.05)
#         y0[i] = random.uniform(0, 0.05)
#     mas = cd.definition(iter_max=500,n=6,beta=[5,5,5,10,10,10],x0=x0,y0=y0,lambada=[0.4,0.35,0.3,0.5,0.45,0.4])#,x0=[0,0.9,0.9,0.1,0.5,0.5],y0=[0,0.9,0.1,0.9,0,0.9]
#     x1,y1,phi1 = mas.dynamics_iter()
#     mas.draw_without_hot()
#     #mas.draw()
# if sign2==1:
#     zt = mas.f_v(mas.xs,mas.ys)
#     ft = lambdify([mas.xs,mas.ys],zt)
#     hot = np.zeros((mas.grids,mas.grids))
#     for i in range(mas.grids):
#         for j in range(mas.grids):
#             hot[mas.grids-i-1][j] = ft(mas.R/mas.grids*i,mas.R/mas.grids*j)
#     plt.figure()
#     clist=['white','lightblue','lightgreen','orange','red']#,'lightyellow','lightcoral'
#     newcmp = LinearSegmentedColormap.from_list('chaos',clist)
#     
#     plt.imshow(hot, extent=(0, mas.R, 0, mas.R),
#                cmap=newcmp)#, norm=LogNorm()hot_r jet seismic gist_ncar plt.cm.rainbow_r newcmp
#     plt.colorbar()
#     plt.xlim(0,mas.R)
#     plt.ylim(0,mas.R)
#     plt.show()
# =============================================================================
# =============================================================================
#     for i in range(t,t+28):
#         plt.figure()
#         exec('plt.imshow(hot, extent=(0, mas.R, 0, mas.R),cmap=plt.cm.'+color[i]+')')
#         plt.colorbar()
#         plt.xlim(0,mas.R)
#         plt.ylim(0,mas.R)
#         plt.show()
# =============================================================================



# =============================================================================
# cmap = colors.ListedColormap(['green','red','black','yellow'])
# bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
# norm = colors.BoundaryNorm(bounds, cmap.N)
# heatmap = plt.pcolor(np.array(data), cmap=cmap, norm=norm)
# plt.colorbar(heatmap, ticks=[0, 1, 2, 3])
# 
# =============================================================================





#t=0   i=2,6,8
#t=10   i=8
#t=40   i=12,24
#t=70   i=6,9,26,30






# =============================================================================
# mas3 = dd.definition(iter_max=2,n=6,initialization='zero',beta_exp = 8000,density_center=[(0.5,0.5)])
# fig = plt.figure()
# ax = Axes3D(fig)
# x = np.arange(0, 1, 0.01)
# y = np.arange(0, 1, 0.01)
# X, Y = np.meshgrid(x, y) # 网格的创建，生成二维数组
# #X=np.array(x)
# #Y=np.array(y)
# #print(type(X),X)
# #Z = np.sin(X) * np.cos(Y)
# #Z=X4+Y2
# #Z=-X2-Y2
# #Z=2X+2Y
# Z = np.zeros((100,100))
# for i in range(100):
#     for j in range(100):
#         Z[i][j] = mas3.f_v(x[i],y[j])
# #print(type(Z),Z)
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
# plt.show()
# =============================================================================







