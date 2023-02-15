# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:27:22 2022

@author: sean
"""

import matplotlib.pyplot as plt
import numpy as np


e_1=np.load('e_1.npy')
e_2=np.load('e_2.npy')
e_3=np.load('e_3.npy')
e_4=np.load('e_4.npy')
y_1=np.load('y_1.npy')
y_2=np.load('y_2.npy')
z1=np.load('z1.npy')
z2=np.load('z2.npy')

#plt.figure(2)
#ax=plt.subplot(2,1,1)
#y_1=y_1[:200,:]
#plt.plot(range(len(y_1)),y_1[:,0],'r-',marker='o',markevery=10,label="$x_1$")
#plt.plot(range(len(y_1)),y_1[:,1],'b-',marker='^',markevery=10,label="$x_2$")
#plt.plot(range(len(y_1)),y_1[:,2],'c-',marker='s',markevery=10,label="$x_3$")
#plt.plot(range(len(y_1)),y_1[:,3],'k-',marker='*',markevery=10,label="$x_4$")
#plt.plot(range(len(y_1)),y_1[:,4],'g-',marker='p',markevery=10,label="$x_5$")
#plt.ylabel('Action profile ($x$)',family='Arial',fontweight="heavy",fontsize="12")
#plt.text(-8,6,'(a)',family='Arial',fontweight="heavy",fontsize="12")
#plt.legend(loc=1)
#
#ax=plt.subplot(2,1,2)
#y_2=y_2[:200,:]
#plt.plot(range(len(y_2)),y_2[:,0],'r-',marker='o',markevery=10,label="$x_1$")
#plt.plot(range(len(y_2)),y_2[:,1],'b-',marker='^',markevery=10,label="$x_2$")
#plt.plot(range(len(y_2)),y_2[:,2],'c-',marker='s',markevery=10,label="$x_3$")
#plt.plot(range(len(y_2)),y_2[:,3],'k-',marker='*',markevery=10,label="$x_4$")
#plt.plot(range(len(y_2)),y_2[:,4],'g-',marker='p',markevery=10,label="$x_5$")
#
#plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
#plt.ylabel('Action profile ($x$)',family='Arial',fontweight="heavy",fontsize="12")
#plt.text(-8,8.5,'(b)',family='Arial',fontweight="heavy",fontsize="12")
#plt.legend(loc=1)

#plt.figure(2)
#ax=plt.subplot(2,1,1)
#z1=z1[:1000,:,:]
#plt.plot(range(len(z1)),z1[:,0,0],'r-',marker='o',markevery=50,label="$z^1_1$")
#plt.plot(range(len(z1)),z1[:,1,1],'b-',marker='^',markevery=50,label="$z^2_2$")
#plt.plot(range(len(z1)),z1[:,2,2],'c-',marker='s',markevery=50,label="$z^3_3$")
#plt.plot(range(len(z1)),z1[:,3,3],'k-',marker='*',markevery=50,label="$z^4_4$")
#plt.plot(range(len(z1)),z1[:,4,4],'g-',marker='p',markevery=50,label="$z^5_5$")
#for i in range(5):
#    for j in range(5):
#        if i!=j:
#            plt.plot(range(len(z1)),z1[:,i,j],'--')
#plt.ylabel('State ($z$)',family='Arial',fontweight="heavy",fontsize="12")
#plt.text(-40,6,'(a)',family='Arial',fontweight="heavy",fontsize="12")
#plt.legend(loc=1)
#
#ax=plt.subplot(2,1,2)
#z2=z2[:1000,:,:]
#plt.plot(range(len(z2)),z2[:,0,0],'r-',marker='o',markevery=50,label="$z^1_1$")
#plt.plot(range(len(z2)),z2[:,1,1],'b-',marker='^',markevery=50,label="$z^2_2$")
#plt.plot(range(len(z2)),z2[:,2,2],'c-',marker='s',markevery=50,label="$z^3_3$")
#plt.plot(range(len(z2)),z2[:,3,3],'k-',marker='*',markevery=50,label="$z^4_4$")
#plt.plot(range(len(z2)),z2[:,4,4],'g-',marker='p',markevery=50,label="$z^5_5$")
#for i in range(5):
#    for j in range(5):
#        if i!=j:
#            plt.plot(range(len(z2)),z2[:,i,j],'--')
#
#plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
#plt.ylabel('State ($z$)',family='Arial',fontweight="heavy",fontsize="12")
#plt.text(-40,6,'(b)',family='Arial',fontweight="heavy",fontsize="12")
#plt.legend(loc=1)

e_1=e_1[:500]
e_2=e_2[:3000]
e_4=e_4[:3000]
plt.loglog(range(len(e_1)),e_1,'r-',marker='o',markevery=30,label="full information & fixed step size")
plt.loglog(range(len(e_2)),e_2,'b-',marker='^',markevery=60,label="full information & diminishing step size")
plt.loglog(range(len(e_3)),e_3,'c-',marker='s',markevery=80,label="partial information & cycle")
plt.loglog(range(len(e_4)),e_4,'k-',marker='*',markevery=60,label="partial information & directed cycle")
plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")
plt.legend()
