# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 13:13:56 2022

@author: sean
"""

import matplotlib.pyplot as plt
import numpy as np

##########################figure_1###############################
#e0=np.load('e0.npy')
#e1=np.load('e1.npy')
#e2=np.load('e2.npy')
#e3=np.load('e3.npy')
#e4=np.load('e4.npy')
#e5=np.load('e5.npy')
#e6=np.load('e6.npy')
#e7=np.load('e7.npy')
#e8=np.load('e8.npy')
#
#e6=e6[:3000]
#e7=e7[:3000]
#e8=e8[:3000]
#
#plt.figure(1)
#plt.loglog(range(1,len(e0)+1),e0/e0[0],'r--',marker='o',markevery=500,label="Cycle & $\lambda(t)=1/(t+1)$")
#plt.loglog(range(1,len(e1)+1),e1/e1[0],'b--',marker='s',markevery=500,label="Directed cycle & $\lambda(t)=1/(t+1)$")
#plt.loglog(range(1,len(e2)+1),e2/e2[0],'k--',marker='^',markevery=500,label="Switching directed cycle & $\lambda(t)=1/(t+1)$")
#plt.loglog(range(1,len(e3)+1),e3/e3[0],'r-',marker='<',markevery=500,label="Cycle & $\lambda(t)=0.001$")
#plt.loglog(range(1,len(e4)+1),e4/e4[0],'b-',marker='>',markevery=500,label="Directed cycle & $\lambda(t)=0.001$")
#plt.loglog(range(1,len(e5)+1),e5/e5[0],'k-',marker='*',markevery=500,label="Switching directed cycle & $\lambda(t)=0.001$")
#plt.loglog(range(1,len(e6)+1),e6/e6[0],'r:',marker='p',markevery=200,label="Cycle & $\lambda(t)=0.01$")
#plt.loglog(range(1,len(e7)+1),e7/e7[0],'b:',marker='d',markevery=200,label="Directed cycle & $\lambda(t)=0.01$")
#plt.loglog(range(1,len(e8)+1),e8/e8[0],'k:',marker='h',markevery=200,label="Switching directed cycle & $\lambda(t)=0.01$")
#
#plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
#plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")
#plt.legend(loc=3)


##########################figure_2###############################
#e0=np.load('e_0.npy')
#e1=np.load('e_1.npy')
#e2=np.load('e_2.npy')
#e3=np.load('e_3.npy')
#e4=np.load('e_4.npy')
#e5=np.load('e_5.npy')
#
#e0=e0[:1500]
#e1=e1[:1800]
#e2=e2[:2000]
#e3=e3[:2200]
#e4=e4[:2200]
#e5=e5[:2200]
#
#plt.figure(1)
#plt.loglog(range(1,len(e0)+1),e0/e0[0],'r--',marker='o',markevery=200,label="Cycle & $\lambda(t)=1/(t+1)$")
#plt.loglog(range(1,len(e1)+1),e1/e1[0],'b--',marker='s',markevery=200,label="Directed cycle & $\lambda(t)=1/(t+1)$")
#plt.loglog(range(1,len(e2)+1),e2/e2[0],'c--',marker='^',markevery=200,label="Switching directed cycle & $\lambda(t)=1/(t+1)$")
#plt.loglog(range(1,len(e3)+1),e3/e3[0],'r-',marker='p',markevery=200,label="Cycle & $\lambda(t)=0.001$")
#plt.loglog(range(1,len(e4)+1),e4/e4[0],'b-',marker='d',markevery=200,label="Directed cycle & $\lambda(t)=0.001$")
#plt.loglog(range(1,len(e5)+1),e5/e5[0],'c-',marker='*',markevery=200,label="Switching directed cycle & $\lambda(t)=0.001$")
#plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
#plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")
#plt.legend(loc=3)

##########################figure_3###############################
e0=np.load('e_00.npy')
e1=np.load('e_01.npy')
e2=np.load('e_02.npy')
e3=np.load('e6.npy')

e0=e0[:2000]
e1=e1[:60000]
e3=e3[:3000]

plt.figure(1)
ax=plt.subplot(1,2,1)
plt.loglog(range(1,len(e0)+1),e0/e0[0],'r--',marker='o',markevery=100,label="Centralized approach")
plt.loglog(range(1,len(e1)+1),e1/e1[0],'b--',marker='s',markevery=5000,label="Consensus-based approach")
plt.loglog(range(1,len(e2)+1),e2/e2[0],'c--',marker='^',markevery=5000,label="Augmented game approach")
plt.loglog(range(1,len(e3)+1),e3/e3[0],'k--',marker='p',markevery=100,label="Timestamp-based approach")
plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")
plt.text(1,0.01,'(a)',family='Arial',fontweight="heavy",fontsize="12")
plt.legend(loc=3)


e4=np.load('e_03.npy')
e5=np.load('e_04.npy')
e6=np.load('e_05.npy')
e7=np.load('e0.npy')
e4=e4[:20000]
e6=e6[:100001]

ax=plt.subplot(1,2,2)
plt.loglog(range(1,len(e4)+1),e4/e4[0],'r--',marker='o',markevery=100,label="Centralized approach")
plt.loglog(range(1,len(e5)+1),e5/e5[0],'b--',marker='s',markevery=5000,label="Consensus-based approach")
plt.loglog(range(1,len(e6)+1),e6/e6[0],'c--',marker='^',markevery=5000,label="Augmented game approach")
plt.loglog(range(1,len(e7)+1),e7/e7[0],'k--',marker='p',markevery=100,label="Timestamp-based approach")
plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")
plt.text(1,0.4,'(b)',family='Arial',fontweight="heavy",fontsize="12")
plt.legend(loc=3)