# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:50:20 2022

@author: sean
"""
import numpy as np
import matplotlib.pyplot as plt


#e0=np.load('e_00.npy')
#e1=np.load('e_01.npy')
#e2=np.load('e_02.npy')
#e3=np.load('e2.npy')
#e4=np.load('e8.npy')
#
#e0=e0[:2000]
#e1=e1[:60000]
#e3=e3[:30000]

#e0=np.load('e_0.npy')
#e1=np.load('e_1.npy')
#e2=np.load('e_3.npy')
#e3=np.load('e_2.npy')
#e4=np.load('e_7.npy')
#
#e0=e0[:2000]
#e1=e1[:15000]
#e2=e2[:400000]
#e3=e3[:3000]
#e4=e4[:1000]

e0=np.load('e5_1.npy')
e1=np.load('e5_2.npy')
e2=np.load('e5_3.npy')
e3=np.load('e5_4.npy')
e4=np.load('e5_5.npy')

e0=e0[:3500]
e1=e1[:2500]
e2=e2[:8000]
e3=e3[:5400]
e4=e4[:2900]

e_c=np.load('e_0.npy')
e_c=e_c[:2000]

plt.figure(1)
ax=plt.subplot(1,2,1)
#plt.loglog(range(1,len(e0)+1),e0/e0[0],'r--',marker='o',markevery=100,label="Centralized approach")
#plt.loglog(range(1,len(e1)+1),e1/e1[0],'b--',marker='s',markevery=800,label="Consensus-based approach")
#plt.loglog(range(1,len(e2)+1),e2/e2[0],'c--',marker='^',markevery=8000,label="Augmented game approach")
#plt.loglog(range(1,len(e3)+1),e3/e3[0],'k--',marker='p',markevery=100,label="Variant of consensus-based approach")
#plt.loglog(range(1,len(e4)+1),e4/e4[0],'m--',marker='*',markevery=100,label="Nesterov’s accelerated approach")
plt.loglog(range(1,len(e_c)+1),e_c/e_c[0],'g-')
plt.loglog(range(1,len(e0)+1),e0/e0[0],'r--',marker='o',markevery=150,label="Cycle graph")
plt.loglog(range(1,len(e1)+1),e1/e1[0],'b--',marker='s',markevery=150,label="Directed cycle")
plt.loglog(range(1,len(e2)+1),e2/e2[0],'c--',marker='^',markevery=200,label="Path graph")
plt.loglog(range(1,len(e3)+1),e3/e3[0],'k--',marker='p',markevery=150,label="Star graph")
plt.loglog(range(1,len(e4)+1),e4/e4[0],'m--',marker='*',markevery=150,label="Wheel graph")
plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")
plt.text(1,0.1,'(a)',family='Arial',fontweight="heavy",fontsize="12")
plt.legend(loc=3)


#e81=np.load('e8_1.npy')
#e82=np.load('e8_2.npy')
#e83=np.load('e8_3.npy')
#e84=np.load('e8_4.npy')
#e85=np.load('e8_5.npy')
#e85=e85[:6000]
#e81=np.load('e9_1.npy')
#e82=np.load('e9_2.npy')
#e83=np.load('e9_3.npy')
#e84=np.load('e9_4.npy')
#e81=e81[:2000]
#e82=e82[:2000]
#e83=e83[:1500]
#e84=e84[:700]
#e85=np.load('e9_5.npy')
#e85=e85[:6000]

e81=np.load('e6_1.npy')
e82=np.load('e6_2.npy')
e83=np.load('e6_3.npy')
e84=np.load('e6_4.npy')
e85=np.load('e6_5.npy')
e81=e81[:1800]
e82=e82[:1300]
e83=e83[:3900]
e84=e84[:2500]
e85=e85[:1600]

ax=plt.subplot(1,2,2)
#plt.loglog(range(1,len(e81)+1),e81/e81[0],'r--',marker='o',markevery=100,label=r'$\beta=0.1$')
#plt.loglog(range(1,len(e82)+1),e82/e82[0],'b--',marker='s',markevery=100,label=r'$\beta=0.3$')
#plt.loglog(range(1,len(e83)+1),e83/e83[0],'c--',marker='^',markevery=100,label=r'$\beta=0.5$')
#plt.loglog(range(1,len(e84)+1),e84/e84[0],'k--',marker='p',markevery=100,label=r'$\beta=0.7$')
#plt.loglog(range(1,len(e85)+1),e85/e85[0],'m--',marker='*',markevery=300,label=r'$\beta=0.8$')
plt.loglog(range(1,len(e_c)+1),e_c/e_c[0],'g-')
plt.loglog(range(1,len(e81)+1),e81/e81[0],'r--',marker='o',markevery=100,label="Cycle graph")
plt.loglog(range(1,len(e82)+1),e82/e82[0],'b--',marker='s',markevery=100,label="Directed cycle")
plt.loglog(range(1,len(e83)+1),e83/e83[0],'c--',marker='^',markevery=100,label="Path graph")
plt.loglog(range(1,len(e84)+1),e84/e84[0],'k--',marker='p',markevery=100,label="Star graph")
plt.loglog(range(1,len(e85)+1),e85/e85[0],'m--',marker='*',markevery=100,label="Wheel graph")
plt.xlabel('Iterations ($t$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")
plt.text(1,0.1,'(b)',family='Arial',fontweight="heavy",fontsize="12")
plt.legend(loc=3)