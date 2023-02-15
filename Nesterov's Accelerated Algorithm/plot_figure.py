# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:39:35 2022

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

e0=np.load('e_1.npy')
e1=np.load('e_2.npy')
e2=np.load('e_3.npy')
e3=np.load('e_4.npy')
e4=np.load('e_2_1.npy')
e5=np.load('e_2_2.npy')

e0=e0[:30000]
e1=e1[:9000]
e2=e2[:30000]
e3=e3[:70000]
e4=e4[:14000]
e5=e5[:21000]

#e0=np.load('e_v1.npy')
#e1=np.load('e_v2.npy')
#e2=np.load('e_v3.npy')
#e3=np.load('e_v4.npy')
#e4=np.load('e_v2_1.npy')
#e5=np.load('e_v2_2.npy')
#
#e0=e0[:1500]
#e1=e1[:400]
#e2=e2[:1500]
#e3=e3[:3200]
#e4=e4[:700]
#e5=e5[:1000]


plt.figure(1)
ax=plt.subplot(1,2,1)
plt.loglog(range(1,len(e0)+1),e0,'r--',marker='o',markevery=1000,label="Projected gradient method")
plt.loglog(range(1,len(e2)+1),e2,'b--',marker='s',markevery=2000,label="Reflected projected method")
plt.loglog(range(1,len(e3)+1),e3,'c--',marker='^',markevery=3000,label="Golden ration method")
plt.loglog(range(1,len(e5)+1),e5,'k--',marker='p',markevery=500,label=r'Our algorithm with $\beta=0.3$')
plt.loglog(range(1,len(e4)+1),e4,'m--',marker='*',markevery=600,label=r'Our algorithm with $\beta=0.5$')
plt.loglog(range(1,len(e1)+1),e1,'g--',marker='>',markevery=800,label=r'Our algorithm with $\beta=0.7$')
#plt.loglog(range(1,len(e0)+1),e0/e0[0],'r--',marker='o',markevery=150,label="Cycle graph")
#plt.loglog(range(1,len(e1)+1),e1/e1[0],'b--',marker='s',markevery=150,label="Directed cycle")
#plt.loglog(range(1,len(e2)+1),e2/e2[0],'c--',marker='^',markevery=200,label="Path graph")
#plt.loglog(range(1,len(e3)+1),e3/e3[0],'k--',marker='p',markevery=150,label="Star graph")
#plt.loglog(range(1,len(e4)+1),e4/e4[0],'m--',marker='*',markevery=150,label="Wheel graph")
plt.xlabel('Iterations ($k$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Convergence Metric',family='Arial',fontweight="heavy",fontsize="12")
plt.text(1,100000,'(a)',family='Arial',fontweight="heavy",fontsize="12")
#plt.text(1,0.0001,'(a)',family='Arial',fontweight="heavy",fontsize="12")
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

e00=np.load('e_10.npy')
e10=np.load('e_20.npy')
e20=np.load('e_30.npy')
e30=np.load('e_40.npy')
e40=np.load('e_20_1.npy')
e50=np.load('e_20_2.npy')
e00=e00[:1100]
e10=e10[:300]
e20=e20[:1100]
e30=e30[:2800]
e40=e40[:600]
e50=e50[:800]

#e00=np.load('e_v10.npy')
#e10=np.load('e_v20.npy')
#e20=np.load('e_v30.npy')
#e30=np.load('e_v40.npy')
#e40=np.load('e_v20_1.npy')
#e50=np.load('e_v20_2.npy')
#e00=e00[:1500]
#e10=e10[:400]
#e20=e20[:1500]
#e30=e30[:3800]
#e40=e40[:700]
#e50=e50[:1000]

ax=plt.subplot(1,2,2)
#plt.loglog(range(1,len(e81)+1),e81/e81[0],'r--',marker='o',markevery=100,label=r'$\beta=0.1$')
#plt.loglog(range(1,len(e82)+1),e82/e82[0],'b--',marker='s',markevery=100,label=r'$\beta=0.3$')
#plt.loglog(range(1,len(e83)+1),e83/e83[0],'c--',marker='^',markevery=100,label=r'$\beta=0.5$')
#plt.loglog(range(1,len(e84)+1),e84/e84[0],'k--',marker='p',markevery=100,label=r'$\beta=0.7$')
#plt.loglog(range(1,len(e85)+1),e85/e85[0],'m--',marker='*',markevery=300,label=r'$\beta=0.8$')
plt.loglog(range(1,len(e00)+1),e00,'r--',marker='o',markevery=100,label="Projected gradient method")
plt.loglog(range(1,len(e20)+1),e20,'b--',marker='s',markevery=200,label="Reflected projected method")
plt.loglog(range(1,len(e30)+1),e30,'c--',marker='^',markevery=200,label="Golden ration method")
plt.loglog(range(1,len(e50)+1),e50,'k--',marker='p',markevery=40,label=r'Our algorithm with $\beta=0.3$')
plt.loglog(range(1,len(e40)+1),e40,'m--',marker='*',markevery=40,label=r'Our algorithm with $\beta=0.5$')
plt.loglog(range(1,len(e10)+1),e10,'g--',marker='>',markevery=20,label=r'Our algorithm with $\beta=0.7$')

#plt.loglog(range(1,len(e0)+1),e0/e0[0],'r--',marker='o',markevery=150,label="Cycle graph")
#plt.loglog(range(1,len(e1)+1),e1/e1[0],'b--',marker='s',markevery=150,label="Directed cycle")
#plt.loglog(range(1,len(e2)+1),e2/e2[0],'c--',marker='^',markevery=200,label="Path graph")
#plt.loglog(range(1,len(e3)+1),e3/e3[0],'k--',marker='p',markevery=150,label="Star graph")
#plt.loglog(range(1,len(e4)+1),e4/e4[0],'m--',marker='*',markevery=150,label="Wheel graph")
plt.xlabel('Iterations ($k$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Convergence Metric',family='Arial',fontweight="heavy",fontsize="12")
#plt.text(1,0.0001,'(b)',family='Arial',fontweight="heavy",fontsize="12")
plt.text(1,10000,'(b)',family='Arial',fontweight="heavy",fontsize="12")
plt.legend(loc=3)
