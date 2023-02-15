# -*- coding: utf-8 -*-
"""
Created on Thu May 12 09:28:13 2022

@author: sean
"""

import numpy as np
import matplotlib.pyplot  as plt

e0=np.load('e0.npy')
e1=np.load('e1.npy')
e2=np.load('e2.npy')
e3=np.load('e3.npy')
e5=np.load('e5.npy')
e6=np.load('e6.npy')
e7=np.load('e7.npy')
e8=np.load('e8.npy')

e_0=np.load('e_0.npy')
e_1=np.load('e_1.npy')
e_2=np.load('e_2.npy')
e_3=np.load('e_3.npy')
e_5=np.load('e_5.npy')
e_6=np.load('e_6.npy')
e_7=np.load('e_7.npy')
e_8=np.load('e_8.npy')


plt.figure()

ax=plt.subplot(2,1,1)
plt.loglog(range(1,20000+2),e0/e0[0],'r--',label="A0 ($k=2$)")
plt.loglog(range(1,20000+2),e1/e1[0],'b--',label="A1 ($k=2$)")
plt.loglog(range(1,20000+2),e2/e2[0],'c--',label="A2 ($k=2$)")
plt.loglog(range(1,20000+2),e3/e3[0],'k--',label="A3 ($k=2$)")
plt.loglog(range(1,8001),e_0[:8000]/e_0[0],'r-',label="A0 ($k=4$)")
plt.loglog(range(1,8001),e_1[:8000]/e_1[0],'b-',label="A1 ($k=4$)")
plt.loglog(range(1,8001),e_2[:8000]/e_2[0],'c-',label="A2 ($k=4$)")
plt.loglog(range(1,8001),e_3[:8000]/e_3[0],'k-',label="A3 ($k=4$)")
plt.legend(loc=3)
plt.xlabel('Iterations ($k$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")

ax=plt.subplot(2,1,2)
iterations = 100000

plt.loglog(range(1,iterations+2),e8/e8[0],'r--',label="C0 ($k=2$)")
plt.loglog(range(1,iterations+2),e5/e5[0],'b--',label="C1 ($k=2$)")
plt.loglog(range(1,iterations+2),e6/e6[0],'c--',label="C2 ($k=2$)")
plt.loglog(range(1,iterations+2),e7/e7[0],'k--',label="C3 ($k=2$)")

plt.loglog(range(1,50001),e_8[:50000]/e_8[0],'r-',label="C0 ($k=4$)")
plt.loglog(range(1,50001),e_5[:50000]/e_5[0],'b-',label="C1 ($k=4$)")
plt.loglog(range(1,50001),e_6[:50000]/e_6[0],'c-',label="C2 ($k=4$)")
plt.loglog(range(1,50001),e_7[:50000]/e_7[0],'k-',label="C3 ($k=4$)")
plt.legend(loc=3)
plt.xlabel('Iterations ($k$)',family='Arial',fontweight="heavy",fontsize="12")
plt.ylabel('Relative Error',family='Arial',fontweight="heavy",fontsize="12")