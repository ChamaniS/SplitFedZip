import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

CR_AE_S1 = np.array([63209.9,63209.9,51717.2,28444.4,16254,14586.9,11302.4,4309.76,1879.59,1301.81,740.419,662.526,300.258])
CR_cheng_S1 = np.array([17964.9,17964.9,17964.9,17964.9,15515.2,15238.1,14341.7,11010.8,4224.42,3468.83,2717.2,2613.58,2873.18])
CR_AE_S2 = np.array([63209.9,63209.9,35555.6,19176,12832.1,10733.8,9922.48,7111.11,2973.29,1605.52,1135.51,917.563,380.952])
CR_cheng_S2 = np.array([18156,17964.9,17964.9,17964.9,16100.6,15515.2,14970.8,11454.1,6881.72,5669.99,4432.9,4675.8,4132.36])
CR_AE_total = np.array([31604.9,31604.9,21070,11454.1,7170.87,6206.06,5300.21,2683.44,1151.6,718.899,448.179,384.731,167.913])
CR_cheng_total = np.array([9029.98,8982.46,8982.46,8982.46,7901.23,7687.69,7324.75,5614.04,2617.59,2152.16,1684.76,1676.49,1694.8])
lambdas = np.array([0.0002,0.002,0.05,0.2,0.8,0.6,1,4,16,32,64,100,10000000000])
xaxis = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

plt.figure(figsize=(10, 6))
plt.plot(lambdas, CR_AE_S1, marker='o', color='g', label='AE_(S1)')
plt.plot(lambdas, CR_cheng_S1, marker='s',color='m', label='Cheng_2020_attn_(S1)')

plt.plot(lambdas, CR_AE_S2, marker='x', color='y', label='AE_(S2)')
plt.plot(lambdas, CR_cheng_S2, marker='p',color='orange', label='Cheng_2020_attn_(S2)')

plt.plot(lambdas, CR_AE_total, marker='*', color='k', label='AE_(S1+S2)')
plt.plot(lambdas, CR_cheng_total, marker='D',color='r', label='Cheng_2020_attn_(S1+S2)')

plt.title('CR: FG both compression')
plt.xlabel('Lambda ($\lambda$)')
plt.ylabel('CR')
plt.xscale('log')  # Using log scale for x-axis as the lambda values vary widely
plt.legend()
plt.show()