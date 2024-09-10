import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

CR_AE_S1 = np.array([63209.87,60952.38,51717.17,51717.17,21880.34,9808.42,4245.44,3504.45,3244.61,1502.34,604.77,452.33,412.14,296.24,357.94])
CR_cheng_S1 = np.array([17964.9,17964.9,17964.9,17964.9,17066.7,15515.2,11689.5,10801.7,10098.6,5746.35,2266.49,2174.1,2171.33,1977.6,1946.03])
CR_AE_S2 = np.array([68266.7,68266.7,68266.7,68266.7,24037.6,12642,8325.2,7231.64,6994.54,3734.5,1284.17,803.894,665.627,298.368,314.014])
CR_cheng_S2 = np.array([17964.9,17964.9,17964.9,17964.9,17594.5,15950.2,14222.2,13438.3,12929.3,8325.2,4063.49,3366.21,2988.91,2868.35,2654.22])
CR_AE_total = np.array([32820.5,32201.3,29425.3,29425.3,11454.1,5523.19,2811.64,2239.72,2334.7,1071.35,411.146,289.462,254.536,148.651,167.271])
CR_cheng_total = np.array([8982.46,8982.46,8982.46,8982.46,8663.28,7864.82,6416.04,5988.3,5669.99,3399.73,1454.96,1319.93,1258.6,1177.01,1150.04])
lambdas = np.array([0.000000002,0.0000002,0.0002,0.002,0.05,0.2,0.6,0.8,1,4,32,64,100,10000000,10000000000])

plt.figure(figsize=(10, 6))
plt.plot(lambdas, CR_AE_S1, marker='o', color='g', label='AE_(S1)')
plt.plot(lambdas, CR_cheng_S1, marker='s',color='m', label='Cheng_2020_attn_(S1)')
plt.plot(lambdas, CR_AE_S2, marker='x', color='y', label='AE_(S2)')
plt.plot(lambdas, CR_cheng_S2, marker='p',color='orange', label='Cheng_2020_attn_(S2)')
plt.plot(lambdas, CR_AE_total, marker='*', color='k', label='AE_(S1+S2)')
plt.plot(lambdas, CR_cheng_total, marker='D',color='r', label='Cheng_2020_attn_(S1+S2)')

plt.title('CR: F only compression')
plt.xlabel('Lambda ($\lambda$)')
plt.ylabel('CR')
plt.xscale('log')  # Using log scale for x-axis as the lambda values vary widely
plt.legend()
plt.show()

