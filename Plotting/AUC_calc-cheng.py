import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt

miou_nc = np.array([0.892,0.892,0.892,0.892,0.892,0.892,0.892,0.892,0.892,0.892,0.892,0.892,0.892])
miou_G = np.array([0.114,0.895,0.908,0.884,0.891,0.893,0.886,0.897,0.903,0.885,0.897,0.905,0.888])
miou_F = np.array([0.074,0.629,0.649,0.762,0.784,0.799,0.807,0.808,0.822,0.844,0.836,0.839,0.849])
miou_FG = np.array([0.0027,0.686,0.696,0.696,0.702,0.764,0.779,0.771,0.791,0.818,0.822,0.819,0.796])
lambdas = np.array([0.000000002,0.0002,0.002,0.05,0.2,0.6,0.8,1,4,32,64,100,10000000000])
xaxis = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

# 1. Area Under the Curve (AUC)
auc1 = simps(miou_nc, xaxis)
auc2 = simps(miou_G, xaxis)
auc_difference1 = auc1 - auc2
print(f"AUC Difference 1: {auc_difference1}")

auc2 = simps(miou_G, xaxis)
auc3 = simps(miou_F, xaxis)
auc_difference2 = auc2 - auc3
print(f"AUC Difference 2 : {auc_difference2}")

auc3 = simps(miou_F, xaxis)
auc4 = simps(miou_FG, xaxis)
auc_difference3 = auc3 - auc4
print(f"AUC Difference 3 : {auc_difference3}")

auc_difference4 = auc1 - auc3
print(f"AUC Difference 4 : {auc_difference4}")

auc_difference5 = auc1 - auc4
print(f"AUC Difference 5 : {auc_difference5}")

plt.figure(figsize=(10, 6))
plt.plot(lambdas, miou_nc, marker='o', color='g', label='miou_nc')
plt.plot(lambdas, miou_G, marker='s',color='orange', label='miou_G')
plt.plot(lambdas, miou_F, marker='^',color='blue', label='miou_F')
plt.plot(lambdas, miou_FG, marker='d', color='red', label='miou_FG')
plt.title('mIoU variation for different F & G compression(Cheng_2020)')
plt.xlabel('Lambda ($\lambda$)')
plt.ylabel('mIoU')
plt.xscale('log')  # Using log scale for x-axis as the lambda values vary widely
plt.legend()
plt.show()