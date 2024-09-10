import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Define a function to format the y-axis labels in scientific notation
def scientific_formatter(x, pos):
    return f'{x:.0e}'

# Data for the first plot
FCR_AE_S1 = np.array([2560000, 2560000, 1706666.667, 1024000, 1024000, 182857.1429, 98461.53846, 50693.06931, 73142.85714, 40314.96063, 35804.1958, 5831.43508, 5452.609159, 1361.702128, 1086.587436, 1092.150171])
FCR_cheng_S1 = np.array([53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684, 52244.89796, 47850.46729, 46545.45455, 44912.2807, 40314.96063, 19768.33977, 16305.73248, 11252.74725, 7781.155015, 6844.919786, 6400])
FCR_AE_S2 = np.array([2560000, 2560000, 2560000, 2560000, 853333.3333, 341333.3333, 182857.1429, 119069.7674, 160000, 85333.33333, 86779.66102, 23813.95349, 17123.74582, 1685.319289, 1343.126967, 1255.51741])
FCR_cheng_S2 = np.array([53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684,53894.73684, 52783.50515, 52244.89796, 52244.89796, 50693.06931, 44521.73913, 37372.26277, 35804.1958, 23703.7037, 22260.86957, 22959.64126])
FCR_AE_total = np.array([1280000, 1280000, 1024000, 731428.5714, 465454.5455, 119069.7674, 64000, 35555.55556, 50196.07843, 27379.67914, 25346.53465, 4684.354986, 4135.702746, 753.1626949, 600.6569686, 584.0748346])
FCR_cheng_total = np.array([26947.36842, 26947.36842, 26947.36842, 26947.36842, 26947.36842, 26528.49741, 25098.03922, 24615.38462, 24150.9434, 22456.14035, 11934.73193, 12929.29293, 8561.87291, 5235.173824, 5858.12357, 5004.887586])
Flambdas = np.array([0.000000002,0.0000002,0.0002,0.002,0.05,0.2,0.6,0.8,1,4,32,64,100,10000,10000000,10000000000])
Fxaxis = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

# Data for the second plot
FGCR_AE_S1 = np.array([2560000, 2560000, 2560000, 2560000, 2560000, 2560000, 2560000, 116363.6364, 93090.90909, 44137.93103, 11203.50109, 6701.570681, 4767.225326, 1943.811693, 1669.928245, 1653.74677])
FGCR_cheng_S1 = np.array([53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684,39384.61538, 22068.96552, 15609.7561, 16050.15674, 7722.473605, 7052.341598, 6781.456954])
FGCR_AE_S2 = np.array([2560000, 2560000, 2560000, 2560000, 2560000, 2560000, 2133333.333, 176551.7241, 142222.2222, 78769.23077, 38208.95522, 19393.93939, 16410.25641, 3651.92582, 3104.912068, 2831.858407])
FGCR_cheng_S2 = np.array([53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684, 53894.73684,53894.73684, 48761.90476, 39083.96947, 27826.08696, 26528.49741, 20562.249, 20157.48031, 19541.98473])
FGCR_AE_total = np.array([1280000, 1280000, 1280000, 1280000, 1280000, 1280000, 1163636.364, 70136.9863, 56263.73626, 28287.29282, 8663.282572, 5701.55902, 3826.606876, 1268.582755, 1085.896076, 1044.045677])
FGCR_cheng_total = np.array([26947.36842, 26947.36842, 26947.36842, 26947.36842, 26947.36842, 26947.36842, 26947.36842, 26947.36842, 26947.36842,21787.23404, 14104.6832, 10000, 10000, 5614.035088, 5224.489796, 5034.414946])
FGlambdas = np.array([0.000000002,0.0000002,0.0002,0.002,0.05,0.2,0.6,0.8,1,4,32,64,100,10000,10000000,10000000000])
FGxaxis = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# First subplot
axs[0].plot(Flambdas, FCR_AE_S1, label='AE_(S1)', color='g', marker='o')
axs[0].plot(Flambdas, FCR_cheng_S1, label='Cheng_AT_(S1)', color='m', marker='s')
axs[0].plot(Flambdas, FCR_AE_S2, label='AE_(S2)', color='y', marker='x')
axs[0].plot(Flambdas, FCR_cheng_S2, label='Cheng_AT_(S2)', color='orange', marker='p')
axs[0].plot(Flambdas, FCR_AE_total, label='AE_(S1+S2)', color='k', marker='*')
axs[0].plot(Flambdas, FCR_cheng_total, label='Cheng_AT(S1+S2)', color='r', marker='D')
axs[0].set_xlabel('Lambda ($\lambda$)', fontsize=15, color='black')  # Font size and color
axs[0].set_ylabel('CR', fontsize=15, color='black')  # Font size and color)
axs[0].set_xscale('log')  # Log scale for x-axis
axs[0].set_title('(i) F', fontsize=16, fontweight='bold', color='black')  # Bold title, size, and color)
axs[0].legend()
axs[0].grid(True)
axs[0].tick_params(axis='both', labelsize=12, colors='black')  # Font size for tick labels
axs[0].yaxis.set_major_formatter(FuncFormatter(scientific_formatter))  # Use scientific notation for y-axis

# Second subplot
axs[1].plot(FGlambdas, FGCR_AE_S1, label='AE_(S1)', color='g', marker='o')
axs[1].plot(FGlambdas, FGCR_cheng_S1, label='Cheng_AT_(S1)', color='m', marker='s')
axs[1].plot(FGlambdas, FGCR_AE_S2, label='AE_(S2)', color='y', marker='x')
axs[1].plot(FGlambdas, FGCR_cheng_S2, label='Cheng_AT_(S2)', color='orange', marker='p')
axs[1].plot(FGlambdas, FGCR_AE_total, label='AE_(S1+S2)', color='k', marker='*')
axs[1].plot(FGlambdas, FGCR_cheng_total, label='Cheng_AT(S1+S2)', color='r', marker='D')
axs[1].set_xlabel('Lambda ($\lambda$)', fontsize=15, color='black')  # Font size and color
axs[1].set_ylabel('CR', fontsize=15, color='black')  # Font size and color)
axs[1].set_xscale('log')  # Log scale for x-axis
axs[1].set_title('(ii) FG', fontsize=16, fontweight='bold', color='black')  # Bold title, size, and color)
axs[1].legend()
axs[1].grid(True)
axs[1].tick_params(axis='both', labelsize=12, colors='black')  # Font size for tick labels
axs[1].yaxis.set_major_formatter(FuncFormatter(scientific_formatter))  # Use scientific notation for y-axis

# Adjust layout and display the plot
plt.tight_layout()
plt.show()