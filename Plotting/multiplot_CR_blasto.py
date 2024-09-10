import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Define a function to format the y-axis labels in scientific notation
def scientific_formatter(x, pos):
    return f'{x:.0e}'

# Data for the first plot
FCR_AE_S1 = np.array([189629.61, 182857.14, 155151.51, 155151.51,  65641.02,  29425.26,12736.32,  10513.35,   9733.83,   4507.02,   1814.31,   1356.99,1236.42,    888.72,   1073.82])
FCR_cheng_S1 = np.array([53894.7,53894.7,53894.7,53894.7,51200.1,46545.6,35068.5,32405.1,30295.8,17239.05,6799.47,6522.3,6513.99,5932.8,5838.09])
FCR_AE_S2 = np.array([204800.1,204800.1,204800.1,204800.1,72112.8,37926,24975.6,21694.92,20983.62,11203.5,3852.51,2411.682,1996.881,895.104,942.042])
FCR_cheng_S2 = np.array([53894.7,53894.7,53894.7,53894.7,52783.5,47850.6,42666.6,40314.9,38787.9,24975.6,12190.47,10098.63,8966.73,8605.05,7962.66])
FCR_AE_total = np.array([98461.5,96603.9,88275.9,88275.9,34362.3,16569.57,8434.92,6719.16,7004.1,3214.05,1233.438,868.386,763.608,445.953,501.813])
FCR_cheng_total = np.array([26947.38,26947.38,26947.38,26947.38,25989.84,23594.46,19248.12,17964.9,17009.97,10199.19,4364.88,3959.79,3775.8,3531.03,3450.12])
Flambdas = np.array([0.000000002,0.0000002,0.0002,0.002,0.05,0.2,0.6,0.8,1,4,32,64,100,10000000,10000000000])
Fxaxis = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

# Data for the second plot
FGCR_AE_S1 = np.array([189629.7,189629.7,155151.6,85333.2,48762.0,43760.7,33907.2,12929.28,5638.77,3905.43,2221.257,1987.578,900.774])
FGCR_cheng_S1 = np.array([53894.7,53894.7,53894.7,53894.7,46545.6,45714.3,43025.1,33032.4,12673.26,10406.49,8151.6,7840.74,8619.54])
FGCR_AE_S2 = np.array([189629.7,189629.7,106666.8,57528.0,38496.3,32201.4,29767.44,21333.33,8919.87,4816.56,3406.53,2752.689,1142.856])
FGCR_cheng_S2 = np.array([54468.0,53894.7,53894.7,53894.7,48301.8,46545.6,44912.4,34362.3,20645.16,17009.97,13298.7,14027.4,12397.08])
FGCR_AE_total = np.array([94814.7,94814.7,63210.0,34362.3,21512.61,18618.18,15900.63,8050.32,3454.8,2156.697,1344.537,1154.193,503.739])
FGCR_cheng_total = np.array([27089.94,26947.38,26947.38,26947.38,23703.69,23063.07,21974.25,16842.12,7852.77,6456.48,5054.28,5029.47,5084.4])
FGlambdas = np.array([0.0002,0.002,0.05,0.2,0.8,0.6,1,4,16,32,64,100,10000000000])
FGxaxis = np.array([1,2,3,4,5,6,7,8,9,10,11,12])

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

