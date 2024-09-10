import numpy as np
import matplotlib.pyplot as plt

# Data for the first plot
miou_nc_1 = np.array([0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892])
miou_G_1 = np.array([0.606, 0.875, 0.905, 0.908, 0.886, 0.911, 0.908, 0.911, 0.905, 0.909, 0.898, 0.898, 0.927, 0.893, 0.896])
miou_F_1 = np.array([0.195, 0.237, 0.81, 0.818, 0.823, 0.845, 0.873, 0.882, 0.882, 0.887, 0.889, 0.9, 0.904, 0.907, 0.908])
miou_FG_1 = np.array([0.51, 0.672, 0.804, 0.816, 0.815, 0.825, 0.837, 0.853, 0.871, 0.872, 0.872, 0.879, 0.895, 0.889, 0.903])
xaxis_1 = np.array([2e-9, 2e-7, 2e-4, 0.002, 0.05, 0.2, 0.6, 0.8, 1, 4, 32, 64, 100, 1e7, 1e10])

# Data for the second plot
miou_nc_2 = np.array([0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892])
miou_G_2 = np.array([0.626, 0.877, 0.909, 0.905, 0.905, 0.906, 0.908, 0.908, 0.909, 0.912, 0.918, 0.924, 0.909, 0.91, 0.908])
miou_F_2 = np.array([0.44, 0.782, 0.836, 0.841, 0.846, 0.868, 0.873, 0.873, 0.898, 0.899, 0.903, 0.915, 0.921, 0.928, 0.93])
miou_FG_2 = np.array([0.517, 0.799, 0.827, 0.829, 0.838, 0.843, 0.844, 0.845, 0.845, 0.873, 0.88, 0.888, 0.893, 0.914, 0.922])
xaxis_2 = np.array([2e-9, 0.0002, 2e-4, 0.002, 0.05, 0.2, 0.6, 0.8, 1, 4, 32, 64, 100, 1e7, 1e10])

# Function to calculate AUC
def calculate_auc(x, y):
    return np.trapz(y, x)

# Calculate AUCs for the first plot
auc_nc_1 = calculate_auc(xaxis_1, miou_nc_1)
auc_G_1 = calculate_auc(xaxis_1, miou_G_1)
auc_F_1 = calculate_auc(xaxis_1, miou_F_1)
auc_FG_1 = calculate_auc(xaxis_1, miou_FG_1)

# Calculate AUCs for the second plot
auc_nc_2 = calculate_auc(xaxis_2, miou_nc_2)
auc_G_2 = calculate_auc(xaxis_2, miou_G_2)
auc_F_2 = calculate_auc(xaxis_2, miou_F_2)
auc_FG_2 = calculate_auc(xaxis_2, miou_FG_2)

# Print AUCs
print(f"AUC for NC (AE): {auc_nc_1:.4f}")
print(f"AUC for G (AE): {auc_G_1:.4f}")
print(f"AUC for F (AE): {auc_F_1:.4f}")
print(f"AUC for FG (AE): {auc_FG_1:.4f}")

print(f"AUC for NC (Cheng_AT): {auc_nc_2:.4f}")
print(f"AUC for G (Cheng_AT): {auc_G_2:.4f}")
print(f"AUC for F (Cheng_AT): {auc_F_2:.4f}")
print(f"AUC for FG (Cheng_AT): {auc_FG_2:.4f}")

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# First subplot
# First subplot
axs[0].plot(xaxis_1, miou_nc_1, label=f'NC (AUC: {auc_nc_1:.4f})', color='g', marker='o')
axs[0].plot(xaxis_1, miou_G_1, label=f'G (AUC: {auc_G_1:.4f})', color='orange', marker='x')
axs[0].plot(xaxis_1, miou_F_1, label=f'F (AUC: {auc_F_1:.4f})', color='b', marker='s')
axs[0].plot(xaxis_1, miou_FG_1, label=f'FG (AUC: {auc_FG_1:.4f})', color='red', marker='*')
axs[0].set_xlabel('Lambda ($\\lambda$)', fontsize=15, color='black')  # Font size and color
axs[0].set_ylabel('MJI', fontsize=15, color='black')  # Font size and color
axs[0].set_title('(i) AE', fontsize=16, fontweight='bold', color='black')  # Bold title, size, and color
axs[0].legend(fontsize=12)  # Legend font size
axs[0].grid(True)
axs[0].set_xticks(xaxis_1)  # To ensure all x-axis labels are displayed
axs[0].set_xscale('log')  # Use logarithmic scale for better visualization
axs[0].tick_params(axis='both', labelsize=12, colors='black')  # Font size for tick labels

# Second subplot
axs[1].plot(xaxis_2, miou_nc_2, label=f'NC (AUC: {auc_nc_2:.4f})', color='g', marker='o')
axs[1].plot(xaxis_2, miou_G_2, label=f'G (AUC: {auc_G_2:.4f})', color='orange', marker='x')
axs[1].plot(xaxis_2, miou_F_2, label=f'F (AUC: {auc_F_2:.4f})', color='b', marker='s')
axs[1].plot(xaxis_2, miou_FG_2, label=f'FG (AUC: {auc_FG_2:.4f})', color='red', marker='*')
axs[1].set_xlabel('Lambda ($\\lambda$)', fontsize=15, color='black')  # Font size and color
axs[1].set_ylabel('MJI', fontsize=15, color='black')  # Font size and color
axs[1].set_title('(ii) Cheng_AT', fontsize=16, fontweight='bold', color='black')  # Bold title, size, and color
axs[1].legend(fontsize=12)  # Legend font size
axs[1].grid(True)
axs[1].set_xticks(xaxis_2)  # To ensure all x-axis labels are displayed
axs[1].set_xscale('log')  # Use logarithmic scale for better visualization
axs[1].tick_params(axis='both', labelsize=12, colors='black')  # Font size for tick labels

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
