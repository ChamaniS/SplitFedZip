import numpy as np
import matplotlib.pyplot as plt

# Data for the first plot
miou_nc_1 = np.array([0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892])
miou_G_1 = np.array([0.13, 0.886, 0.896, 0.885, 0.89, 0.889, 0.896, 0.888, 0.898, 0.895, 0.9, 0.894])
miou_F_1 = np.array([0.083, 0.52, 0.716, 0.798, 0.821, 0.83, 0.831, 0.839, 0.855, 0.861, 0.867, 0.879])
miou_FG_1 = np.array([0.0027, 0.508, 0.652, 0.734, 0.788, 0.807, 0.795, 0.804, 0.83, 0.851, 0.839, 0.843])
xaxis_1 = np.array([2e-9, 0.002, 0.05, 0.2, 0.6, 0.8, 1, 4, 32, 64, 100, 1e10])

# Data for the second plot
miou_nc_2 = np.array([0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892, 0.892])
miou_G_2 = np.array([0.114, 0.895, 0.908, 0.884, 0.891, 0.893, 0.886, 0.897, 0.903, 0.885, 0.897, 0.905, 0.888])
miou_F_2 = np.array([0.074, 0.629, 0.649, 0.762, 0.784, 0.799, 0.807, 0.808, 0.822, 0.844, 0.836, 0.839, 0.849])
miou_FG_2 = np.array([0.0027, 0.686, 0.696, 0.696, 0.702, 0.764, 0.779, 0.771, 0.791, 0.818, 0.822, 0.819, 0.796])
xaxis_2 = np.array([2e-9, 0.0002, 0.002, 0.05, 0.2, 0.6, 0.8, 1, 4, 32, 64, 100, 1e10])

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
