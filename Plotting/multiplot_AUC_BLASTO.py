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
xaxis_2 = np.array([2e-9, 0.0002,0.002, 0.05, 0.2, 0.6, 0.8, 1, 4, 32, 64, 100, 1e10])

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# First subplot
axs[0].plot(xaxis_1, miou_nc_1, label='NC', color='g', marker='o')
axs[0].plot(xaxis_1, miou_G_1, label='G', color='orange',marker='x')
axs[0].plot(xaxis_1, miou_F_1, label='F', color='b',marker='s')
axs[0].plot(xaxis_1, miou_FG_1, label='FG',color='red',marker='*')
axs[0].set_xlabel('Lambda ($\lambda$)')
axs[0].set_ylabel('MJI')
axs[0].set_title('(i) AE')
axs[0].legend()
axs[0].grid(True)
axs[0].set_xticks(xaxis_1)  # To ensure all x-axis labels are displayed
axs[0].set_xscale('log')  # Use logarithmic scale for better visualization

# Second subplot
axs[1].plot(xaxis_2, miou_nc_2, label='NC',color='g', marker='o')
axs[1].plot(xaxis_2, miou_G_2, label='G', color='orange',marker='x')
axs[1].plot(xaxis_2, miou_F_2, label='F',color='b', marker='s')
axs[1].plot(xaxis_2, miou_FG_2, label='FG',color='red',marker='*')
axs[1].set_xlabel('Lambda ($\lambda$)')
axs[1].set_ylabel('MJI')
axs[1].set_title('(ii) Cheng_AT')
axs[1].legend()
axs[1].grid(True)
axs[1].set_xticks(xaxis_2)  # To ensure all x-axis labels are displayed
axs[1].set_xscale('log')  # Use logarithmic scale for better visualization

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
