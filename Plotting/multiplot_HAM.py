#Blastocyst dataset

import matplotlib.pyplot as plt
import numpy as np

# Data for each plot
# First row
# AE-split 1
x11 = [0.0002, 0.0002, 0.0003, 0.0005, 0.0005, 0.0028, 0.0052, 0.0101, 0.007, 0.0127, 0.0143, 0.0878, 0.0939, 0.376, 0.4712, 0.4688]
y11 = [0.195,0.237,  0.81, 0.818, 0.823, 0.845, 0.873, 0.882, 0.882, 0.887, 0.889, 0.904, 0.908, 0.894, 0.907, 0.9]
# Cheng_2020 attention-split 1
x21 = [0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0098, 0.0107, 0.011, 0.0114, 0.0127, 0.0259, 0.0314, 0.0455, 0.0658, 0.0748, 0.08]
y21 = [0.44, 0.782, 0.836, 0.841, 0.846, 0.868, 0.873, 0.873, 0.898, 0.899, 0.903, 0.915, 0.921,0.925, 0.928,0.932]

# AE-split 2
x12 = [0.0002, 0.0002, 0.0002, 0.0002, 0.0006, 0.0015, 0.0028, 0.0043, 0.0032, 0.006, 0.0059, 0.0215, 0.0299, 0.3038, 0.3812, 0.4078]
y12 = [0.195,0.237, 0.81, 0.818, 0.823, 0.845, 0.873, 0.882, 0.882, 0.887, 0.889, 0.904, 0.908, 0.894, 0.907, 0.9]
# Cheng_2020 attention-split 2
x22 = [0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095,0.0097, 0.0098, 0.0098, 0.0101, 0.0115, 0.0137, 0.0143, 0.0216, 0.023, 0.0223]
y22 = [0.44, 0.782, 0.836, 0.841, 0.846, 0.868, 0.873, 0.873, 0.898, 0.899, 0.903, 0.915, 0.921,0.925, 0.928,0.932]

# AE
x1 = [0.0004, 0.0004, 0.0005, 0.0007, 0.0011, 0.0043, 0.008, 0.0144, 0.0102, 0.0187, 0.0202, 0.1093, 0.1238, 0.6798, 0.8524, 0.8766]
y1 = [0.195,0.237, 0.81, 0.818, 0.823, 0.845, 0.873, 0.882, 0.882, 0.887, 0.889, 0.904, 0.908, 0.894, 0.907, 0.9]
# Cheng_2020 attention
x2 = [0.019, 0.019, 0.019, 0.019, 0.019, 0.0193, 0.0204, 0.0208, 0.0212, 0.0228, 0.0429, 0.0396, 0.0598, 0.0978, 0.0874, 0.1023]
y2 = [0.44, 0.782, 0.836, 0.841, 0.846, 0.868, 0.873, 0.873, 0.898, 0.899, 0.903, 0.915, 0.921,0.925, 0.928,0.932]

# Second row

# AE-split 1
x11_2 = [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0044, 0.0055, 0.0116, 0.0457, 0.0764, 0.1074, 0.2634, 0.3066, 0.3096]
y11_2 = [0.51, 0.672, 0.804, 0.816, 0.815, 0.825, 0.837, 0.853, 0.871, 0.872, 0.872, 0.879, 0.895, 0.897, 0.889, 0.903]
# Cheng_2020 attention-split 1
x21_2 = [0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095,0.013, 0.0232, 0.0328, 0.0319, 0.0663, 0.0726, 0.0755]
y21_2 = [0.517, 0.799, 0.827, 0.829, 0.838, 0.843, 0.844, 0.845, 0.845, 0.873, 0.88, 0.888, 0.893, 0.906, 0.914, 0.922]

# AE-split 2
x12_2 = [0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0002, 0.0029, 0.0036, 0.0065, 0.0134, 0.0264,0.0312, 0.1402, 0.1649, 0.1808]
y12_2 = [0.51, 0.672, 0.804, 0.816, 0.815, 0.825, 0.837, 0.853, 0.871, 0.872, 0.872, 0.879, 0.895, 0.897, 0.889, 0.903]
# Cheng_2020 attention-split 2
x22_2 = [0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0095, 0.0105, 0.0131, 0.0184, 0.0193, 0.0249, 0.0254, 0.0262]
y22_2 = [0.517, 0.799, 0.827, 0.829, 0.838, 0.843, 0.844, 0.845, 0.845, 0.873, 0.88, 0.888, 0.893, 0.906, 0.914, 0.922]

# AE
x1_2 = [0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0026, 0.0073, 0.0091, 0.0181, 0.0591, 0.1076, 0.1338, 0.4036, 0.4715, 0.4904]
y1_2 = [0.51, 0.672, 0.804, 0.816, 0.815, 0.825, 0.837, 0.853, 0.871, 0.872, 0.872, 0.879, 0.895, 0.897, 0.889, 0.903]
# Cheng_2020 attention
x2_2 = [0.019, 0.019, 0.019, 0.019, 0.019, 0.019, 0.019, 0.019, 0.019, 0.0235, 0.0363, 0.0512, 0.0512, 0.0912, 0.098, 0.1017]
y2_2 = [0.517, 0.799, 0.827, 0.829, 0.838, 0.843, 0.844, 0.845, 0.845, 0.873, 0.88, 0.888, 0.893, 0.906, 0.914, 0.922]

# Create a figure and a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot in each subplot
# First row
axs[0, 0].axhline(y=0.892, color='g', linestyle='-', label='NC')
axs[0, 0].plot(x11, y11, marker='o', linestyle='-', color='b', label='AE (S1)')
axs[0, 0].plot(x21, y21, marker='o', linestyle='-', color='r', label='Cheng_AT (S1)')
axs[0, 0].set_title('(a) MJI vs BPP-S1', fontsize=14, fontweight='bold', color='black')
axs[0, 0].legend(fontsize=12)
axs[0, 0].annotate('0.892', xy=(-0.02, 0.86), xytext=(-0.13, 0.892),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=12, color='green')
axs[0, 0].set_xlabel('BPP-S1', fontsize=14)
axs[0, 0].set_ylabel('MJI', fontsize=14)
axs[0, 0].grid(True)

axs[0, 1].axhline(y=0.892, color='g', linestyle='-', label='NC')
axs[0, 1].plot(x12, y12, marker='x', linestyle='-', color='b', label='AE (S2)')
axs[0, 1].plot(x22, y22, marker='x', linestyle='-', color='r', label='Cheng_AT (S2)')
axs[0, 1].set_title('(b) MJI vs BPP-S2', fontsize=14, fontweight='bold', color='black')
axs[0, 1].annotate('0.892', xy=(-0.02, 0.86), xytext=(-0.15, 0.892),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=12, color='green')
axs[0, 1].set_xlabel('BPP-S2', fontsize=14)
axs[0, 1].set_ylabel('MJI', fontsize=14)
axs[0, 1].legend(fontsize=12)
axs[0, 1].grid(True)

axs[0, 2].axhline(y=0.892, color='g', linestyle='-', label='NC')
axs[0, 2].plot(x1, y1, marker='s', linestyle='-', color='b', label='AE (S1+S2)')
axs[0, 2].plot(x2, y2, marker='s', linestyle='-', color='r', label='Cheng_AT (S1+S2)')
axs[0, 2].set_title('(c) MJI vs BPP-T', fontsize=14, fontweight='bold', color='black')
axs[0,2].annotate('0.892', xy=(-0.04, 0.86), xytext=(-0.15, 0.892),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=12, color='green')
axs[0, 2].set_xlabel('BPP-T', fontsize=14)
axs[0, 2].set_ylabel('MJI', fontsize=14)
axs[0, 2].legend(fontsize=12)
axs[0, 2].grid(True)

# Second row
axs[1, 0].axhline(y=0.892, color='g', linestyle='-', label='NC')
axs[1, 0].plot(x11_2, y11_2, marker='o', linestyle='-', color='b', label='AE (S1)')
axs[1, 0].plot(x21_2, y21_2, marker='o', linestyle='-', color='r', label='Cheng_AT (S1)')
axs[1, 0].set_title('(a) MJI vs BPP-S1', fontsize=14, fontweight='bold', color='black')
axs[1, 0].annotate('0.892', xy=(-0.015, 0.875), xytext=(-0.15, 0.892),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=12, color='green')
axs[1, 0].set_xlabel('BPP-S1', fontsize=14)
axs[1, 0].set_ylabel('MJI', fontsize=14)
axs[1, 0].legend(fontsize=12)
axs[1, 0].grid(True)

axs[1, 1].axhline(y=0.892, color='g', linestyle='-', label='NC')
axs[1, 1].plot(x12_2, y12_2, marker='x', linestyle='-', color='b', label='AE (S2)')
axs[1, 1].plot(x22_2, y22_2, marker='x', linestyle='-', color='r', label='Cheng_AT (S2)')
axs[1, 1].set_title('(b) MJI vs BPP-S2', fontsize=14, fontweight='bold', color='black')
axs[1, 1].annotate('0.892', xy=(-0.009, 0.875), xytext=(-0.15, 0.892),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=12, color='green')
axs[1, 1].set_xlabel('BPP-S2', fontsize=14)
axs[1, 1].set_ylabel('MJI', fontsize=14)
axs[1, 1].legend(fontsize=12)
axs[1, 1].grid(True)

axs[1, 2].axhline(y=0.892, color='g', linestyle='-', label='NC')
axs[1, 2].plot(x1_2, y1_2, marker='s', linestyle='-', color='b', label='AE (S1+S2)')
axs[1, 2].plot(x2_2, y2_2, marker='s', linestyle='-', color='r', label='Cheng_AT (S1+S2)')
axs[1, 2].set_title('(c) MJI vs BPP-T', fontsize=14, fontweight='bold', color='black')
axs[1, 2].annotate('0.892', xy=(-0.023, 0.875), xytext=(-0.15, 0.892),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=12, color='green')
axs[1, 2].set_xlabel('BPP-T', fontsize=14)
axs[1, 2].set_ylabel('MJI', fontsize=14)
axs[1, 2].legend(fontsize=12)
axs[1, 2].grid(True)

# Add common titles
fig.text(0.5, 0.93, '(i) F', ha='center', fontsize=16, fontweight='bold', color='black')
fig.text(0.5, 0.47, '(ii) FG', ha='center', fontsize=16, fontweight='bold', color='black')

# Adjust layout to prevent overlap
plt.subplots_adjust(wspace=0.3, hspace=0.6)

# Show the plot
plt.show()
