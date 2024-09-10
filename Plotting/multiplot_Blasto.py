#Blastocyst dataset

import matplotlib.pyplot as plt
import numpy as np

# Data for each plot
# First row
# AE-split 1
x11 = [0.0033, 0.0078, 0.0174, 0.0402, 0.0487, 0.0526, 0.1136, 0.2822, 0.3773, 0.4141, 0.4768]
y11 = [0.52, 0.716, 0.798, 0.821, 0.83, 0.831, 0.839, 0.855, 0.861, 0.867, 0.879]
# Cheng_2020 attention-split 1
x21 = [0.0095, 0.01, 0.011, 0.0146, 0.0158, 0.0169, 0.0297, 0.0753, 0.0785, 0.0786, 0.0877]
y21 = [0.649, 0.762, 0.784, 0.799, 0.807, 0.808, 0.822, 0.834, 0.836, 0.839, 0.849]

# AE-split 2
x12 = [0.0025, 0.0071, 0.0135, 0.0205, 0.0236, 0.0244, 0.0457, 0.1329, 0.2123, 0.2564, 0.5435]
y12 = [0.52, 0.716, 0.798, 0.821, 0.83, 0.831, 0.839, 0.855, 0.861, 0.867, 0.879]
# Cheng_2020 attention-split 2
x22 = [0.0095, 0.0097, 0.0107, 0.012, 0.0127, 0.0132, 0.0205, 0.042, 0.0507, 0.0571, 0.0643]
y22 = [0.649, 0.762, 0.784, 0.799, 0.807, 0.808, 0.822, 0.834, 0.836, 0.839, 0.849]

# AE
x1 = [0.0058, 0.0149, 0.0309, 0.0607, 0.0762, 0.0731, 0.1593, 0.4151, 0.5896, 0.6705, 1.0203]
y1 = [0.52, 0.716, 0.798, 0.821, 0.83, 0.831, 0.839, 0.855, 0.861, 0.867, 0.879]
# Cheng_2020 attention
x2 = [0.019, 0.0197, 0.0217, 0.0266, 0.0285, 0.0301, 0.0502, 0.1173, 0.1293, 0.1356, 0.1484]
y2 = [0.649, 0.762, 0.784, 0.799, 0.807, 0.808, 0.822, 0.834, 0.839, 0.839, 0.849]

# Second row
# AE-split 1
x11_2 = [0.0027, 0.0033, 0.006, 0.0105, 0.0117, 0.0151, 0.03960, 0.0908, 0.1311, 0.2305]
y11_2 = [0.508, 0.652, 0.734, 0.788, 0.792, 0.795, 0.804, 0.824, 0.83, 0.851]
# Cheng_2020 attention-split 1
x21_2 = [0.0095, 0.0095, 0.0095, 0.011, 0.0112, 0.0119, 0.0155, 0.0404, 0.0492, 0.0628]
y21_2 = [0.686, 0.69, 0.702, 0.764, 0.779, 0.778, 0.791, 0.813, 0.819, 0.822]

# AE-split 2
x12_2 = [0.0027, 0.0048, 0.0089, 0.0133, 0.0159, 0.0172, 0.024, 0.0574, 0.1063, 0.1503]
y12_2 = [0.508, 0.652, 0.734, 0.788, 0.792, 0.795, 0.804, 0.824, 0.83, 0.851]
# Cheng_2020 attention-split 2
x22_2 = [0.0095, 0.0095, 0.0095, 0.0106, 0.011, 0.0114, 0.0149, 0.0248, 0.0301, 0.0385]
y22_2 = [0.686, 0.69, 0.702, 0.764, 0.779, 0.778, 0.791, 0.813, 0.818, 0.822]

# AE
x1_2 = [0.0054, 0.0081, 0.0149, 0.0238, 0.0275, 0.0322, 0.0636, 0.1482, 0.2374, 0.3808]
y1_2 = [0.508, 0.652, 0.734, 0.788, 0.792, 0.795, 0.804, 0.824, 0.83, 0.851]
# Cheng_2020 attention
x2_2 = [0.019, 0.019, 0.019, 0.0216, 0.0222, 0.0233, 0.0304, 0.0652, 0.0793, 0.1013]
y2_2 = [0.686, 0.69, 0.702, 0.764, 0.779, 0.778, 0.791, 0.813, 0.818, 0.822]

# Create a figure and a 2x3 grid of subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Plot in each subplot
# First row
axs[0, 0].axhline(y=0.892, color='g', linestyle='-', label='NC')
axs[0, 0].plot(x11, y11, marker='o', linestyle='-', color='b', label='AE (S1)')
axs[0, 0].plot(x21, y21, marker='o', linestyle='-', color='r', label='Cheng_AT (S1)')
axs[0, 0].set_title('(a) MJI vs BPP-S1', fontsize=14, fontweight='bold', color='black')
axs[0, 0].legend(fontsize=12)
axs[0, 0].annotate('0.892', xy=(-0.021, 0.879), xytext=(-0.15, 0.892),
                    textcoords='offset points', ha='right', va='center',
                    fontsize=12, color='green')
axs[0, 0].set_xlabel('BPP-S1', fontsize=14)
axs[0, 0].set_ylabel('MJI', fontsize=14)
axs[0, 0].grid(True)

axs[0, 1].axhline(y=0.892, color='g', linestyle='-', label='NC')
axs[0, 1].plot(x12, y12, marker='x', linestyle='-', color='b', label='AE (S2)')
axs[0, 1].plot(x22, y22, marker='x', linestyle='-', color='r', label='Cheng_AT (S2)')
axs[0, 1].set_title('(b) MJI vs BPP-S2', fontsize=14, fontweight='bold', color='black')
axs[0, 1].annotate('0.892', xy=(-0.023, 0.879), xytext=(-0.15, 0.892),
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
axs[0, 2].annotate('0.892', xy=(-0.04, 0.879), xytext=(-0.15, 0.892),
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
axs[1, 0].annotate('0.892', xy=(-0.009, 0.879), xytext=(-0.15, 0.892),
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
axs[1, 1].annotate('0.892', xy=(-0.0049, 0.879), xytext=(-0.15, 0.892),
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
axs[1, 2].annotate('0.892', xy=(-0.014, 0.879), xytext=(-0.15, 0.892),
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