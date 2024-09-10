import matplotlib.pyplot as plt
import pandas as pd

# Data for the first plot
data1 = {
    "Local epoch": list(range(1, 13)),
    "Average S1(f)": [0.00345361, 0.00344453, 0.00333566, 0.00331761, 0.00329415, 0.003247118, 0.003235432, 0.003233818, 0.003224232, 0.003208091, 0.003201207, 0.003200678],
    "Average S2(f)": [0.003292201, 0.003290065, 0.003280028, 0.003273316, 0.003252391, 0.00322972,0.00315118, 0.0031233, 0.00308826, 0.00308824, 0.00306687, 0.00304047 ],
    "Average S1(g)": [0.000365137, 0.0003569, 0.000338957, 0.000329854, 0.000327427, 0.000326069, 0.000323274, 0.000322092, 0.000320124, 0.000317644, 0.00030752, 0.000304115],
    "Average S2(g)": [0.000344721, 0.00034357, 0.000326441, 0.000322236, 0.000314912, 0.000314425, 0.000310589, 0.000309984, 0.000306421, 0.000312758, 0.000312758, 0.000301517]
}
df1 = pd.DataFrame(data1)

# Data for the second plot
data2 = {
    "BPP-S1": [0.0125, 0.0102, 0.01, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009, 0.009],
    "BPP-S2": [0.0111, 0.0048, 0.0044, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043, 0.0043],
    "BPP-T": [0.0236, 0.015, 0.0144, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133, 0.0133]
}
df2 = pd.DataFrame(data2)

# Creating the multiplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# First plot
ax1.plot(df1["Local epoch"], df1["Average S1(f)"], label="BPPs during f compression at S1", marker='o')
ax1.plot(df1["Local epoch"], df1["Average S2(f)"], label="BPPs during f compression at S2", marker='x')
ax1.plot(df1["Local epoch"], df1["Average S1(g)"], label="BPPs during g compression at S1", marker='s')
ax1.plot(df1["Local epoch"], df1["Average S2(g)"], label="BPPs during g compression at S2", marker='*')

ax1.set_xlabel("Local epochs", fontsize=15, color='black')  # Font size and color
ax1.set_ylabel("Client's average training BPP", fontsize=15, color='black')  # Font size and color
ax1.set_title("(i) Client's average training BPP with local epochs\n(10th global round)",
              fontsize=16, fontweight='bold', color='black')
ax1.legend(fontsize=12)  # Legend font size
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=12, colors='black')  # Tick label size and color

# Second plot
ax2.plot(df2["BPP-S1"], label="BPP-S1", marker='o')
ax2.plot(df2["BPP-S2"], label="BPP-S2", marker='x')
ax2.plot(df2["BPP-T"], label="BPP-T", marker='s')

ax2.set_xlabel("Global epochs", fontsize=15, color='black')  # Font size and color
ax2.set_ylabel("Test BPP", fontsize=15, color='black')  # Font size and color
ax2.set_title("(ii) Test BPP over global epochs", fontsize=16, fontweight='bold', color='black')  # Title font size and color
ax2.legend(fontsize=12)  # Legend font size
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=12, colors='black')  # Tick label size and color

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
