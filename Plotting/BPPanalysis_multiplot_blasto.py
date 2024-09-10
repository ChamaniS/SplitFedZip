import matplotlib.pyplot as plt
import pandas as pd

# Data for the first plot
data1 = {
    "Local epoch": list(range(1, 13)),
    "Average S1(f)": [0.012132, 0.01213, 0.0121, 0.01209,0.012084, 0.012066, 0.012062, 0.01206, 0.012058, 0.012056, 0.012052, 0.012012],
    "Average S2(f)": [0.011642, 0.01162,0.011602, 0.011596, 0.011586, 0.011566, 0.011558, 0.011558, 0.011504, 0.01144, 0.011434, 0.0114],
    "Average S1(g)": [0.009502, 0.009496, 0.00949, 0.00948, 0.00948, 0.009476, 0.009468, 0.0094632, 0.009462, 0.009462, 0.009458, 0.009475],
    "Average S2(g)": [0.009496, 0.00949, 0.009488, 0.009487, 0.009482, 0.009476, 0.009474, 0.0094738, 0.0094672, 0.00947, 0.00947, 0.00946]

}
df1 = pd.DataFrame(data1)

# Data for the second plot
data2 = {
    "BPP-S1": [0.0134, 0.0126, 0.0124, 0.0124, 0.0122, 0.0121, 0.0121, 0.0120, 0.0120, 0.0120],
    "BPP-S2": [0.0137, 0.0119, 0.0114, 0.0113, 0.0111, 0.0111, 0.0112, 0.0111, 0.0111, 0.0110],
    "BPP-T": [0.0270, 0.0245, 0.0237, 0.0237, 0.0233, 0.0233, 0.0233, 0.0231, 0.0231, 0.0230]
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
