import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend('Agg')

# Define your results:
labels = ["50Hz, kw=32", "50Hz, kw=14", "125Hz, kw=32", "250Hz, kw=32", "250Hz, kw=64"]
val_CER = [83.27426147460938, 20.602569580078125, 20.425342559814453, 32.18874740600586, 23.836952209472656]
test_CER = [83.81240844726562, 19.53749656677246, 21.56905174255371, 35.35768127441406, 24.832504272460938]

# Create the x positions for each group:
x = np.arange(len(labels))
width = 0.35  # width of each bar

# Create the plot:
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, val_CER, width, label='Validation CER')
rects2 = ax.bar(x + width/2, test_CER, width, label='Test CER')

# Add labels and title:
ax.set_ylabel('CER')
ax.set_title('CER for Different Sampling Rates and Kernel Widths')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Optional: Add text labels above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("sampling_rates_and_kernel_widths.png", bbox_inches='tight', dpi=300)
