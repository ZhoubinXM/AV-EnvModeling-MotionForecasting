from pickle import TRUE
import numpy as np
import os
from matplotlib import pyplot as plt
from sympy import true
import seaborn as sns

output_dir = "./output"
predictions = np.load(os.path.join(output_dir, 'saved_model', 'tri_prediction.npy'))
gt = np.load(os.path.join(output_dir, 'saved_model', 'tri_gt.npy'))
diff = abs(predictions - gt)
all_values = len(diff)

mse = ((predictions - gt)**2).mean()
abs_error = diff.mean()
rel_error = (diff / gt).mean()
print("MSE:", mse, "abs_error:", abs_error, "rel_error:", rel_error)

# plot
bins = np.arange(start=0, stop=15, step=0.5)
hist, bin_edges = np.histogram(diff, range=(0, 15), bins=30)
hist = hist / all_values

fig, ax = plt.subplots()
ax.bar(bins + 0.25, hist, width=0.5)
ax.set_title("abs error distribution")
plt.show()

csum = np.cumsum(hist)
log = ""
for i in np.arange(0.5, 15.5, 0.5):
    log += ("|" + str(i))
log += "|"
print(log)
log = ""
for i in np.arange(0.5, 15.5, 0.5):
    log += "|:----:"
log += "|"
print(log)
log = ""
for i in csum:
    log +=  ("|" + str(round(i,2)))
log += "|"
print(log)


log = ""
for i in np.arange(0, 15, 0.5):
    log += ("|" + "[" + str(i) + ", " + str(i + 0.5) + "]")
log += "|"
print(log)
log = ""
for i in np.arange(0, 15, 0.5):
    log += "|:----:"
log += "|"
print(log)
log = ""
for i in np.arange(0, 15, 0.5):
    index = np.logical_and(gt < i + 0.5, gt >= i)
    metric = sum(diff[index]) / len(diff[index])
    log +=  ("|" + str(round(metric, 2)) + " ")
log += "|"
print(log)
log = ""
for i in np.arange(0, 15, 0.5):
    index = np.logical_and(gt < i + 0.5, gt >= i)
    metric = sum(diff[index] / gt[index]) / len(diff[index])
    log +=  ("|" + str(round(metric, 2)) + " ")
log += "|"
print(log)

