import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print("Usage: python 3_Arnav.py <mean0> <var0> <csv_filename>")
    sys.exit(1)

mean0 = float(sys.argv[1])
var0 = float(sys.argv[2])
csv_filename = sys.argv[3]
df = pd.read_csv(csv_filename, header=None)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

samples = df[0].values
varss = df[1].values

N = len(samples)
mmse= []

for n in range(1, N + 1):
    Y = samples[:n]
    vari = varss[:n]
    est= ((mean0 / var0) + np.sum(Y / vari))/((1 / var0) + np.sum(1 / vari))
    mmse.append(est)
plt.figure(figsize=(8, 6))
plt.scatter(range(1, N + 1), mmse,color='blue',alpha=0.5)
plt.xlabel('Number of Samples (N)')
plt.ylabel('MMSE Estimate')
plt.title('MMSE Estimates vs Number of Samples')
plt.grid(True)
plt.tight_layout()
plt.show()

