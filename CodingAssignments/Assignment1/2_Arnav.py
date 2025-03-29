import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

def bernoulli(p,infile):
    a=str(p)
    if a[0]=='0':
       a=a[1:]
    outfile=f"Bernoulli_{a.replace('.','p')}.csv"
    uniform_samples = np.genfromtxt(infile, delimiter=",", dtype=float, invalid_raise=False)
    uniform_samples = uniform_samples[~np.isnan(uniform_samples)]#Removes all strings from uniform samples if any
    bernoulli_samples=(uniform_samples<=p).astype(int)
    mean=np.mean(bernoulli_samples)
    np.savetxt(outfile,bernoulli_samples,delimiter=",",fmt="%d")
    print("Sample Mean=",mean)

def exponential(l,infile):
    a=str(l)
    if a[0]=='0':
       a=a[1:]
    outfile=f"Exponential_{a.replace('.','p')}.csv"
    uniform_samples = np.genfromtxt(infile, delimiter=",", dtype=float, invalid_raise=False)
    uniform_samples = uniform_samples[~np.isnan(uniform_samples)]
    exponential_samples=-np.log(1-uniform_samples+1e-10)/l
    np.savetxt(outfile,exponential_samples,delimiter=",")
    N=len(uniform_samples)
    plt.hist(exponential_samples,bins=int(np.sqrt(N)))
    plt.xlabel("Exponential Samples")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Exponential({l}) Samples")
    plt.show()

def cdfx(infile):
    outfile="CDFX.csv"
    uniform_samples = np.genfromtxt(infile, delimiter=",", dtype=float, invalid_raise=False)
    uniform_samples = uniform_samples[~np.isnan(uniform_samples)]
    N=len(uniform_samples)
    cdfx_samples=np.zeros(N)
    for u in range(N):
        U=uniform_samples[u]
        if U>=0 and U<=1/3:
            cdfx_samples[u]=((3*U)**0.5)
        elif U>1/3 and U<=2/3:
            cdfx_samples[u]=2
        elif U>2/3 and U<=1:
            cdfx_samples[u]=(6*U-2)

    np.savetxt(outfile,cdfx_samples,delimiter=",")
    print("Number of 2:", np.count_nonzero(cdfx_samples == 2))
    plt.hist(cdfx_samples,bins=int(np.sqrt(N)))
    plt.xlabel("CDFX Samples")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of CDFX Samples")
    plt.show()

if len(sys.argv) < 3:
    print("Usage: python 2_Arnav.py <mode> <filename> [parameter]")
    sys.exit(1)

mode = int(sys.argv[1])  # Mode: 0 (Bernoulli), 1 (Exponential), 2 (CDFX)
filename = sys.argv[2]  # File containing uniform samples

if mode == 0:
    if len(sys.argv) < 4:
        print("Error: Bernoulli mode requires parameter p")
        sys.exit(1)
    p = float(sys.argv[3])
    bernoulli(p,filename) #Refers to part 1


elif mode == 1:
    if len(sys.argv) < 4:
        print("Error: Exponential mode requires parameter λ")
        sys.exit(1)
    l = float(sys.argv[3])
    exponential(l,filename) #Refers to part 2

elif mode == 2: 
    cdfx(filename) #Refers to part 3

else:
    print("Error: Invalid mode. Use 0 for Bernoulli, 1 for Exponential, 2 for CDFX")
