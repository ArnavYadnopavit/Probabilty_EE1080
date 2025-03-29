import numpy as np
import argparse
import matplotlib.pyplot as plt

def generatingImatrix(U,I,n,k,N):
    Isum=np.zeros(N) #Contains the sigma term of each column
    for j in range(N):
        if U[0][j]<=k/n:
            I[0][j]=1
        else:
            I[0][j]=0
        Isum[j] = I[0][j]
    for i in range(1,n):
        for j in range(N):
            if U[i][j]<=((k-Isum[j])/(n-i)):
                I[i][j]=1
            else:
                I[i][j]=0
            Isum[j]+=I[i][j]
            
def Binary_I_to_decimal_I_matrix(I,n,N):
    I_decimal=np.zeros(N,dtype=int)
    for i in range(N):
        for j in range(n):
            I_decimal[i]+=int(I[j][i]*2**(n-j-1))
    return I_decimal

def plot_histogram(I_decimal,n,k,N):
    max_subset_index = 2**n
    counts=np.zeros(max_subset_index,dtype=int)    
    for val in I_decimal:
        counts[val]+=1
    plt.figure(figsize=(8,5))
    plt.bar(range(max_subset_index),counts,label="Observed",alpha=0.7,color="blue")
    plt.xlabel("Subset Index (Decimal Value)")
    plt.ylabel("Frequency")
    plt.xticks(range(max_subset_index))
    plt.title(f"Histogram of Subset Values (n={n}, k={k}, N={N})")
    plt.legend()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("n",type=int,help="n")
parser.add_argument("k",type=int,help="k")
parser.add_argument("N",type=int,help="N")
args = parser.parse_args()
np.random.seed(41)

U=np.random.rand(args.n,args.N) #The initial matrix with uniform distribution
I=np.zeros((args.n,args.N))#Inititalising I
generatingImatrix(U,I,args.n,args.k,args.N)
I_dec=Binary_I_to_decimal_I_matrix(I,args.n,args.N)
plot_histogram(I_dec,args.n,args.k,args.N)
