import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.stats import norm
np.random.seed(7)
def Plot(Y, mean, var, n, title):
    sigma = np.sqrt(var / n)
    x = np.linspace(min(Y), max(Y), 1000)
    pdf = norm.pdf(x, mean, sigma)

    plt.figure(figsize=(8, 5))
    plt.hist(Y, bins=int(np.sqrt(n)), density=True, alpha=0.6, color='skyblue', label='Histogram of Sample Means',edgecolor='black') #density simulates pdf ie area under graph is 1
    plt.plot(x, pdf, 'r-', linewidth=2, label=f"PDF of Normal($\mu$={mean:.3f}, $\sigma^2$={var/n:.3f})")
    plt.title(title)
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def Bernoulli(N,n,p):
    random_matrix=np.random.rand(N,n)
    bernoulli=(random_matrix<=p).astype(int)
    Y=np.mean(bernoulli,axis=1)
    mean = p
    var = p * (1 - p)
    Plot(Y, mean, var, n, f'Bernoulli (p={p})')
    return Y
def Uniform(N,n):
    uniform=np.random.rand(N,n)          
    Y=np.mean(uniform,axis=1)
    mean = 0.5
    var = 1/12   
    Plot(Y, mean, var, n, f'Uniform (0,1)') 
    return Y
def Exponential(N,n,l):
    expo=np.random.exponential(size=(N,n),scale=1/l)          
    Y=np.mean(expo,axis=1)
    mean = 1/l
    var = 1/(l**2)   
    Plot(Y, mean, var, n, f'Exponential ($\lambda$={l})') 
    return Y
parser = argparse.ArgumentParser(description="Generate random samples based on mode and parameters.")
parser.add_argument("mode", type=int, choices=[0, 1, 2], help="0: Bernoulli, 1: Uniform, 2: Exponential")
parser.add_argument("n", type=int, help="Number of columns")
parser.add_argument("N", type=int, help="Number of rows")
parser.add_argument("para", type=float, nargs='?', default=None,help="Parameter for Bernoulli (p) or Exponential (λ). Not needed for Uniform (mode=1)")

args = parser.parse_args()
if args.mode == 0:
    Bernoulli(args.N,args.n,args.para) #Refers to part 1


elif args.mode == 1:
    Uniform(args.N,args.n)

elif args.mode == 2: 
    Exponential(args.N,args.n,args.para) #Refers to part 3

