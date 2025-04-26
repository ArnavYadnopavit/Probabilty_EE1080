import numpy as np
import matplotlib.pyplot as plt
import argparse
def Plot(samples,N):
    bins = int(np.sqrt(N))
    hist_vals, bin_edges = np.histogram(samples, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Evaluate fa and fb at bin centers
    fa = 60 * (bin_centers**3) * ((1 - bin_centers)**2)
    fb = 30 * (bin_centers**4) * (1 - bin_centers)

    # Compute Mean Squared Error
    mse_fa = np.mean((hist_vals - fa)**2)
    mse_fb = np.mean((hist_vals - fb)**2)
    if mse_fa < mse_fb:
        print("a")
    else:
        print("b")
    x=np.linspace(0,1,N)
    fa=60*(x**3)*((1-x)**2)
    fb=30*(x**4)*(1-x)
    plt.figure(figsize=(8, 5))
    plt.hist(samples, bins=bins, density=True, alpha=0.6, color='skyblue', label='Histogram of Sample Means') #density simulates pdf ie area under graph is 1
    plt.plot(x,fa,label="fa")
    plt.plot(x,fb,label="fb")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def samplegen(N):
    samples=np.random.rand(N,6)
    sorted_samples = np.sort(samples, axis=1)
    fourth=sorted_samples[:,3]
    Plot(fourth,N)
    return

parser = argparse.ArgumentParser()
parser.add_argument("N",type=int,help="n")
args = parser.parse_args()
np.random.seed(7)

samplegen(args.N)
