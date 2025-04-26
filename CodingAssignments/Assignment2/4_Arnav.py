import numpy as np
import matplotlib.pyplot as plt
import argparse
from numpy.linalg import eig
from scipy.stats import multivariate_normal

parser = argparse.ArgumentParser(description="Generate jointly Gaussian vectors")
parser.add_argument("mean1", type=float, help="Mean of first variable")
parser.add_argument("mean2", type=float, help="Mean of second variable")
parser.add_argument("var1", type=float, help="Variance of first variable")
parser.add_argument("var2", type=float, help="Variance of second variable")
parser.add_argument("cov12", type=float, help="Covariance between variables")
parser.add_argument("cov21", type=float, help="Covariance (should be same as cov12)")
parser.add_argument("N", type=int, help="Number of samples to generate")

args = parser.parse_args()
np.random.seed(7)

mean = np.array([args.mean1, args.mean2])
cov_matrix = np.array([[args.var1, args.cov12],
                       [args.cov21, args.var2]])

print("Mean vector:\n", mean.reshape(-1, 1))
print("Covariance matrix:\n", cov_matrix)

if not np.allclose(cov_matrix, cov_matrix.T):
    print("Covariance matrix is not symmetric.")
else:
    print("Covariance matrix is symmetric.")

eigvals = np.linalg.eigvals(cov_matrix)
if np.all(eigvals >= 0):
    print("Positive semi-definite.")
else:
    print("NOT positive semi-definite.")

samples_builtin = np.random.multivariate_normal(mean, cov_matrix, args.N)

D_vals, U = eig(cov_matrix)
D = np.diag(D_vals)
sqrt_D = np.sqrt(D)
A = U @ sqrt_D

S = np.random.randn(2, args.N)
samples_manual = A @ S + mean.reshape(2, 1)

def plot_samples(samples, title):
    x, y = samples[0, :], samples[1, :]
    plt.scatter(x, y, alpha=0.5, label='Samples')

    x1, x2 = np.mgrid[-4:4:.01, -4:4:.01]
    pos = np.dstack((x1, x2))
    rv = multivariate_normal(mean, cov_matrix)
    plt.contour(x1, x2, rv.pdf(pos), colors='red')

    plt.plot(mean[0], mean[1], 'ko', label='Mean')
    max_ev_index = np.argmax(D_vals)
    u1 = U[:, max_ev_index]
    plt.quiver(mean[0], mean[1], u1[0], u1[1], angles='xy', scale_units='xy', scale=1, color='green', label='Eigenvector u1')
    
    plt.title(title)
    plt.axis('equal')
    plt.legend()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_samples(samples_builtin.T, 'Built-in Sampling with Contour')

plt.subplot(1, 2, 2)
plot_samples(samples_manual, 'Manual Sampling with Contour')

plt.tight_layout()
plt.show()

