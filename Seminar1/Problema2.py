import numpy as np
import matplotlib.pyplot as plt

lambdas = [1, 2, 5, 10]
size = 1000

#A)
X1 = np.random.poisson(lam=1, size=size)
X2 = np.random.poisson(lam=2, size=size)
X3 = np.random.poisson(lam=5, size=size)
X4 = np.random.poisson(lam=10, size=size)

#B)
rand_lambda = np.random.choice(lambdas, size=size, replace=True)
rand_X = np.random.poisson(lam=rand_lambda)

#C)
plt.figure(figsize=(10, 6))

plt.hist(X1, bins=range(0, max(X4)+2), alpha=0.6, density=True, label="λ = 1")
plt.hist(X2, bins=range(0, max(X4)+2), alpha=0.6, density=True, label="λ = 2")
plt.hist(X3, bins=range(0, max(X4)+2), alpha=0.6, density=True, label="λ = 5")
plt.hist(X4, bins=range(0, max(X4)+2), alpha=0.6, density=True, label="λ = 10")

plt.hist(rand_X, bins=range(0, max(X4)+2), alpha=0.8, density=True, 
         label="Randomized λ ∈ {1,2,5,10}", color='black', histtype='step', linewidth=2)

plt.show()