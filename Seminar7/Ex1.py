import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])

x_bar = data.mean()
s_sd = data.std(ddof=1)
print(f"Sample mean = {x_bar:.4f}, Sample SD = {s_sd:.4f}")

with pm.Model() as model:
    mu = pm.Normal("mu", mu=x_bar, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=10)
    y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)
    trace = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42, cores=1)

summary = az.summary(trace, var_names=["mu", "sigma"], hdi_prob=0.95)
hdis = az.hdi(trace, var_names=["mu", "sigma"], hdi_prob=0.95)
print(summary)
print(hdis)

az.plot_posterior(trace, var_names=["mu", "sigma"], hdi_prob=0.95)
plt.show()
