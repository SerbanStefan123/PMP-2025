import pymc as pm
import numpy as np
import pandas as pd
import pytensor as pt
import seaborn as sns
import scipy.stats as stats
from scipy.special import expit as logistic
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

df = pd.read_csv('Seminar12/date_promovare_examen.csv')
df.head()

y_1 = df['Promovare'].astype(int).values
x_n = ['Ore_Studiu', 'Ore_Somn']
x_1 = df[x_n].values.astype(float)

counts = pd.Series(y_1).value_counts().sort_index()
prop = counts / len(y_1)

print("Counts (0/1):")
print(counts)
print("\nProportions (0/1):")
print(prop)

plt.bar(['0', '1'], [counts.get(0, 0), counts.get(1, 0)])
plt.xlabel('Promovare')
plt.ylabel('Count')
plt.show()

x_1 = x_1 - x_1.mean(axis=0)

with pm.Model() as model_1:
    α = pm.Normal('α', mu=0, sigma=10)
    β = pm.Normal('β', mu=0, sigma=2, shape=len(x_n))
    µ = α + pm.math.dot(x_1, β)
    θ = pm.Deterministic('θ', 1 / (1 + pm.math.exp(-µ)))
    bd = pm.Deterministic('bd', -α/β[1] - β[0]/β[1] * x_1[:,0])
    yl = pm.Bernoulli('yl', p=θ, observed=y_1)
    idata_1 = pm.sample(2000, return_inferencedata=True)

az.summary(idata_1, var_names=['α', 'β'])

idx = np.argsort(x_1[:,0])
bd_m = idata_1.posterior['bd'].mean(("chain", "draw"))[idx]
plt.scatter(x_1[:,0], x_1[:,1], c=[f'C{x}' for x in y_1])
plt.plot(x_1[:,0][idx], bd_m, color='k');
az.plot_hdi(x_1[:,0], idata_1.posterior['bd'], color='k')
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
plt.show()

az.plot_posterior(idata_1, var_names=['β'], hdi_prob=0.94)
plt.show()

posterior_1 = idata_1.posterior.stack(samples=("chain", "draw"))
β_st = posterior_1['β'].sel(β_dim_0=0).values
β_sl = posterior_1['β'].sel(β_dim_0=1).values

print(f"P(β_studiu > 0) = {(β_st > 0).mean():.3f}")
print(f"P(β_somn   > 0) = {(β_sl > 0).mean():.3f}")
print(f"P(|β_studiu| > |β_somn|) = {(np.abs(β_st) > np.abs(β_sl)).mean():.3f}")
