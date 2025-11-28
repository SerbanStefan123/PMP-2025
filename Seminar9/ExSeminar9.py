import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

mu_prior = 10.0

print("Model: n ~ Poisson(10),  Y | n, θ ~ Binomial(n, θ)")
print("Rulăm toate combinațiile Y ∈ {0, 5, 10}, θ ∈ {0.2, 0.5}.\n")

results = {}

for y_obs in Y_values:
    for theta in theta_values:
        print(f"Scenariu: Y = {y_obs}, θ = {theta}")

        with pm.Model() as model:
            n = pm.Poisson("n", mu=mu_prior)
            Y = pm.Binomial("Y", n=n, p=theta, observed=y_obs)
            Y_future = pm.Binomial("Y_future", n=n, p=theta)
            step = pm.Metropolis()
            idata = pm.sample(
                draws=2000,
                tune=2000,
                chains=2,
                cores=1,
                step=step,
                random_seed=2025,
                progressbar=True,
                return_inferencedata=True,
            )
            idata = pm.sample_posterior_predictive(
                idata,
                model=model,
                var_names=["Y_future"],
                extend_inferencedata=True,
                random_seed=2025,
                progressbar=True,
            )

        results[(y_obs, theta)] = idata
        print("  Sampling pentru acest scenariu a fost finalizat.\n")

fig_post, axes_post = plt.subplots(
    nrows=len(Y_values),
    ncols=len(theta_values),
    figsize=(10, 8),
    sharex=True,
    sharey=True,
)

for i, y_obs in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        ax = axes_post[i, j]
        idata = results[(y_obs, theta)]

        az.plot_posterior(
            idata,
            var_names=["n"],
            ax=ax,
            hdi_prob=0.94,
            point_estimate="mean",
        )
        ax.set_title(f"Posterior n | Y={y_obs}, θ={theta}")

fig_post.suptitle("Distribuțiile a posteriori pentru n", fontsize=14)
fig_post.tight_layout(rect=[0, 0.03, 1, 0.95])

fig_pred, axes_pred = plt.subplots(
    nrows=len(Y_values),
    ncols=len(theta_values),
    figsize=(10, 8),
    sharex=True,
    sharey=True,
)

for i, y_obs in enumerate(Y_values):
    for j, theta in enumerate(theta_values):
        ax = axes_pred[i, j]
        idata = results[(y_obs, theta)]

        y_future_samples = idata.posterior_predictive["Y_future"].values.ravel()

        az.plot_dist(
            y_future_samples,
            ax=ax,
            kind="hist",
        )
        ax.set_xlabel("Y*")
        ax.set_title(f"Predictivă Y* | Y={y_obs}, θ={theta}")

fig_pred.suptitle("Distribuțiile predictiv-posteriori pentru Y*", fontsize=14)
fig_pred.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()
