import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

def main():
    df = pd.read_csv("Prices.csv")

    y  = df["Price"].astype(float).to_numpy()
    x1 = df["Speed"].astype(float).to_numpy()
    x2 = np.log(df["HardDrive"].astype(float).to_numpy())

    x1_mean, x2_mean = x1.mean(), x2.mean()
    x1_c = x1 - x1_mean
    x2_c = x2 - x2_mean

    with pm.Model() as model:
        α  = pm.Normal("α", mu=0, sigma=10)
        β1 = pm.Normal("β1", mu=0, sigma=10)
        β2 = pm.Normal("β2", mu=0, sigma=10)
        ϵ  = pm.HalfCauchy("ϵ", 5)

        x1_shared = pm.Data("x1_shared", x1_c)
        x2_shared = pm.Data("x2_shared", x2_c)

        µ = α + β1 * x1_shared + β2 * x2_shared

        y_obs = pm.Normal("y_obs", mu=µ, sigma=ϵ, observed=y)

        idata = pm.sample(
            3000, tune=2000, target_accept=0.9,
            random_seed=42, return_inferencedata=True
        )

    print(az.hdi(idata, var_names=["β1", "β2"], hdi_prob=0.95))

    summ = az.summary(idata, var_names=["β1", "β2"], hdi_prob=0.95)[["hdi_2.5%", "hdi_97.5%"]]
    print("β1 useful?", not (summ.loc["β1","hdi_2.5%"] <= 0 <= summ.loc["β1","hdi_97.5%"]))
    print("β2 useful?", not (summ.loc["β2","hdi_2.5%"] <= 0 <= summ.loc["β2","hdi_97.5%"]))

    x1_new = 33.0
    x2_new = np.log(540.0)
    x1_new_c = x1_new - x1_mean
    x2_new_c = x2_new - x2_mean

    α_s  = idata.posterior["α"].values.flatten()
    β1_s = idata.posterior["β1"].values.flatten()
    β2_s = idata.posterior["β2"].values.flatten()
    ϵ_s  = idata.posterior["ϵ"].values.flatten()

    µ_samples = α_s + β1_s * x1_new_c + β2_s * x2_new_c
    print("90% HDI for µ:", az.hdi(µ_samples, hdi_prob=0.90))

    rng = np.random.default_rng(42)
    y_tilde = rng.normal(loc=µ_samples, scale=ϵ_s)
    print("90% HDI for y~:", az.hdi(y_tilde, hdi_prob=0.90))

if __name__ == "__main__":
    main()
