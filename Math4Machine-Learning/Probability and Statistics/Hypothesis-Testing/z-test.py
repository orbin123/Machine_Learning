# A company claims the average weight of its packaged sugar is 1 kg (1000 grams).
# A sample of 50 packets has an average weight of 990 grams, 
# with a population standard deviation of 20 grams. 
# Test the claim at a 5% significance level.

import numpy as np
from scipy.stats import norm

# Given Data
population_mean = 1000 # Hypothesised mean
sample_mean = 990 # Sample Mean 
sample_size = 50  # Sample size (n)
standard_diviation = 20 # Population standard deviation (σ)
alpha = 0.05 # Significance level

# Step 1: Calculate the Z-Statistic
# Formula: Z = (X̄ - μ) / (σ / √n)
z_stat = (sample_mean-population_mean)/ (standard_diviation/np.sqrt(sample_size))
print(f"z-statistics: {z_stat:.2f}")

# Step 2: Determine the critical Z-value
# For a two-tailed test, find the critical value using the alpha level
# At alpha = 0.05, critical z-values for a two-tailed test are ±1.96 (looked up manually or in a z-table)
z_critical = norm.ppf(1-alpha/2)
print(f"z-critical: +-{z_critical:.2f}")

# Step 3: Compare Z-Statistic with Critical Values
# If |Z-Statistic| > Critical Z-Value, reject the null hypothesis (H0)
if abs(z_stat) > z_critical:
    print("Reject the null hypothesis (H0)")
    print("There is enough evidence to conclude that the average weight of sugar is not 1000grams")
else:
    print("Fail to reject the null hypothesis (H0)")
    print("There is not enough evidence to conclude that the average weight of sugar is not 1000grams")
# import numpy as np   
# population_mean = 1000
# sample_mean = 990
# s_dP = 20
# sample_size = 50
# alpha = 0.05

# # Cal z statistics
# z_stat = (sample_mean - population_mean)/(s_dP/np.sqrt(sample_size))
# print(f"Z-STAT = {z_stat}")

# from scipy.stats import norm
# z_crit = norm.ppf(1-alpha/2)
# print(f"Z-CRIT = {z_crit}")

# if abs(z_stat) > z_crit:
#     print("reject h0")
# else:
#     print("Failed to reject H0")