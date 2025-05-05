from scipy.stats import t
import math

# Given Data 
sample_data = [68, 72, 71, 69, 65, 74, 73, 68, 66, 70]
hypo_mean = 70
sample_size = len(sample_data)
sample_mean = sum(sample_data)/sample_size
alpha = 0.05

variance = sum([(x-sample_mean)**2 for x in sample_data])/(sample_size-1)
sample_std = math.sqrt(variance)

t_statistics = (sample_mean- hypo_mean)/ (sample_std/math.sqrt(sample_size))

df = sample_size-1
critical_t = t.ppf(1-alpha/2, df)
print(f"T-statistics: {t_statistics:.3f}")
print(f"Critical T-value: {critical_t:.3f}")

if abs(t_statistics) > critical_t:
    print("Reject the null hypothesis: The mean is significantly different")
else:
    print("Fail to reject the null hypothesis: The mean is not significantly different")

# # with 2 methods

# import scipy.stats as stats
# import numpy as np

# # Given data
# sample_mean = 69.6
# population_mean = 70
# sample_std_dev = 2.95
# sample_size = 10
# alpha = 0.05

# # Step 1: Calculate the t-value
# t_value = (sample_mean - population_mean) / (sample_std_dev / np.sqrt(sample_size))
# df = sample_size - 1  # Degrees of freedom

# # Method 1: Comparing t-value with critical t-value
# critical_t = stats.t.ppf(1 - alpha / 2, df)  # Two-tailed test
# print("Calculated t-value:", t_value)
# print("Critical t-value:", critical_t)

# if abs(t_value) > critical_t:
#     print("Reject the null hypothesis (H0).")
# else:
#     print("Fail to reject the null hypothesis (H0).")

# # Method 2: Comparing p-value with significance level
# p_value = 2 * (1 - stats.t.cdf(abs(t_value), df))  # Two-tailed p-value
# print("Calculated p-value:", p_value)

# if p_value < alpha:
#     print("Reject the null hypothesis (H0).")
# else:
#     print("Fail to reject the null hypothesis (H0).")
# import math
# from scipy.stats import t
# sample_data = [68, 72, 71, 69, 65, 74, 73, 68, 66, 70]
# hypo_mean = 70 
# sample_size = 10
# sample_mean = sum(sample_data)/sample_size
# variance = sum([(x - sample_mean)**2 for x in sample_data])/ sample_size-1
# sd_SA = math.sqrt(variance)
# alpha = 0.05
# print(sd_SA)
# df = sample_size-1

# t_statistics = (sample_mean - hypo_mean) / (sd_SA / math.sqrt(sample_size))

# # Critical
# t_critical = t.ppf(1-alpha/2, df)

# if t_statistics > t_critical:
#     print("Rejext h0")
# else:
#     print("hello")