# import numpy as np
# from scipy.stats import f

# # Sales Data for Strategies A, B, C
# strategy_A = np.array([30, 32, 35, 40, 45])
# strategy_B = np.array([50, 55, 53, 57, 60])
# strategy_C = np.array([70, 75, 80, 78, 77])

# # Step 1: Group Means
# mean_A = np.mean(strategy_A)
# mean_B = np.mean(strategy_B)
# mean_C = np.mean(strategy_C)

# # Step 2: Overall Mean
# overall_mean = np.mean(np.concatenate([strategy_A, strategy_B, strategy_C]))

# # Step 3: Calculate SSB (Sum of Squares Between Groups)
# n_A = len(strategy_A)
# n_B = len(strategy_B)
# n_C = len(strategy_C)

# SSB = n_A * (mean_A - overall_mean)**2 + n_B * (mean_B - overall_mean)**2 + n_C * (mean_C - overall_mean)**2

# # Step 4: Calculate SSW (Sum of Squares Within Groups)
# SSW = np.sum((strategy_A - mean_A)**2) + np.sum((strategy_B - mean_B)**2) + np.sum((strategy_C - mean_C)**2)

# # Step 5: Degrees of Freedom
# df_between = 3 - 1  # k - 1
# df_within = len(strategy_A) + len(strategy_B) + len(strategy_C) - 3  # N - k

# # Step 6: Mean Squares
# MSB = SSB / df_between
# MSW = SSW / df_within

# # Step 7: F-Statistic
# F_stat = MSB / MSW

# # Step 8: Method 1 - Compare F-Statistic with Critical F-Value
# alpha = 0.05
# critical_F = f.ppf(1 - alpha, df_between, df_within)

# print("F-Statistic:", F_stat)
# print("Critical F-Value:", critical_F)
# if F_stat > critical_F:
#     print("Reject the null hypothesis (H0).")
# else:
#     print("Fail to reject the null hypothesis (H0).")

# # Step 9: Method 2 - Compare P-Value with Significance Level
# p_value = 1 - f.cdf(F_stat, df_between, df_within)

# print("P-Value:", p_value)
# if p_value < alpha:
#     print("Reject the null hypothesis (H0).")
# else:
#     print("Fail to reject the null hypothesis (H0).")

    ##### PRACTICE #####

import numpy as np
from scipy.stats import f

# Data
strategy_A = np.array([30, 32, 34, 35, 36])
strategy_B = np.array([40, 42, 44, 45, 46])
strategy_C = np.array([50, 52, 54, 55, 56])

# Step 1 group mean
mean_A = np.mean(strategy_A)
mean_B = np.mean(strategy_B)
mean_C = np.mean(strategy_C)

# Step 2 Overall Mean
overall_mean = np.mean(np.concatenate([strategy_A,strategy_B,strategy_C]))

# Sum of squares
n_A = len(strategy_A)
n_B = len(strategy_B)
n_C = len(strategy_C)

# SSB =E n *  (mean - overall )**2
SSB = n_A * (mean_A - overall_mean)**2 + n_B * (mean_B - overall_mean)**2 + n_C * (mean_C - overall_mean)**2 

# SSW = E (group value - group mean)**2
SSW = np.sum((strategy_A-mean_A)**2) + np.sum((strategy_B-mean_B)**2) + np.sum((strategy_C-mean_C)**2) 

# Degree of freedom
k = 3 # no of groups
N = n_A+n_B+n_C
df_betweeen = k-1
df_within = N-k

# Mean of squares
MSB = SSB/df_betweeen
MSW  = SSW/df_within

# F- statistics
f_statistics = MSB/MSW

# Calculate F _ Crirtical
alpha = 0.05
f_critical = f.ppf(1-alpha, df_betweeen, df_within)

# Decision Rule 
if f_statistics > f_critical:
    print("Reject H0")
else: 
    print('fail to reject H0')

