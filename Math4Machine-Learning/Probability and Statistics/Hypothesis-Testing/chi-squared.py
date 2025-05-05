
# import numpy as np
# from scipy.stats import chi2

# # Observed Frequencies (O)
# observed = np.array([[30, 50, 20],  # Product A
#                      [20, 40, 40]])  # Product B

# # Step 1: Calculate the expected frequencies (E)
# row_totals = observed.sum(axis=1)
# col_totals = observed.sum(axis=0)
# grand_total = observed.sum()

# expected = np.outer(row_totals, col_totals) / grand_total

# # Step 2: Calculate the Chi-Square statistic (χ²)
# chi_square_stat = ((observed - expected) ** 2 / expected).sum()

# # Step 3: Degrees of freedom
# rows, cols = observed.shape
# df = (rows - 1) * (cols - 1)

# # Step 4: Method 1 - Compare χ² value with critical value
# critical_value = chi2.ppf(1 - 0.05, df)
# print("Chi-Square Statistic:", chi_square_stat)
# print("Critical Value:", critical_value)

# if chi_square_stat > critical_value:
#     print("Reject the null hypothesis (H0).")
# else:
#     print("Fail to reject the null hypothesis (H0).")

# # Step 5: Method 2 - Compare p-value with significance level
# p_value = 1 - chi2.cdf(chi_square_stat, df)
# print("P-value:", p_value)

# if p_value < 0.05:
#     print("Reject the null hypothesis (H0).")
# else:
#     print("Fail to reject the null hypothesis (H0).")

### PRACTICE ###
import numpy as np
from scipy.stats import chi2

alpha = 0.05

# Observed data
observed_data = np.array([[30,50,20],  # Product A
                         [20,40,40]])  # Product B

# Expected Frequencies
row_totals = observed_data.sum(axis=1)
col_totals = observed_data.sum(axis=0)
grand_total = observed_data.sum()

expected_data = (np.outer(row_totals, col_totals))/grand_total

# step 3 calculate chi - statistics
chi_statistics = ((expected_data - observed_data)**2 /expected_data).sum()
print("Chi_Statistics: ", chi_statistics)

#step 4 degree of freedom
df = (observed_data.shape[0]-1)*(observed_data.shape[1]-1)
print("df: ",df)

# step 4 calculate chi critical
chi_critical = chi2.ppf(1-alpha, df)
print("Chi_critical: " , chi_critical )

# Decision Rule
if chi_statistics>chi_critical:
    print("Reject H0")
else: 
    print("Failed to reject H0")