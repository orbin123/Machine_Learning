# Example: Medical test
# P(Disease) = 0.01, P(Positive | Disease) = 0.95, P(Positive | No Disease) = 0.05
# P(A|B) = (P(B|A) * P(A))/P(B) # Bayes theorem
# P(B) = (P(B|A) * P(A)) + P(B| Not A) * P(Not A)

# Probabilities
P_Disease = 0.01
P_NoDisease = 1 - P_Disease
P_Positive_given_Disease = 0.95
P_Positive_given_NoDisease= 1 - P_Positive_given_Disease

# calculate P(Positive)
P_Positive = (P_Positive_given_Disease * P_Disease) + (P_Positive_given_NoDisease * P_NoDisease) 

# Calculate P(Disease|Positive)
P_Disease_given_Positive = ((P_Positive_given_Disease * P_Disease)/P_Positive)

print(f"Probabilty of P(Disease|Positive) = {P_Disease_given_Positive * 100:.2f} %")