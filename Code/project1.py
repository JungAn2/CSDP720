import pandas as pd
from scipy.stats import skew, kurtosis, norm

# Part 1. Statistic calculations

# Load the data
data = pd.read_csv('/Users/jung/Desktop/personal/CSDP720/CSDP720/Data/sampled_ratings_Electronics.csv', header=None)
# Rating 1 = user_id, 2 = item_id, 3 = rating, 4 = timestamp
ratings = data[2]

# Calculate statistics
mean_rating = ratings.mean()
median_rating = ratings.median()
std_dev_rating = ratings.std(ddof=1)  # Use sample standard deviation
range_rating = ratings.max() - ratings.min()
skewness_rating = skew(ratings, bias=False)  # Use sample skewness
kurtosis_rating = kurtosis(ratings, bias=False)  # Use sample kurtosis

# Print statistics
print(f"Mean: {mean_rating}")
print(f"Median: {median_rating}")
print(f"Standard Deviation: {std_dev_rating}")
print(f"Range: {range_rating}")
print(f"Skewness: {skewness_rating}")
print(f"Kurtosis: {kurtosis_rating}")

# Graph the data
import matplotlib.pyplot as plt
import seaborn as sns

# Create a histogram from 0 to 5 with 1 interval
plt.hist(ratings, bins=5, range=(1, 6), edgecolor='black')
plt.title('Amazon Electronics Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Part 2. Confidence Interval

# Calculate the confidence interval
n = len(ratings)
confidence_level = 0.95
z = 1.96  # 95% confidence level
margin_of_error = z * std_dev_rating / n**0.5
lower_bound = mean_rating - margin_of_error
upper_bound = mean_rating + margin_of_error

# Print the confidence interval
print()
print(f"Confidence Interval: ({lower_bound}, {upper_bound})")

# Part 3. Hypothesis Testing

# Load the data
data = pd.read_csv('/Users/jung/Desktop/personal/CSDP720/CSDP720/Data/sampled_ratings_Electronics.csv', header=None)
ratings = data[2]   
ratings = ratings.dropna()

# Perform the hypothesis test
alpha = 0.05
null_mean = 4.0
# Use z-test since the sample size is large
z_stat = (mean_rating - null_mean) / (std_dev_rating / n**0.5)
z_critical = norm.ppf(1 - alpha / 2)

print()
print(f"Z-statistic: {z_stat}")
print(f"Z-critical: {z_critical}")

# Print the results
if abs(z_stat) > z_critical:
    print("Reject the null hypothesis")
else:
    print("Fail to reject the null hypothesis")

# Part 4. Normal Distribution using shapiro-wilk test

from scipy.stats import shapiro

# Perform the Shapiro-Wilk test
statistic, p_value = shapiro(ratings)

print()
print(f"Statistic: {statistic}")
print(f"P-value: {p_value}")

# Print the results
if p_value < alpha:
    print("The data does not come from a normal distribution")
else:
    print("The data comes from a normal distribution")
