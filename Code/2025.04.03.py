import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set_theme(style="darkgrid")

data = pd.read_csv('/Users/jung/Desktop/personal/CSDP720/CSDP720/Data/1.01.+Simple+linear+regression.csv')
print(data)
print(data.describe())

y = data['GPA']
x1 = data['SAT']

# Regression
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())

# Plot the data
plt.scatter(x1, y)
plt.xlabel('SAT')
plt.ylabel('GPA')
yhat = 0.0017*x1 + 0.275
fig = plt.plot(x1, yhat, lw=4, color='orange', label='Regression Line')
plt.show()