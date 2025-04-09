y = data['GPA']
x1 = data[['SAT', 'Rand 1,2,3']]

# Regression
x = sm.add_constant(x1)
results = sm.OLS(y, x).fit()
print(results.summary())

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(data['SAT'], y, alpha=0.5, label='SAT vs GPA')
plt.scatter(data['Rand 1,2,3'], y, alpha=0.5, label='Rand 1,2,3 vs GPA', color='orange')
plt.xlabel('Independent Variables')
plt.ylabel('GPA')
plt.legend()

# Regression line for SAT
yhat_sat = results.params['SAT'] * data['SAT'] + results.params['const']
plt.plot(data['SAT'], yhat_sat, color='blue', label='Regression Line (SAT)')

# Regression line for Rand 1,2,3
yhat_rand = results.params['Rand 1,2,3'] * data['Rand 1,2,3'] + results.params['const']
plt.plot(data['Rand 1,2,3'], yhat_rand, color='green', label='Regression Line (Rand 1,2,3)')

plt.legend()
plt.show()