

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


df = pd.read_csv(r"C:\Users\THANOJ\Downloads\road_accidents_india_12k.csv")

print("Shape of data:", df.shape)
print(df.head())


print("\nMissing Values:\n", df.isnull().sum())

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)


print("\n===== BASIC STATISTICS =====")
print("Mean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True).iloc[0])
print("\nStandard Deviation:\n", df.std(numeric_only=True))


Q1 = df['Total_Accidents'].quantile(0.25)
Q3 = df['Total_Accidents'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Total_Accidents'] < lower) | (df['Total_Accidents'] > upper)]
print("\nNumber of Outliers:", len(outliers))


yearly = df.groupby('Year')['Total_Accidents'].sum()
state_data = df.groupby('State')['Total_Accidents'].sum().head(10)


# 1. BAR CHART
plt.figure()
yearly.plot(kind='bar')
plt.title("Year-wise Accidents (Bar Chart)")
plt.show()

# 2. LINE CHART
plt.figure()
yearly.plot(kind='line', marker='o')
plt.title("Trend Over Years (Line Chart)")
plt.show()

# 3. PIE CHART
plt.figure()
state_data.plot(kind='pie', autopct='%1.1f%%')
plt.title("Top States (Pie Chart)")
plt.ylabel("")
plt.show()

# 4. HISTOGRAM
plt.figure()
sns.histplot(df['Total_Accidents'], kde=True)
plt.title("Histogram")
plt.show()

# 5. BOXPLOT
plt.figure()
sns.boxplot(x=df['Total_Accidents'])
plt.title("Boxplot")
plt.show()

# 6. SCATTER PLOT
plt.figure()
plt.scatter(df['Year'], df['Total_Accidents'])
plt.title("Scatter Plot")
plt.xlabel("Year")
plt.ylabel("Accidents")
plt.show()

# 7. HEATMAP
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# 8. COUNT PLOT
plt.figure()
sns.countplot(x='Year', data=df)
plt.title("Count of Records per Year")
plt.xticks(rotation=45)
plt.show()

# 9. COLUMN CHART
plt.figure()
plt.bar(yearly.index, yearly.values)
plt.title("Column Chart")
plt.xlabel("Year")
plt.ylabel("Accidents")
plt.show()

# 10. HORIZONTAL BAR
plt.figure()
state_data.sort_values().plot(kind='barh')
plt.title("Horizontal Bar (States)")
plt.show()



# 15. YEAR-WISE BOXPLOT
plt.figure()
sns.boxplot(x='Year', y='Total_Accidents', data=df)
plt.title("Year-wise Boxplot")
plt.xticks(rotation=45)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Prepare data (Year vs Total Accidents)
yearly_df = df.groupby('Year')['Total_Accidents'].sum().reset_index()

X = yearly_df[['Year']]
y = yearly_df['Total_Accidents']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions for existing years
y_pred = model.predict(X)

future_years = pd.DataFrame({
    'Year': [yearly_df['Year'].max() + i for i in range(1, 6)]
})

future_predictions = model.predict(future_years)

# Print predictions
print("\n===== FUTURE PREDICTIONS =====")
for year, pred in zip(future_years['Year'], future_predictions):
    print(f"Year {year}: {int(pred)} accidents")


plt.figure()
future_years = pd.DataFrame({
    'Year': [yearly_df['Year'].max() + i for i in range(1, 6)]
})

future_predictions = model.predict(future_years)

# Print predictions
print("\n===== FUTURE PREDICTIONS =====")
for year, pred in zip(future_years['Year'], future_predictions):
    print(f"Year {year}: {int(pred)} accidents")


plt.figure()

