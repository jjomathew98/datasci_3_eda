# datasci_3_eda

## Objective:
Engage in the critical phase of Exploratory Data Analysis (EDA) using the tools and techniques from Python to uncover patterns, spot anomalies, test hypotheses, and identify the main structures of your dataset.

### Instructions:

#### 1. Univariate Analysis:

##### Loading dataset
First, we load a dataset from a prior assignment. I chose a dataset from the Week 2 assignment folder: https://github.com/hantswilliams/HHA_507_2023/blob/main/WK2/data/Hospital_Cost_Report_2019.csv to google colab

df = pd.read_csv('https://raw.githubusercontent.com/hantswilliams/HHA_507_2023/main/WK2/data/Hospital_Cost_Report_2019.csv')
df

##### 1.2 Calculating measures of central tendency (mean, median, mode) and measures of spread (range, variance, standard deviation, IQR).
Then, we manually perform a univariate analysis to understand the distribution of each variable. This includes calculating measures of central tendency (mean, median, mode) and measures of spread (range, variance, standard deviation, IQR).

mean_value_chargeratio = df['Cost To Charge Ratio'].mean()
median_value_netrevenue = df['Net Revenue from Medicaid'].median()
mode_value_MedicaidCharges = df['Medicaid Charges'].mode().iloc[0]
variance_chargeratio = np.var(df['Cost To Charge Ratio'])
std_deviation_chargeration = np.std(df['Cost To Charge Ratio'])
percentile_25_Net_Revenue_from_standalone_SCHIP = np.percentile(df['Net Revenue from Stand-Alone SCHIP'], 25)
percentile_75_Net_Revenue_from_standalone_SCHIP = np.percentile(df['Net Revenue from Stand-Alone SCHIP'], 75)
df_range_Stand_Alone_SCHIP_Charges = df['Stand-Alone SCHIP Charges'].max() - df['Stand-Alone SCHIP Charges'].min()
correlation_matrix_1 = df[['Cost To Charge Ratio', 'Net Revenue from Medicaid', 'Medicaid Charges']].corr()
covariance_matrix_2 = df[['Net Revenue from Stand-Alone SCHIP', 'Stand-Alone SCHIP Charges']].cov()

##### 1.3 Visualize the distribution of select numerical variables using histograms.
to provide the visualise we need load a package,Installing matplotlib to visualize the distribution of select numerical variables using histograms

import matplotlib.pyplot as plt

###### Sample data 
mean_cost_to_charge_ratio = 1.031929805371989
median_net_revenue_medicaid = 7086967.0
mode_medicaid_charges = 1034208.0
cost_to_charge_ratio = 1133.9190893943135
std_dev_cost_to_charge_ratio = 33.6737151112602
data_range_schip_charges = 328464138.0

###### Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

###### Plot histograms
axes[0, 0].hist(mean_cost_to_charge_ratio, bins=20, color='skyblue', alpha=0.7)
axes[0, 0].set_title('Mean Cost To Charge Ratio')

axes[0, 1].hist(median_net_revenue_medicaid, bins=20, color='lightcoral', alpha=0.7)
axes[0, 1].set_title('Median Net Revenue from Medicaid')

axes[1, 0].hist(mode_medicaid_charges, bins=20, color='lightgreen', alpha=0.7)
axes[1, 0].set_title('Mode Medicaid Charges')

axes[1, 1].hist(data_range_schip_charges, bins=20, color='gold', alpha=0.7)
axes[1, 1].set_title('Data Range (Stand-Alone SCHIP Charges)')

###### Add labels and adjust spacing
plt.tight_layout()
plt.show()

#### 2. Bivariate Analysis:
##### 2.1 Analyze the relationship between pairs of variable
Here, we analyze the relationship between Net Revenue from Medicaid and Medicaid charges. Then, use scatter plots to explore potential relationships between two numerical variables.

###### Specify the actual column names you want to use for the scatter plot
x_variable = 'Net Revenue from Medicaid'
y_variable = 'Medicaid Charges'

###### Create a scatter plot
plt.scatter(df[x_variable], df[y_variable])
plt.xlabel(x_variable)
plt.ylabel(y_variable)
plt.title(f'Scatter Plot: {x_variable} vs {y_variable}')
plt.show()

##### 2.2 Computing correlation coefficients for numerical variables

correlation_coefficient = df[x_variable].corr(df[y_variable])
print(f'Correlation coefficient between {x_variable} and {y_variable}: {correlation_coefficient}')

#### 3. Handling Outliers:
Here, we identify outliers in the dataset using the IQR method or visualization tools, Decide on an approach to handle these outliers (e.g., remove, replace, or retain) 

column_to_analyze = 'Net Revenue from Medicaid'

###### Calculate the IQR
Q1 = df[column_to_analyze].quantile(0.25)
Q3 = df[column_to_analyze].quantile(0.75)
IQR = Q3 - Q1

###### Determine lower and upper bounds for identifying outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

###### Identify outliers
outliers = df[(df[column_to_analyze] < lower_bound) | (df[column_to_analyze] > upper_bound)]

###### Decide on an approach to handle outliers
df_cleaned = df[(df[column_to_analyze] >= lower_bound) & (df[column_to_analyze] <= upper_bound)]

Justify your decision in a markdown cell
For instance, removing outliers can improve the reliability of statistical analyses, especially when the outliers are due to data entry errors or other anomalies.

###### Check if there are no outliers based on z-scores >= 1
z_scores = (df[column_to_analyze] - df[column_to_analyze].mean()) / df[column_to_analyze].std()
no_outliers = df[abs(z_scores) < 1]

###### Print the results
print("Outliers:")
print(outliers)
print("\nDataFrame after removing outliers:")
print(df_cleaned)
print("\nDataFrame with no outliers based on z-scores:")
print(no_outliers)

#### 4. Automated Analysis:
we need to use the automated EDA tool for pandas profiling to analyze it

import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_csv('https://raw.githubusercontent.com/hantswilliams/HHA_507_2023/main/WK2/data/Hospital_Cost_Report_2019.csv')
profile = ProfileReport(df, minimal=True)
profile.to_file('eda_report.html')
