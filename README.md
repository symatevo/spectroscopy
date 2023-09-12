# Absorption Spectroscopy Analysis

## Introduction

This repository contains Python code for the analysis of Absorption Spectroscopy data obtained during a class on Absorption Spectroscopy of DNA in SSC (Saline-Sodium Citrate) solution. Absorption spectroscopy is a technique that measures the absorption of light by a substance as a function of wavelength or frequency. In this analysis, we will focus on temperature-dependent absorption data.

## Data

The data used in this analysis was collected during the class and is stored in the file `080923nucl.csv`. The dataset consists of two columns:

- `Temperature`: Temperature in degrees Celsius.
- `Absorption`: Absorption values corresponding to each temperature point.

Here is a snippet of the dataset:

```
Temperature  Absorption
0      38.3070    0.351907
1      38.7333    0.351826
2      39.1739    0.352305
...
65     66.8499    0.465379
```

## Data Visualization

### Scatter Plot

To visualize the relationship between temperature and absorption, we create a scatter plot. The following code generates this plot:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('080923nucl.csv', names=['Temperature', 'Absorption'])

# Extract the Temperature and Absorption columns
temperature = data['Temperature']
absorption = data['Absorption']

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(temperature, absorption, marker='o', color='b', label='Data Points')
plt.title('Temperature vs. Absorption')
plt.xlabel('Temperature (°C)')
plt.ylabel('Absorption')
plt.legend()
plt.grid(True)
plt.show()
```

### Color-Mapped Scatter Plot

We also create a color-mapped scatter plot to visualize the absorption values in relation to temperature. The code for this plot is as follows:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
data = pd.read_csv('080923nucl.csv', names=['Temperature', 'Absorption'])

# Extract the Temperature and Absorption columns
temperature = data['Temperature']
absorption = data['Absorption']

# Find the index of the maximum absorption value
max_absorption_index = absorption.idxmax()
max_absorption_temperature = temperature[max_absorption_index]
max_absorption_value = absorption[max_absorption_index]

# Create a color map based on absorption values
colors = absorption / max(absorption)

# Create a scatter plot with custom colors
plt.figure(figsize=(10, 6))
scatter = plt.scatter(temperature, absorption, c=colors, cmap='viridis', marker='o', label='Data Points')
plt.title('Temperature vs. Absorption', fontsize=16)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('Absorption', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

# Add a colorbar to indicate absorption values
cbar = plt.colorbar(scatter)
cbar.set_label('Absorption Value', fontsize=16)
cbar.ax.tick_params(labelsize=10)

# Highlight the maximum absorption point
plt.scatter(max_absorption_temperature, max_absorption_value, color='red', marker='o', s=70, label='Max Absorption')

plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
```

## Analysis

### Polynomial Regression

We perform polynomial regression to analyze the data and predict absorption values at specific temperatures. In this case, we use polynomial features with a degree of 7 to fit the data accurately. The code for this analysis is as follows:

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv('080923nucl.csv', names=['Temperature', 'Absorption'])

# Split the dataset into features and target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the features to include polynomial terms up to degree 7
poly = PolynomialFeatures(degree=7)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)

# Fit the transformed features to Linear Regression
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)

# Predict new results with the model
y_pred = poly_reg.predict(X_val_poly)

# Calculate the R^2 value for the validation set
r2 = r2_score(y_val, y_pred)

print(f'R^2 score: {r2:.4f}')
```

### Predicted Absorption Values

We also predict absorption values at specific temperatures using the trained polynomial regression model. Here are the predicted absorption values for temperatures `[66, 67, 68, 69, 70, 71, 72, 73]`:

```
[0.46725664, 0.4618115, 0.44107114, 0.39804564, 0.32390317, 0.20767238, 0.03591778, -0.207613]
```

## Conclusion

This analysis provides insights into the temperature-dependent absorption of DNA in SSC solution. We visualized the data, performed polynomial regression, and made predictions for absorption values at specific temperatures. The R^2 score indicates the goodness of fit of the polynomial regression model. The results can be further analyzed and interpreted to draw meaningful conclusions about the behavior of DNA in the given conditions.
