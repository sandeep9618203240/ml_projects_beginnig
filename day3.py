import pandas as pd
import pandas as pd  # Import pandas for reading CSV files
import matplotlib.pyplot as plt
from sklearn import linear_model

# Load the dataset
file = pd.read_csv("homeprices.csv")

# Print the contents of the CSV file
print(file)

# Scatter plot of area vs price
plt.scatter(file['area'], file['price'], color='red', marker='o')

# Add labels and title
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Scatter Plot of Area vs Price')
plt.show()

# Create a LinearRegression model
find_cost = linear_model.LinearRegression()

# Fit the model using 'area' as the feature and 'price' as the target
find_cost.fit(file[['area']], file['price'])

# Predict the price of a house with 3300 sq ft area
dog = find_cost.predict([[4439]])

# Print the predicted cost
print("Cost of the house with {} sq ft area is: ".format(dog[0]))
print(dog)
