import matplotlib.pyplot as plt
import pandas as pd
# Data for plotting
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
file=pd.read_csv("homeprices.csv")
# Create the plot
plt.scatter(file.area,file.price,color='red',marker='o')

# Add title and labels
plt.title('Simple Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Display the plot
plt.show()
