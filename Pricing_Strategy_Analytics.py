import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('pricing_data.csv')


le = LabelEncoder()
data['Category_encoded'] = le.fit_transform(data['Category'])

X = data[['Price', 'Category_encoded']]
y = data['UnitsSold']

model = LinearRegression()
model.fit(X, y)

# Predict UnitsSold for a new product
new_data = pd.DataFrame({'Price': [200], 'Category_encoded': [le.transform(['Clothing'])[0]]})
predicted_units = model.predict(new_data)[0]
print(f"Predicted Units Sold for new product: {predicted_units:.2f}")


kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Price', 'UnitsSold']])


plt.scatter(data['Price'], data['UnitsSold'], c=data['Cluster'])
plt.xlabel('Price')
plt.ylabel('Units Sold')
plt.title('Pricing Strategy Clusters')
plt.show()
