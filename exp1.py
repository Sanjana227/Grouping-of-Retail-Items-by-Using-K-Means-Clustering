import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your sales data into a Pandas DataFrame
# Replace 'your_data.csv' with the actual file path to your sales data
data = pd.read_csv('sales_data.csv')

# Extract relevant features for clustering
# Assuming 'year', 'month', 'item_count', and 'transaction_value' are relevant features
features = data[['year', 'month', 'item_count', 'transaction_value']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Analyze the results
centroid_values = scaler.inverse_transform(kmeans.cluster_centers_)

# Identify 'fast moving' and 'slow moving' groups based on centroids
fast_moving_cluster = centroid_values.argmax()
slow_moving_cluster = centroid_values.argmin()

# Assign 'fast moving' and 'slow moving' labels to the clusters
data['movement_category'] = 'Slow Moving'
data.loc[data['cluster'] == fast_moving_cluster, 'movement_category'] = 'Fast Moving'

# Display the resulting data with movement categories
print(data[['year', 'month', 'item_count', 'transaction_value', 'movement_category']])
