import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

videogames_df = pd.read_csv(r"C:\Users\manikanta\Downloads\videogames.csv")

median_year = videogames_df['Year'].median()
videogames_df['Year']=videogames_df['Year'].fillna(median_year)

videogames_df['Publisher']=videogames_df['Publisher'].fillna('Unknown')

videogames_encoded = pd.get_dummies(videogames_df, columns=['Platform', 'Publisher'])

#Bar graph of genre distribution
genre_distribution = videogames_df['Genre'].value_counts().reset_index()
genre_distribution.columns = ['Genre', 'Count']

plt.figure(figsize=(12, 6))
sns.barplot(data=genre_distribution, x='Genre', y='Count', palette='viridis',hue='Genre')
plt.title('Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

#Publisher Performance Analysis
publisher_performance = videogames_df.groupby(['Publisher', 'Year']).agg({'Global_Sales': 'sum'}).reset_index()

plt.figure(figsize=(14, 8))
sns.lineplot(data=publisher_performance[publisher_performance['Publisher'].isin(videogames_df['Publisher'].value_counts().index[:5])],
             x='Year', y='Global_Sales', hue='Publisher', errorbar=None)
plt.title('Top Publishers Global Sales Performance Over the Years')
plt.xlabel('Year')
plt.ylabel('Global Sales (in millions)')
plt.legend(title='Publisher')
plt.show()

#Time Series Analysis
yearly_sales = videogames_df.groupby('Year')['Global_Sales'].sum().reset_index()

arima_model = sm.tsa.ARIMA(yearly_sales['Global_Sales'], order=(1, 1, 1))
arima_result = arima_model.fit()

forecast = arima_result.forecast(steps=5)
forecast_years = range(int(yearly_sales['Year'].max()) + 1, int(yearly_sales['Year'].max()) + 1 + len(forecast))

plt.figure(figsize=(10, 6))
plt.plot(yearly_sales['Year'], yearly_sales['Global_Sales'], label='Historical Sales')
plt.plot(forecast_years, forecast, label='Forecast', linestyle='--')
plt.title('Time Series Analysis and Forecast of Global Sales')
plt.xlabel('Year')
plt.ylabel('Global Sales (in millions)')
plt.legend()
plt.show()



#Scatter plot of global sales vs year vs genre
scatter_plot = videogames_df.groupby(['Year', 'Genre'])['Global_Sales'].sum().reset_index()

plt.figure(figsize=(14, 8))
sns.scatterplot(data=scatter_plot, x='Year', y='Global_Sales', hue='Genre', size='Global_Sales', sizes=(20, 200), alpha=0.7)
plt.title('Global Sales vs Year vs Genre')
plt.xlabel('Year')
plt.ylabel('Global Sales (in millions)')
plt.legend(title='Genre')
plt.show()

# Stacked bar graph of sales distribution across regions for each genre
stacked_bar = videogames_df.groupby('Genre').agg({
    'EU_Sales': 'sum',
    'JP_Sales': 'sum',
    'NA_Sales': 'sum',
    'Other_Sales': 'sum'
}).reset_index()

stacked_bar_long = pd.melt(stacked_bar, id_vars='Genre', var_name='Region', value_name='Sales')

plt.figure(figsize=(14, 8))
sns.barplot(data=stacked_bar_long, x='Genre', y='Sales', hue='Region')
plt.title('Sales Distribution Across Regions for Each Genre')
plt.xlabel('Genre')
plt.ylabel('Sales (in millions)')
plt.xticks(rotation=45)
plt.legend(title='Region')
plt.show()

#Pie chart of platform market sales
platform_pie = videogames_df.groupby('Platform')['Global_Sales'].sum().reset_index()

plt.figure(figsize=(8, 8))
plt.pie(platform_pie['Global_Sales'], labels=platform_pie['Platform'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(platform_pie)))
plt.title('Platform Market Sales')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

#Sales Prediction Model
features_sales = videogames_encoded.drop(['Rank', 'Name', 'Global_Sales', 'Genre'], axis=1)
target_sales = videogames_encoded['Global_Sales']

X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(features_sales, target_sales, test_size=0.2, random_state=42)

sales_model = RandomForestRegressor(random_state=42)
sales_model.fit(X_train_sales, y_train_sales)

sales_predictions = sales_model.predict(X_test_sales)

sales_mse = mean_squared_error(y_test_sales, sales_predictions)
sales_rmse = np.sqrt(sales_mse)
sales_r2 = r2_score(y_test_sales, sales_predictions)

print(f"Sales Prediction Model - RMSE: {sales_rmse}")
print(f"Sales Prediction Model - R²: {sales_r2}")

# Genre Classification
features_genre = videogames_encoded.drop(['Rank', 'Name', 'Global_Sales', 'Genre'], axis=1)
target_genre = videogames_df['Genre']

target_genre_encoded = pd.get_dummies(target_genre).idxmax(1)

X_train_genre, X_test_genre, y_train_genre, y_test_genre = train_test_split(features_genre, target_genre_encoded, test_size=0.2, random_state=42)

genre_model = GradientBoostingClassifier(random_state=42)
genre_model.fit(X_train_genre, y_train_genre)

genre_predictions = genre_model.predict(X_test_genre)

genre_accuracy = accuracy_score(y_test_genre, genre_predictions)
genre_classification_report = classification_report(y_test_genre, genre_predictions, target_names=target_genre.unique())

print(f"\nGenre Classification Model - Accuracy: {genre_accuracy}")
print(f"Genre Classification Model - Classification Report:\n{genre_classification_report}")

#Market Segmentation

features_clustering = videogames_encoded[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_clustering)

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
videogames_df['Cluster'] = clusters

cluster_summary = videogames_df.groupby('Cluster').agg({
    'NA_Sales': 'mean',
    'EU_Sales': 'mean',
    'JP_Sales': 'mean',
    'Other_Sales': 'mean',
    'Global_Sales': 'mean'
}).reset_index()

cluster_labels = {
    0: 'High NA Sales',
    1: 'High JP Sales',
    2: 'Balanced Global Sales',
    3: 'Low Sales',
    4: 'High EU Sales'
}

videogames_df['Cluster_Label'] = videogames_df['Cluster'].map(cluster_labels)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='NA_Sales', y='EU_Sales', hue='Cluster_Label', data=videogames_df, palette='viridis')
plt.title('Market Segmentation using K-Means Clustering with Labels')
plt.xlabel('NA Sales')
plt.ylabel('EU Sales')
plt.legend(title='Cluster Label', loc='upper right', bbox_to_anchor=(1, 1))
plt.show()