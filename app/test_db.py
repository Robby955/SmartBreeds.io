import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
filename = 'C:/Users/robby/OneDrive/Desktop/Breed_App/breed_data_20240328-204518.csv'
breed_data_df = pd.read_csv(filename)

# Sort the data based on precision and recall
breed_data_df_sorted_precision = breed_data_df.sort_values('precision', ascending=False)
breed_data_df_sorted_recall = breed_data_df.sort_values('recall', ascending=False)

# Select top 10 and bottom 10 breeds based on precision and recall
top_bottom_precision = pd.concat([breed_data_df_sorted_precision.head(10), breed_data_df_sorted_precision.tail(10)])
top_bottom_recall = pd.concat([breed_data_df_sorted_recall.head(10), breed_data_df_sorted_recall.tail(10)])

# Set the style of the visualization
sns.set(style="whitegrid")

# Initialize a grid of plots with an Axes for each metric
fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Top and bottom breeds based on precision
sns.barplot(x="precision", y="Breed", data=top_bottom_precision, palette="coolwarm", ax=axes[0])
axes[0].set_title('Top and Bottom 10 Breeds by Precision')

# Top and bottom breeds based on recall
sns.barplot(x="recall", y="Breed", data=top_bottom_recall, palette="coolwarm", ax=axes[1])
axes[1].set_title('Top and Bottom 10 Breeds by Recall')

# Final adjustments
plt.tight_layout()
plt.show()
# Set the style of the visualization
sns.set(style="whitegrid")

# Create a combined barplot for precision and recall
fig, ax = plt.subplots(figsize=(12, 10))
top_bottom_combined = pd.concat([top_bottom_precision, top_bottom_recall]).drop_duplicates().reset_index(drop=True)
melted_data = pd.melt(top_bottom_combined, id_vars=['Breed'], value_vars=['precision', 'recall'], var_name='Metric', value_name='Score')
sns.barplot(x='Score', y='Breed', hue='Metric', data=melted_data, palette='viridis')
ax.set_title('Top and Bottom 10 Breeds by Precision and Recall')
plt.tight_layout()

# Create a violin plot for precision and recall distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.violinplot(x="variable", y="value", data=pd.melt(breed_data_df[['precision', 'recall']]), split=True, inner="quart", ax=axes[0])
axes[0].set_title('Distribution of Precision and Recall Scores')

# Create a scatter plot for precision vs recall
sns.scatterplot(x='precision', y='recall', data=breed_data_df, ax=axes[1])
axes[1].set_xlabel('Precision')
axes[1].set_ylabel('Recall')
axes[1].set_title('Precision vs Recall for Each Breed')
plt.tight_layout()

plt.show()


