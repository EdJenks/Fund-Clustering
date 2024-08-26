import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load_and_clean_data():
    # Load the dataset
    data = pd.read_csv('./data.csv')

    # Data Cleaning
    missing_threshold = 0.8  # Drop columns with more than 20% NaNs
    data_cleaned = data.dropna(thresh=(len(data) * missing_threshold), axis=1)

    # Filter out funds with incomplete time periods as this is a small number
    data_cleaned = data_cleaned.groupby('Security ID').filter(lambda x: x['Date'].nunique() == 3)

    # Convert the Date column to datetime format to ensure correct sorting (might not be needed but saw errors)
    data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], format='%m/%d/%Y')

    # Sort by Security ID and Date to ensure correct order for completeness
    data_cleaned = data_cleaned.sort_values(by=['Security ID', 'Date'])
    return data_cleaned


def feature_engineering_and_preprocessing(data_cleaned):

    # Feature Engineering: Calculate percentage change to help clustering
    data_cleaned['NAV_pct_change'] = data_cleaned.groupby('Security ID')['NAV (Daily - USD)'].pct_change()
    data_cleaned['NAV_pct_change'].fillna(1e-8, inplace=True)
    data_cleaned['Fund_Size_pct_change'] = data_cleaned.groupby('Security ID')['Fund Size (USD)'].pct_change()
    data_cleaned['Fund_Size_pct_change'].fillna(1e-8, inplace=True)

    # Impute missing values for numeric data on a per-fund basis
    numeric_cols = data_cleaned.select_dtypes(include=[np.number]).columns
    data_cleaned[numeric_cols] = data_cleaned.groupby('Security ID')[numeric_cols].transform(
        lambda x: x.fillna(x.median() if not x.median() else 1e-8)
    )

    # Impute missing values for categorical data on a per-fund basis (using the most frequent strategy)
    categorical_cols = data_cleaned.select_dtypes(exclude=[np.number, 'datetime']).columns
    data_cleaned[categorical_cols] = data_cleaned.groupby('Security ID')[categorical_cols].transform(
        lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'empty')
    )

    # Flatten the data to create a single row per fund with separate columns for each time point
    pivoted_data = data_cleaned.pivot(index='Security ID', columns='Date', values=numeric_cols)

    # Rename columns to reflect the time point (e.g., NAV_T1)
    pivoted_data.columns = [f'{col[0]}_T{col[1].month}' for col in pivoted_data.columns]

    # Combine numeric and categorical data into a single DataFrame
    categorical_data = data_cleaned.groupby('Security ID').first()[categorical_cols.drop('Security ID')]
    data_final = pd.concat([pivoted_data, categorical_data], axis=1)

    # Separate numeric and categorical columns again after pivoting
    numeric_features = data_final.select_dtypes(include=[np.number])
    categorical_features = data_final.select_dtypes(exclude=[np.number])

    # Scale numeric data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numeric_features)

    # Encode categorical data
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    data_encoded = encoder.fit_transform(categorical_features)

    # Combine scaled numeric data and encoded categorical data
    data_preprocessed = np.hstack((data_scaled, data_encoded))

    # Dimensionality Reduction 
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data_preprocessed)
    return data_reduced, data_final


def fit_data_to_clusters_and_plot(data_reduced):

    # Clustering using Agglomerative Clustering
    agg_cluster = AgglomerativeClustering(n_clusters=6, metric='euclidean', linkage='ward')
    cluster_labels = agg_cluster.fit_predict(data_reduced)

    # Evaluate Clustering
    sil_score = silhouette_score(data_reduced, cluster_labels)
    print(f'Silhouette score : {sil_score}')

    # Visualization using TSNE
    tsne = TSNE(n_components=2)
    data_2d = tsne.fit_transform(data_reduced)
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=cluster_labels, cmap='viridis')
    plt.title('Clusters Visualized in 2D Space using t-SNE')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    return cluster_labels


def print_cluster_size_and_save(data_final, data_cleaned, cluster_labels):

    # Create a DataFrame to hold the Security IDs and their cluster labels
    cluster_df = pd.DataFrame({
        'Security ID': data_final.index,
        'Cluster': cluster_labels
    })

    # Group the Security IDs by their cluster label
    clustered_funds = cluster_df.groupby('Cluster')['Security ID'].apply(list)

    # Print number of funds in each cluster
    for cluster_num, funds in clustered_funds.items():
        print(f'Cluster {cluster_num}: {len(funds)} funds')

    # Export to CSV
    pd.merge(data_cleaned, cluster_df, left_on='Security ID', right_on='Security ID', how='left').to_csv('cluster_composition.csv', index=False)


def main():
    data_cleaned = load_and_clean_data()

    data_reduced, data_final = feature_engineering_and_preprocessing(data_cleaned=data_cleaned)

    cluster_labels = fit_data_to_clusters_and_plot(data_reduced=data_reduced)

    print_cluster_size_and_save(data_final=data_final, data_cleaned=data_cleaned, cluster_labels=cluster_labels)

if __name__ == "__main__":
    main()
