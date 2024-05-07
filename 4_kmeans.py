"""
Script to put EEG epochs through KMEANS (after DBSCAN)
"""

# start outer loop: 
    # load in epoch data from folder_epochs
    # get "animal_id" and "treatment" from metadata
    # start inner loop: 
        # if "animal_id" AND "treatment" are in filename from folder_features: select file
        # end inner loop
    # filter out the epoch rows where DBSCAN == -1 (=outliers)
    # save sliced epoch
    # select the features from features_df file, exclude non-essential features
    # scale the features
    # KMEANS on the features + add KMEANS_cluster column
    # put it back on metadata & save df features
    # PCA plot & save
    # correlation plots & save
    # plot individual PSD & save
    # end outer loop
# next epoch file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import mne
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Channels of interest
channels_interest = ['EEG 3', 'EEG 12'] # Channels that have signal (-0 for python indexing)
channel_names = ['OFC_right', 'OFC_left' ] # Corresponding channel names

# Specify folder paths
folder_epochs = 'folder_epochs/sliced_epochs'  # Folder for epoch files 
folder_features = 'features_dataframes/dataframes_DBSCAN' # Folder for features_dataframes

"""
Data 
"""
# Loop through the folder with epoch files
for filename in os.listdir(folder_epochs):
    if not filename.endswith("epo.fif"):
        continue

    # Read epoch data from file & pick channels
    data = mne.read_epochs(os.path.join(folder_epochs, filename), preload=True)
    data.pick(channels_interest)

    # From metadata, select animal and treatment
    animal_id = data.metadata['Animal'].unique()[0]
    treatment = data.metadata['Treatment'].unique()[0]
    print(f"Selected: {animal_id} {treatment}, file: {filename}")

    # Loop through features_folder to select corresponding feature dataframe file
    for excel_file  in os.listdir(folder_features):
        if not excel_file .endswith(".xlsx"):
            continue
        
        # Select file that corresponds with animal_id and treatment of epoch file 
        if animal_id in excel_file and treatment in excel_file:
            print(f"Selected excel: {excel_file}")
            
            # Read df of features of selected file
            features_df = pd.read_excel(os.path.join(folder_features, excel_file), index_col=0)
            
            # Break bc matching file is found
            break

    # Filter rows where the "DBSCAN_cluster" column is not equal to -1
    features_df = features_df[features_df['DBSCAN_cluster'] != -1]

    # Define & exlude unnecessary columns
    columns_to_drop = ['OFC_right_mean', 'OFC_right_std', 'OFC_right_skewness', 
        'OFC_right_kurtosis', 'OFC_right_variance', 'OFC_right_min', 
        'OFC_right_max', 'OFC_right_slope', 'OFC_left_mean', 'OFC_left_std', 
        'OFC_left_skewness', 'OFC_left_kurtosis', 'OFC_left_variance', 
        'OFC_left_min', 'OFC_left_max', 'OFC_left_slope']
    
    features_df_clean = features_df.drop(columns=columns_to_drop)

    """
    KMEANS clustering
    """
    # Subset for just the features
    features_OFC_df = features_df_clean.filter(regex='^OFC_')

    # Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_OFC_df)

    # Perform k-means clustering
    kmeans = KMeans(random_state=69, n_clusters=2, n_init='auto')
    kmeans.fit(scaled_features)

    # Add new column to the dataframe for the cluster label
    features_OFC_df.loc[:, 'KMEANS_cluster'] = kmeans.labels_
    
    # Check how many datapoints per cluster
    print(f" Cluster KMEANS {animal_id} {treatment}:")
    print(features_OFC_df['KMEANS_cluster'].value_counts())

    # Subset for cluster 0 and 1
    cluster_0_df = features_OFC_df[features_OFC_df['KMEANS_cluster'] == 0]
    cluster_1_df = features_OFC_df[features_OFC_df['KMEANS_cluster'] == 1]

    # Calculate the sum of OFC_left_psd_around_70hz and OFC_right_psd_around_70hz for each cluster
    avg_sum_cluster_0 = (cluster_0_df['OFC_left_psd_around_70hz'].mean() + cluster_0_df['OFC_right_psd_around_70hz'].mean())/2
    avg_sum_cluster_1 = (cluster_1_df['OFC_left_psd_around_70hz'].mean() + cluster_1_df['OFC_right_psd_around_70hz'].mean())/2
    print("Average Sum for Cluster 0:", avg_sum_cluster_0)
    print("Average Sum for Cluster 1:", avg_sum_cluster_1)

    # Compare average PSD values between clusters and assign correct cluster label
    if avg_sum_cluster_1 > avg_sum_cluster_0:
        cluster_labels = {0: "Low_70Hz", 1: "High_70Hz"}
    else:
        cluster_labels = {1: "Low_70Hz", 0: "High_70Hz"}

    # Map cluster labels to dataframe
    features_OFC_df['Cluster_Label'] = features_OFC_df['KMEANS_cluster'].map(cluster_labels)

    # Save final dataframe with selected columns incl. KMEANS cluster column
    os.makedirs("features_dataframes/dataframes_KMEANS", exist_ok=True)
    features_df_clean.to_excel(f"features_dataframes/dataframes_KMEANS/dataframe_OFC_features_KMEANS_{animal_id}_{treatment}.xlsx")

    # Add colums with KMEANS cluster and cluster labels to the epochs metadata
    data.metadata = pd.concat([data.metadata, features_OFC_df[['KMEANS_cluster', 'Cluster_Label']]], axis=1) 
    
    # Save the sliced epoch file with new metadata
    os.makedirs("folder_epochs/sliced_epochs_KMEANS", exist_ok=True)
    data.save(f"folder_epochs/sliced_epochs_KMEANS/sliced_{filename}")
    print(f"file {animal_id} {treatment} saved to folder_epochs/sliced_epochs_KMEANS")
    
    """
    Principal Component Analysis of KMEANS
    """
    # PCA for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df.loc[:, 'Cluster_Label'] = features_OFC_df['Cluster_Label'].values

    # Plot PCA scatter plot
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(features_OFC_df['Cluster_Label'])
    colors = ['#256c74', '#9947eb', '#ff5733', '#33ff57', '#5733ff'] 
    for label, color in zip(unique_labels, colors):
        cluster_data = pca_df[pca_df['Cluster_Label'] == label]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'{label}', alpha=0.7, color=color)
    plt.title(f'PCA Scatter Plot of KMEANS Clustered Data {animal_id} {treatment}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    factor = 1.5  # Scale factor
    for text_obj in plt.findobj(match=plt.Text):
        text_obj.set_fontsize(text_obj.get_fontsize() * factor)
    plt.tight_layout()
    os.makedirs("plots/PCA_KMEANS_OFC", exist_ok=True)
    plt.savefig(f'plots/PCA_KMEANS_OFC/PCA_KMEANS_OFC_{animal_id}_{treatment}.png')
    plt.close()

    # PCA1 correlation plot
    # Get the absolute correlation between features and PC1, store in df
    pc1_correlation = np.abs(pca.components_[0])
    pc1_correlation_df = pd.DataFrame({'Feature': features_OFC_df.iloc[:, :-2].columns, 'PC1 Correlation': pc1_correlation})

    # Sort df by absolute correlation values
    pc1_correlation_df = pc1_correlation_df.sort_values(by='PC1 Correlation', ascending=False)

    # Plot the correlation bar plot
    plt.figure(figsize=(10, 8))
    plt.bar(pc1_correlation_df['Feature'], pc1_correlation_df['PC1 Correlation'], color='#99bbff')
    plt.xlabel('Feature')
    plt.ylabel('Absolute Correlation with PC1')
    plt.title(f'Absolute Correlation of Features with PC1 {animal_id} {treatment}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs("plots/PCA_KMEANS_OFC", exist_ok=True)
    plt.savefig(f'plots/PCA_KMEANS_OFC/Corr_PCA1_OFC_{animal_id}_{treatment}.png')
    plt.close()

    # PCA2 correlation plot
    # Get the absolute correlation between features and PC2, store in df
    pc2_correlation = np.abs(pca.components_[1])
    pc2_correlation_df = pd.DataFrame({'Feature': features_OFC_df.iloc[:, :-2].columns, 'PC2 Correlation': pc2_correlation})

    # Sort df by absolute correlation values
    pc2_correlation_df = pc2_correlation_df.sort_values(by='PC2 Correlation', ascending=False)

    # Plot the correlation bar plot
    plt.figure(figsize=(10, 8))
    plt.bar(pc2_correlation_df['Feature'], pc2_correlation_df['PC2 Correlation'], color='#99bbff')
    plt.xlabel('Feature')
    plt.ylabel('Absolute Correlation with PC2')
    plt.title(f'Absolute Correlation of Features with PC2 {animal_id} {treatment}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    os.makedirs("plots/PCA_KMEANS_OFC", exist_ok=True)
    plt.savefig(f'plots/PCA_KMEANS_OFC/Corr_PCA2_OFC_{animal_id}_{treatment}.png')
    plt.close()

    """
    Correlation  plot with KMEANS
    """
    correlation_series = features_OFC_df.iloc[:, :-1].corr()['KMEANS_cluster'].sort_values()
    plt.figure(figsize=(8, 6))

    # Colours for negative & positive correlation
    corr_colors = ['#ff6666' if corr < 0 else '#99bbff' for corr in correlation_series]
    
    # Plot correlation
    correlation_series.plot(kind='bar', color=corr_colors)
    plt.xlabel('Feature')
    plt.ylabel('Correlation')
    plt.title(f'Correlation of Features with KMEANS Cluster {animal_id} {treatment}')
    plt.tight_layout()
    os.makedirs("plots/PCA_KMEANS_OFC", exist_ok=True)
    plt.savefig(f'plots/PCA_KMEANS_OFC/PCA_KMEANS_corr_OFC_{animal_id}_{treatment}.png')
    plt.close()

    """
    PSD plot for all epochs (clustered)
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    ax = ax.ravel()

    # Iterate over each cluster label
    for cluster_label, color in zip(['High_70Hz', 'Low_70Hz'], (['#256c74', '#b366ff'])):  
        # Filter data for epochs belonging to the current cluster
        cluster_data = data[data.metadata.index.isin(features_OFC_df[features_OFC_df['Cluster_Label'] == cluster_label].index)]

        # Iterate over channels of interest
        for i, (channel, ch_name) in enumerate(zip(channels_interest, channel_names)):
            # Extract data corresponding to the channel from the DataFrame
            channel_data = cluster_data.get_data(picks=[channel])

            # Calculate the PSD for the channel data  
            psds, freqs = mne.time_frequency.psd_array_multitaper(channel_data, fmin=0, fmax=100, sfreq=data.info['sfreq'])

            # Calculate total power, relative PSD, and mean relative PSD
            total_power = np.sum(psds, axis=-1)        
            rel_psd = psds / total_power[:, np.newaxis]
            mean_rel_psd = np.median(rel_psd[:, 0, :], axis=0)
            
            # Individual PSD plotting
            for ep in range(psds.shape[0]):
                ax[i].plot(freqs, rel_psd[ep, 0, :], color=color, alpha=0.5, linewidth=0.01)
            ax[i].plot(freqs, mean_rel_psd, color=color, label=f'{cluster_label}')
            ax[i].set_yscale('log')
            ax[i].set_xlabel('Frequency (Hz)')
            ax[i].set_ylabel('Mean PSD')
            ax[i].set_title(f'Average relative PSD per cluster {animal_id} {treatment} {ch_name}')
            ax[i].grid(True)
            ax[i].legend(title='Cluster')
    os.makedirs("plots/individual_psds/KMEANS_individual_PSD", exist_ok=True)
    plt.savefig(f'plots/individual_psds/KMEANS_individual_PSD/KMEANS_individual_PSD_{animal_id}_{treatment}.png')
    plt.close()

    