"""
Script to put EEG epochs through DBSCAN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import mne
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

# Channels of interest
channels_interest = ['EEG 3', 'EEG 12'] # Channels that have signal (-0 for python indexing)
channel_names = ['OFC_right', 'OFC_left' ] # Corresponding channel names

# Specify folder path for the epochs
folder_epochs = 'folder_epochs'  

# Select start of timepoint and duration of data selection
start_timepoint = 900 # Index of epoch. 
duration_hours = 2 # Select the number of hours to analyse   

# Define animals
s1_animals  = ["8.9", "8.12", "8.14", "8.15"]
s2_animals = ["8.10", "8.11", "8.13", "8.16"]

# Define exluded animals (animal, treatment)
exclusions = [
    ("8.15", "CNO5"),
    ("8.9", "CNO10"),
    ("8.11", "CNO1"),
    ("8.13", "CNO1"),
    ("8.10", "CNO5"),
    ("8.13", "CNO5"),
    ("8.13", "CNO10")
]

# DBSCAN hyperparameters
epsilon = 3
min_samples = 4

# Custom colors for DBSCAN
custom_colors = ['#ff6633', '#148191', '#149014', '#00b3b3', '#40bf80'] #first is for outliers

""" 
Define basic functions to calculate features
"""
def calculate_slope(signal):
    return linregress(range(len(signal)), signal).slope

def calculate_zero_crossing_rate(signal):
    zero_crossings = ((signal[:-1] * signal[1:]) < 0).sum()
    return zero_crossings / len(signal)

def calculate_psd_features(signal, sfreq, fmin, fmax):
    psd, _ = mne.time_frequency.psd_array_multitaper(signal, fmin=fmin, fmax=fmax, sfreq=sfreq)
    return np.mean(psd)

# Define function to convert power to dB
def nanpow2db(y):
    """ Power to dB conversion, setting bad values to nans
        Arguments:
            y (float or array-like): power
        Returns:
            ydB (float or np array): inputs converted to dB with 0s and negatives resulting in nans
    """
    if isinstance(y, int) or isinstance(y, float):
        if y == 0:
            return np.nan
        else:
            ydB = 10 * np.log10(y)
    else:
        if isinstance(y, list):  # if list, turn into array
            y = np.asarray(y)
        y = y.astype(float)  # make sure it's a float array so we can put nans in it
        y[y == 0] = np.nan
        ydB = 10 * np.log10(y)
    return ydB

# Defining function to extract features
def extract_features(epoch, channel_names):
    features = {}
    for i, (channel_data, channel_name) in enumerate(zip(epoch, channel_names)):
        features.update({
            channel_name + '_peak_to_peak': channel_data.ptp(),
            channel_name + '_mean': channel_data.mean(),
            channel_name + '_std': channel_data.std(),
            channel_name + '_skewness': pd.Series(channel_data).skew(),
            channel_name + '_kurtosis': pd.Series(channel_data).kurtosis(),
            channel_name + '_variance': channel_data.var(),
            channel_name + '_min': channel_data.min(),
            channel_name + '_max': channel_data.max(),
            channel_name + '_slope': calculate_slope(channel_data),
            channel_name + '_zero_crossing_rate': calculate_zero_crossing_rate(channel_data),
            channel_name + '_psd_around_70hz': calculate_psd_features(channel_data, sfreq=data.info['sfreq'], fmin=65, fmax=70),
            channel_name + '_delta': calculate_psd_features(channel_data, sfreq=data.info['sfreq'], fmin=0.5, fmax=4),
            channel_name + '_theta': calculate_psd_features(channel_data, sfreq=data.info['sfreq'], fmin=4, fmax=8),
            channel_name + '_alpha': calculate_psd_features(channel_data, sfreq=data.info['sfreq'], fmin=8, fmax=12),
            channel_name + '_beta':  calculate_psd_features(channel_data, sfreq=data.info['sfreq'], fmin=12, fmax=30),
            channel_name + '_gamma': calculate_psd_features(channel_data, sfreq=data.info['sfreq'], fmin=30, fmax=100),
        })
    return features

"""
Data
"""
# Loop through folder with epochs
for filename in os.listdir(folder_epochs):
    if not filename.endswith("epo.fif"):
        continue

    # From filename, select animal and treatment
    animal_id = str(filename.split('_')[5])
    treatment = str(filename.split('_')[9])

    # Skip files with excluded animals
    if (animal_id, treatment) in exclusions:
        continue

    # Skip unwanted treatments
    if animal_id in s1_animals and treatment in ['Saline1', 'washout', 'acc']:
        continue
    if animal_id in s2_animals and treatment in ['Saline1', 'Saline2', 'washout', 'Acclimate']:
        continue
    
    # Read epoch data from file & pick channels
    data = mne.read_epochs(os.path.join(folder_epochs, filename), preload=True)
    data.pick(channels_interest)
    print(f"Selected epoch file: {filename} {animal_id} {treatment}")

    # Select data from selected start and endpoint
    duration_epochs = duration_hours * 60 * 12  # 60 minutes/hour * 12 epochs/minute
    end_timepoint = start_timepoint + duration_epochs
    data = data[start_timepoint:end_timepoint]

    # Exclude bad epochs
    data = data[data.metadata['bad_epoch']==0] # Good epoch == 0
   
    """
    Feature extraction and dB conversion
    """
    # Extract features for all epochs
    features = [extract_features(epoch, channel_names) for epoch in data.get_data()]
    features_df = pd.DataFrame(features)

    # Define PSD columns to convert
    psd_columns = ['OFC_right_psd_around_70hz', 'OFC_right_delta', 'OFC_right_theta', 'OFC_right_alpha',
                'OFC_right_beta', 'OFC_right_gamma', 
                'OFC_left_psd_around_70hz', 'OFC_left_delta',
                'OFC_left_theta', 'OFC_left_alpha', 'OFC_left_beta', 'OFC_left_gamma']

    # Apply nanpow2db function to columns with PSD values
    features_df[psd_columns] = features_df[psd_columns].apply(nanpow2db)

    """ 
    DBSCAN of OFC subset
    """    
    # Subset for only the features of OFC
    features_OFC_df = features_df.filter(regex='^OFC_')

    # Normalize the features in the subset of only the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_OFC_df)
    print(f'DBSCAN on selected feature columns: {features_OFC_df.columns}')

    # Apply DBSCAN to the scaled features
    db = DBSCAN(eps=epsilon, min_samples=min_samples).fit(scaled_features)
    labels = db.labels_

    # Add column with labels of DBSCAN
    features_OFC_df['DBSCAN_cluster'] = labels

    # Print no. of clusters & noise
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    """
    Principal Component Analysis of DBSCAN
    """
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['DBSCAN_cluster'] = features_OFC_df['DBSCAN_cluster']

    # Plot PCA scatter plot
    plt.figure(figsize=(8, 6))
    for i, cluster in enumerate(np.unique(features_OFC_df['DBSCAN_cluster'])):
        cluster_data = pca_df[pca_df['DBSCAN_cluster'] == cluster]
        plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'DBSCAN Cluster {cluster}', alpha=0.7, color=custom_colors[i])
    plt.title(f'PCA Scatter Plot of DBSCAN Clustered Data {animal_id} {treatment}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    factor = 1.5  # Scale factor
    for text_obj in plt.findobj(match=plt.Text):
        text_obj.set_fontsize(text_obj.get_fontsize() * factor)
    plt.tight_layout()
    os.makedirs("plots/PCA_DBSCAN_OFC", exist_ok=True)
    plt.savefig(f'plots/PCA_DBSCAN_OFC/PCA_DBSCAN_OFC_eps={epsilon}_minsamp={min_samples}_{animal_id}_{treatment}.png')
    plt.close()

    """
    Save sliced epoch with updated metadata, save df of features
    """
    # Put the OFC feature columns back on the metadata of the epochs
    full_df_OFC = pd.concat([data.metadata.copy(), features_OFC_df.set_index(data.metadata.index)],axis=1) 
    data.metadata = full_df_OFC
    
    # Save the sliced epoch file
    os.makedirs("folder_epochs/sliced_epochs", exist_ok=True)
    data.save(f"folder_epochs/sliced_epochs/sliced_{filename}")
    print(f"Sliced epoch of {animal_id} {treatment} saved")
    
    # Save final dataframe incl. DBSCAN cluster column
    os.makedirs("features_dataframes/dataframes_DBSCAN", exist_ok=True)
    full_df_OFC.to_excel(f"features_dataframes/dataframes_DBSCAN/dataframe_OFC_features_DBSCAN_{animal_id}_{treatment}.xlsx")
    print(f"Df of features of {animal_id} {treatment} saved")
