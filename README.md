Pharmacodynamic study of CNO

This folder contains scripts to filter, cluster and analyse EEG data after administration of different CNO doses, and a notebook to visualise the results in PSD plots.

Short overview:
1_filtering.py -- Filters raw EEG 
2_ploss_and_epoching.py -- Takes the preprocessed EEG files, marks EEG package loss and creates epochs of the data.
3_clustering_and_dbscan.py -- Processes EEG epoch data by extracting features and applying DBSCAN to mark the outliers.
4_kmeans.py -- Processes EEG epoch data through K-means clustering and generates various plots (PCA plots, correlation plots, PSD plots) to visualise results
5_PSD_relative.ipynb -- Notebook to visualise the data using PSD plots.

Change before use: 
- "folder_path" : folder containing raw EEG files (.edf)
- "preproc_path" : folder where the preprocessed EEG files will be saved.
- "channels_interest" : the EEG channels to be included in the analysis. NOTE: use -1 for python indexing.

Good luck!









