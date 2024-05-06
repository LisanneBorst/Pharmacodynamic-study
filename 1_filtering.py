"""
Script for filtering raw EEG data

Loops through raw files in folder (.edf extension)
Converts MNE to dictionary and selects channels
Filters files using filtering from filtering_functions
Files are converted to MNE raw objects
Preprocessed files are saved
"""

# Packages
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne

from filtering_functions import filtering

# Specify
folder_path = 'data' # Folder path containing the .edf files
preproc_path = 'preproc' # Path to save filtered (preprocessed) files
channels_interest = ['EEG 3', 'EEG 4', 'EEG 7', 'EEG 10', 'EEG 11', 'EEG 12'] # Define channels that have signal (-0 for python indexing)
channel_names = ['OFC_right', 'S_right', 'EMG_right', 'EMG_left', 'S_left', 'OFC_left' ] # Define corresponding channel names

 
""" 
Function to convert dictionary into an MNE object
"""
# Convert a dictionary of EEG data to an MNE Raw object.
def create_mne_object_from_dict(data_dict, sfreq):
   
    # Extract channel names and data
    ch_names = list(data_dict.keys())
    data = np.array(list(data_dict.values()))

    # Ensure data is in the correct shape (n_channels, n_times)
    if data.shape[0] != len(ch_names):
        data = data.T  # Transpose if necessary

    # Create an Info object
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    # Create the RawArray object
    raw = mne.io.RawArray(data, info)

    return raw
    # # Save the Raw object to disk (e.g., "eeg_data.fif")
    # raw.save("eeg_data.fif", overwrite=True)

    # print("Data saved to eeg_data.fif")


"""
Filtering data
"""
# Loop through the files in the folder
for filename in os.listdir(folder_path):
    # Check for .edf extension, read raw file, get sample frequency
    if filename.endswith('.edf'):
        file_path = os.path.join(folder_path, filename)
        data = mne.io.read_raw_edf(file_path, include=channels_interest)
        sfreq = data.info['sfreq']

        # Creating dictionary, selecting necessary channels
        all_filtered = {}
        for channel in channels_interest:
            print(channel)
            channel_data = data.get_data(picks=channel)[0]

            # Filter data
            filtered_channel_data = filtering(channel_data, data.info['sfreq'])

            # Save to dictonary
            all_filtered[channel] = filtered_channel_data
        
        filtered_mne = create_mne_object_from_dict(all_filtered, sfreq=sfreq)

        # Rename and save file to preproc_path
        temp_filename = os.path.splitext(filename)[0] #split extension
        temp_filename = os.path.join(preproc_path, f'preprocessed_{temp_filename}_raw.fif')
        filtered_mne.save(temp_filename, overwrite=True)

        print(f'{temp_filename} saved to {preproc_path}')

    else:
        print(f"non-processed file: {filename}")


