"""
Script to exclude EEG package loss and create epochs of the data 

Input: EEG fif files, after filtering

"""

# packages
import os
import pandas as pd
import numpy as np
import mne


# Specify
raw_folder_path = 'data' # Folder path containing the raw files (.edf)
preproc_path = 'preproc' # Folder path containing the preprocessed files (.raw.fif)
folder_epochs = 'folder_epochs' # Output path to save the epoch files 
channels_interest = ['EEG 3', 'EEG 4', 'EEG 7', 'EEG 10', 'EEG 11', 'EEG 12'] # Define channels that have signal (-0 for python indexing)
channel_names = ['OFC_right', 'S_right', 'EMG_right', 'EMG_left', 'S_left', 'OFC_left' ] # Define corresponding channel names
low_val, high_val = 0.006, 0.013 # Rejection values for package loss
dur = 5  # Duration of epochs in seconds
# Define the threshold for rejecting epochs based on the number of NaN values
ploss_threshold = 500  # miliseconds

"""
Exclude the package loss
"""

# Loop through the folder with raw files
for raw_filename in os.listdir(raw_folder_path):
    
    # Check if the file has an .edf extension
    if not raw_filename.endswith('.edf'):
        continue

    # Select filepath and load data
    raw_file_path = os.path.join(raw_folder_path, raw_filename)
    raw_data = mne.io.read_raw_edf(raw_file_path, include=channels_interest)

    # Loop through folder with preprocessed files
    for preproc_filename in os.listdir(preproc_path):
        if not preproc_filename.endswith("raw.fif"):
            continue
        temp_preproc_filename = os.path.splitext(os.path.basename(preproc_filename))[0]
        temp_raw_filename = os.path.splitext(os.path.basename(raw_filename))[0]

        # Check if the file mathces, break if found
        if temp_raw_filename in temp_preproc_filename:
            data = mne.io.read_raw_fif(os.path.join(preproc_path, preproc_filename))
            print(f'preproc data selected: {preproc_filename}')
            break

    # Get sample frequency from info of the raw data
    sfreq = raw_data.info['sfreq']

    # Find package loss and put it in a dictionary
    ploss_data = {}
    for channel in raw_data.info['ch_names']:
        print(channel)
        s = raw_data.get_data(picks=[channel])[0]
        rej = np.where(s > low_val, s , np.nan)
        rej = np.where(s < high_val, rej , np.nan)
        ploss_data[channel] = rej

    # Make MNE Raw Object for package loss data
    ploss_raw = mne.io.RawArray(np.array(list(ploss_data.values())), raw_data.info)
    print(f'Package loss calculated for {raw_filename}')

    """
    Making epochs from preprocessed files
    """
    # Make epochs
    epochs = mne.make_fixed_length_epochs(data, duration=dur, preload=True)
    # Make epochs from the package loss data
    ploss_epochs = mne.make_fixed_length_epochs(ploss_raw, duration=dur)

    """ 
    Select the bad epochs
    """
    # Iterate through epochs and apply artifact rejection criteria
    bad_epochs = []
    for idx, epoch in enumerate(ploss_epochs):
        # Check p-loss for every channel. If there's ploss in at least one channel, the epoch is bad
        for ch_idx in range(epoch.shape[0]):
            # Count the number of NaN values in the epoch
            nan_count = np.sum(np.isnan(epoch[ch_idx]))
            # If the number of NaN values exceeds the threshold, mark the epoch as bad
            if nan_count > int(sfreq * ploss_threshold / 1000):
                bad_epochs.append(idx)
                break # Only one channel is enough to classify an epoch as bad

    bad_epochs = np.array(bad_epochs)
    
    # Check how many bad epochs & which ones are bad
    bad_epochs.shape 
    print(bad_epochs)

    """
    Create metadata file
    """
    # Split filename, select elements
    filename_parts = preproc_filename.split('_')

    # Define metadata as elements of the filename
    transmitter = filename_parts[2]
    batch = filename_parts[3]
    animal_id = filename_parts[4]
    earclip_id = filename_parts[5]
    surgery = filename_parts[6]
    camera_side = filename_parts[7]
    treatment = filename_parts[8]
    date = filename_parts[9]
    time = filename_parts[10]

    # Define the number of epochs
    num_epochs = len(epochs._data)

    metadata= {
        'Transmitter': [transmitter]*num_epochs, #Times the num_epochs to assign it to all the individual epochs
        'Batch' : [batch]*num_epochs,
        'Animal': [animal_id]*num_epochs, 
        'Earclip' : [earclip_id]*num_epochs,
        'Surgery' : [surgery]*num_epochs,
        'Camera_side' :[camera_side]*num_epochs,
        'Treatment' : [treatment]*num_epochs,
        'Date' : [date]*num_epochs,
        'Time': [time]*num_epochs
    }

    # Create df of metadata, put a column with 1 if the epoch is bad
    mt = pd.DataFrame(metadata)
    mt['bad_epoch'] = 0  # Create a column with zeros
    for idx in epochs.selection:
        if idx in bad_epochs:
            mt.loc[idx, 'bad_epoch'] = 1 # bad epoch = 1

    # Put the metadata on the epochs
    epochs.metadata = mt
    
    # Save the epochs & metadata
    epochs.save(os.path.join(folder_epochs, f'epochs_{temp_preproc_filename}-epo.fif'), overwrite=True)
    mt.to_excel(f"folder_epochs/metadata_{temp_preproc_filename}.xlsx", index=False)
    print(f"Epochs & metadata of {temp_preproc_filename} is saved")

