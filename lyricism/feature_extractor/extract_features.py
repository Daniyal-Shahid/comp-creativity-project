import os
import librosa
import numpy as np
import pandas as pd
import csv
from pathlib import Path

def extract_features(filename, genre_label=None):
    """
    Extract audio features to match the format in features_30_sec.csv
    
    Parameters:
    -----------
    filename : str
        Path to the audio file - can be absolute or relative path
    genre_label : str, optional
        Genre label for classification
        
    Returns:
    --------
    numpy.ndarray
        Feature vector for prediction (without filename and label)
    """
    # Handle both absolute and relative paths
    if os.path.isabs(filename):
        filepath = filename
    else:
        # Try different base directories to handle imports from different locations
        possible_paths = [
            filename,  # Direct path
            os.path.join(os.path.dirname(__file__), filename),  # Relative to this script
            os.path.join(os.getcwd(), filename)  # Relative to current working directory
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        else:
            raise FileNotFoundError(f"Audio file not found: {filename}")
    
    print(f"Processing audio file: {filepath}")
    
    # Load the audio file
    y, sr = librosa.load(filepath, mono=True, duration=30)
    
    # Get audio length
    length = len(y)
    
    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = float(np.mean(chroma_stft))
    chroma_stft_var = float(np.var(chroma_stft))
    
    # RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))
    rms_var = float(np.var(rms))
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = float(np.mean(spectral_centroid))
    spectral_centroid_var = float(np.var(spectral_centroid))
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = float(np.mean(spectral_bandwidth))
    spectral_bandwidth_var = float(np.var(spectral_bandwidth))
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = float(np.mean(rolloff))
    rolloff_var = float(np.var(rolloff))
    
    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = float(np.mean(zero_crossing_rate))
    zero_crossing_rate_var = float(np.var(zero_crossing_rate))
    
    # Harmony and Perceptr
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmony_mean = float(np.mean(y_harmonic))
    harmony_var = float(np.var(y_harmonic))
    perceptr_mean = float(np.mean(y_percussive))
    perceptr_var = float(np.var(y_percussive))
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)  # Ensure tempo is a scalar float
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = []
    mfcc_vars = []
    
    for i in range(20):
        mfcc_means.append(float(np.mean(mfccs[i])))
        mfcc_vars.append(float(np.var(mfccs[i])))
    
    # Create feature vector for model prediction (numeric values only)
    feature_vector = np.array([
        float(length), 
        chroma_stft_mean, chroma_stft_var,
        rms_mean, rms_var,
        spectral_centroid_mean, spectral_centroid_var,
        spectral_bandwidth_mean, spectral_bandwidth_var,
        rolloff_mean, rolloff_var,
        zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var,
        perceptr_mean, perceptr_var,
        tempo
    ] + mfcc_means + mfcc_vars, dtype=float)
    
    # For CSV saving, create a full features list with filename
    features_with_filename = [os.path.basename(filename)] + feature_vector.tolist()
    
    # For CSV saving with label
    if genre_label:
        features_csv = features_with_filename + [genre_label]
    else:
        features_csv = features_with_filename
    
    # Store the CSV format as an attribute
    feature_vector.csv_format = features_csv
    
    print(f"Extracted {len(feature_vector)} features")
    return feature_vector

def save_features_to_csv(features_list, output_file=None):
    """
    Save features to CSV in the same format as features_30_sec.csv
    
    Parameters:
    -----------
    features_list : list
        List of feature arrays with .csv_format attribute or list of feature lists
    output_file : str, optional
        Path to output CSV file
    """
    if output_file is None:
        output_dir = Path(os.path.dirname(__file__)) / "feature-extraction-output"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "features_output.csv"
    
    # Convert to CSV format if needed
    csv_rows = []
    for features in features_list:
        if hasattr(features, 'csv_format'):
            csv_rows.append(features.csv_format)
        else:
            csv_rows.append(features)
    
    # Define the column headers to match the original CSV
    headers = [
        'filename', 'length', 'chroma_stft_mean', 'chroma_stft_var',
        'rms_mean', 'rms_var', 'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean', 'rolloff_var',
        'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo'
    ]
    
    # Add MFCC headers
    for i in range(1, 21):
        headers.append(f'mfcc{i}_mean')
        headers.append(f'mfcc{i}_var')
    
    # Add label header if the features include a label
    if len(csv_rows[0]) > 59:  # 59 is the number of features without label
        headers.append('label')
    
    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_rows)
    
    print(f"Features saved to {output_file}")
    return str(output_file)

def extract_and_save_features(audio_files, labels=None, output_file=None):
    """
    Process multiple audio files and save their features to a CSV file
    
    Parameters:
    -----------
    audio_files : list
        List of paths to audio files
    labels : list, optional
        List of genre labels corresponding to audio files
    output_file : str, optional
        Path to output CSV file
    """
    features_list = []
    
    for i, audio_file in enumerate(audio_files):
        label = labels[i] if labels and i < len(labels) else None
        features = extract_features(audio_file, label)
        features_list.append(features)
    
    return save_features_to_csv(features_list, output_file)

if __name__ == "__main__":
    # Example usage
    audio_file = "pop.00099.wav"
    genre = "pop"
    
    # Extract features for a single file
    features = extract_features(audio_file, genre)
    
    # Save features for a single file
    output_dir = Path(os.path.dirname(__file__)) / "feature-extraction-output"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "features_output.csv"
    
    save_features_to_csv([features], output_file)
    print(f"Feature extraction complete")
