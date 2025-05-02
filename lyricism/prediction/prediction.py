import joblib
import numpy as np
import os
import sys

# Add parent directory to path to allow import from feature_extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor.extract_features import extract_features as original_extract_features

def safe_extract_features(audio_file_path):
    """
    Wrapper around the original extract_features function that handles the csv_format attribute issue
    and fixes potential array shape issues
    
    Parameters:
    -----------
    audio_file_path : str
        Path to the audio file
        
    Returns:
    --------
    numpy.ndarray
        Feature vector ready for prediction
    """
    try:
        # Try the original approach first
        features = original_extract_features(audio_file_path)
        # Ensure all elements are float to avoid inhomogeneous array issues
        if isinstance(features, np.ndarray) and features.dtype != np.float64:
            features = features.astype(np.float64)
        return features
    except (AttributeError, ValueError) as e:
        print(f"Error encountered: {str(e)}")
        if "csv_format" in str(e) or "inhomogeneous shape" in str(e):
            # Handle both the csv_format attribute error and the inhomogeneous shape error
            print("Handling extraction error with custom implementation...")
            
            import librosa
            # Load the audio file
            y, sr = librosa.load(audio_file_path, mono=True, duration=30)
            
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
            # Ensure all values are floats and handled consistently
            features_list = [
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
            ]
            features_list.extend(mfcc_means)
            features_list.extend(mfcc_vars)
            
            # Debug: Print feature vector length
            print(f"Feature vector length: {len(features_list)}")
            
            feature_vector = np.array(features_list, dtype=np.float64)
            return feature_vector
        else:
            # If it's a different error, re-raise it
            raise

def predict_genre(audio_file_path):
    """
    Predict the genre of an audio file
    
    Parameters:
    -----------
    audio_file_path : str
        Path to the audio file
        
    Returns:
    --------
    str
        Predicted genre
    """
    print(f"Predicting genre for: {audio_file_path}")
    
    # Step 1: Load the trained model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training/genre_predictor.pkl')
    model = joblib.load(model_path)
    
    try:
        # Step 2: Extract features from the audio file using our safe wrapper
        features = safe_extract_features(audio_file_path)
        
        # Print feature shape for debugging
        print(f"Feature shape: {features.shape}")
        
        # Step 3: Ensure features are in the correct format (1 sample with many features)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Step 4: Predict genre
        predicted_genre = model.predict(features)
        return predicted_genre[0]
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == "__main__":
    # Get the absolute path of the audio file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(current_dir, 'soft_spot_30_sec.wav')
    
    # Predict genre
    genre = predict_genre(audio_file)
    print(f"Predicted genre: {genre}")
