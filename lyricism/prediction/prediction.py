import joblib
import numpy as np
import os
import sys

import pandas as pd

# Add parent directory to path to allow import from feature_extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor.extract_features import extract_features as original_extract_features
from prediction.bpm_predictor import predict_bpm

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

def predict_genre(audio_file_path, use_calibrated_model=True):
    """
    Predict the genre of an audio file
    
    Parameters:
    -----------
    audio_file_path : str
        Path to the audio file
    use_calibrated_model : bool, optional
        Whether to use the calibrated model with improved confidence scores
        
    Returns:
    --------
    tuple
        (predicted_genre, confidence): The predicted genre and confidence probability
    """
    print(f"Predicting genre for: {audio_file_path}")
    
    # Step 1: Load the trained model and scaler
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if use_calibrated_model:
        # Use calibrated model for better confidence scores
        model_path = os.path.join(parent_dir, 'training/models/calibrated_genre_predictor.pkl')
        scaler_path = os.path.join(parent_dir, 'training/models/feature_scaler.pkl')
        
        # Check if calibrated model exists
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            print("Using calibrated model with feature scaling")
        else:
            # Fall back to original model if calibrated model not found
            model_path = os.path.join(parent_dir, 'training/genre_predictor.pkl')
            model = joblib.load(model_path)
            scaler = None
            print("Calibrated model not found, using original model")
    else:
        # Use original model without calibration
        model_path = os.path.join(parent_dir, 'training/genre_predictor.pkl')
        model = joblib.load(model_path)
        scaler = None
        print("Using original model without calibration")
    
    try:
        # Step 2: Extract features from the audio file using our safe wrapper
        features = safe_extract_features(audio_file_path)
        
        # Print feature shape for debugging
        print(f"Feature shape: {features.shape}")
        
        # Step 3: Apply feature scaling if available
        if scaler is not None:
            features = scaler.transform(features.reshape(1, -1))
            print("Applied feature scaling")
        elif features.ndim == 1:
            # Ensure features are in the correct format (1 sample with many features)
            features = features.reshape(1, -1)
        
        # Step 4: Get prediction and probability
        predicted_genre = model.predict(features)[0]
        
        # Get probability scores if the model supports it
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            # Find the index of the predicted class
            predicted_index = None
            for i, class_label in enumerate(model.classes_):
                if class_label == predicted_genre:
                    predicted_index = i
                    break
            
            if predicted_index is not None:
                confidence = probabilities[predicted_index]
        
        return predicted_genre, confidence
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise
    
    

def predict_multiple_genres(audio_file_path, top_n=3):
    """
    Predict the top N most likely genres for an audio file
    
    Parameters:
    -----------
    audio_file_path : str
        Path to the audio file
    top_n : int, optional
        Number of top genres to return
        
    Returns:
    --------
    list of tuples
        List of (genre, confidence) tuples, sorted by confidence (highest first)
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try to use calibrated model first
    model_path = os.path.join(parent_dir, 'training/models/calibrated_genre_predictor.pkl')
    scaler_path = os.path.join(parent_dir, 'training/models/feature_scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Using calibrated model with feature scaling for multiple genres")
    else:
        # Fall back to original model
        model_path = os.path.join(parent_dir, 'training/genre_predictor.pkl')
        model = joblib.load(model_path)
        scaler = None
        print("Using original model for multiple genres")
    
    try:
        # Extract features
        features = safe_extract_features(audio_file_path)
        
        # Apply feature scaling if available
        if scaler is not None:
            features = scaler.transform(features.reshape(1, -1))
        elif features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Get probability scores for all genres
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            
            # Create list of (genre, probability) tuples
            genre_probs = [(model.classes_[i], prob) for i, prob in enumerate(probabilities)]
            
            # Sort by probability (highest first) and take top N
            genre_probs.sort(key=lambda x: x[1], reverse=True)
            return genre_probs[:top_n]
        else:
            # If model doesn't support probabilities, return only the top prediction
            predicted_genre = model.predict(features)[0]
            return [(predicted_genre, None)]
    except Exception as e:
        print(f"Error during multiple genre prediction: {str(e)}")
        raise

if __name__ == "__main__":
    # Get the absolute path of the audio file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use command line argument for audio file path if provided
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Default to a test file if no argument provided
        audio_file = os.path.join(current_dir, 'soft_spot_30_sec.wav')
    
    print("Predicting BPM...")
    bpm, beat_times = predict_bpm(audio_file)
    print(f"Estimated BPM: {bpm} \n Beat times: {beat_times}")

    # Get the top genres first to ensure consistency between console and CSV output
    print("\nPredicting top genres...")
    top_genres = predict_multiple_genres(audio_file, top_n=3)
    
    # Use the top genre as the main prediction
    main_genre, main_confidence = top_genres[0]
    
    # Print main prediction
    print(f"\nPredicted genre: {main_genre}")
    if main_confidence is not None:
        print(f"Confidence: {main_confidence:.2%}")
    else:
        print("Confidence: Not available (model doesn't support probability estimation)")
    
    # Print all top genres
    print("\nTop 3 genre predictions:")
    for i, (genre, prob) in enumerate(top_genres, 1):
        if prob is not None:
            print(f"{i}. {genre}: {prob:.2%}")
        else:
            print(f"{i}. {genre}: probability not available")
    
    # Create a DataFrame with the prediction results
    prediction_data = {'predicted_genre': [main_genre]}
    if main_confidence is not None:
        prediction_data['confidence'] = [f"{main_confidence:.2%}"]
    
    # Add top alternative genres if available
    if len(top_genres) > 1:
        alt_genres = []
        alt_confidences = []
        for alt_genre, alt_prob in top_genres[1:]:
            alt_genres.append(alt_genre)
            if alt_prob is not None:
                alt_confidences.append(f"{alt_prob:.2%}")
            else:
                alt_confidences.append("N/A")
        
        prediction_data['alternative_genres'] = [', '.join(alt_genres)]
        prediction_data['alternative_confidences'] = [', '.join(alt_confidences)]
    
    prediction_df = pd.DataFrame(prediction_data)
    
    # Save to CSV, overwriting if it exists
    prediction_df.to_csv('genre_prediction.csv', index=False)
    print(f"\nPrediction saved to genre_prediction.csv")
    print(f"Main genre: {main_genre}, Confidence: {main_confidence:.2%}")
    
    # Display CSV content for verification
    csv_content = pd.read_csv('genre_prediction.csv')
    print("\nCSV Content Verification:")
    print(csv_content)
