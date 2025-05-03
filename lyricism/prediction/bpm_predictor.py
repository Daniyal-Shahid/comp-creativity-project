import librosa
import numpy as np
import pandas as pd
import joblib
import os

def predict_bpm(audio_file):
    y, sr = librosa.load(audio_file, sr=None)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    
    print(f'Estimated tempo: {tempo} BPM')

# Create a DataFrame with the results
    results_df = pd.DataFrame({
        'estimated_tempo': [float(tempo)]
    })
    
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Save the results to a CSV file
    results_df.to_csv('bpm_prediction.csv', index=False)
    
    return float(tempo), beat_times
