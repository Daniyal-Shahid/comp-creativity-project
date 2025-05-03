import os
import sys
import subprocess
import pandas as pd
import shutil
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import components from other modules
from prediction.prediction import predict_genre, predict_multiple_genres
from prediction.bpm_predictor import predict_bpm
from lyric_generation.prompt_builder import build_prompt
from lyric_generation.generate_lyrics import generate_lyrics
from sentiment_analysis.wordcloud_sentiment import analyze_sentiment as sentiment_analyzer, generate_wordcloud, analyze_sentiment_over_time
from html_report.html_report_builder import create_html_report

def check_and_train_models():
    """Check if models exist and train them if they don't"""
    print("Checking if models exist...")
    
    # Define paths to model files
    training_dir = os.path.join(parent_dir, 'training')
    base_model_path = os.path.join(training_dir, 'genre_predictor.pkl')
    calibrated_model_path = os.path.join(training_dir, 'models', 'calibrated_genre_predictor.pkl')
    scaler_path = os.path.join(training_dir, 'models', 'feature_scaler.pkl')
    
    # Check if models exist
    models_exist = os.path.exists(base_model_path) and \
                  os.path.exists(calibrated_model_path) and \
                  os.path.exists(scaler_path)
    
    if not models_exist:
        print("Models not found. Running training script...")
        
        # Change to training directory
        os.chdir(training_dir)
        
        # Run the training script
        try:
            subprocess.run([sys.executable, 'training.py'], check=True)
            print("Training completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Error during training: {str(e)}")
            sys.exit(1)
        
        # Return to original directory
        os.chdir(current_dir)
    else:
        print("All models found. Skipping training step.")

def predict_audio_features(audio_file_path):
    """Run prediction on the audio file"""
    print(f"Analysing audio file: {audio_file_path}")
    
    # Change to prediction directory for consistent file outputs
    prediction_dir = os.path.join(parent_dir, 'prediction')
    os.chdir(prediction_dir)
    
    # Predict genre
    print("\nPredicting genre...")
    top_genres = predict_multiple_genres(audio_file_path, top_n=3)
    
    # Extract main genre and confidence
    main_genre, main_confidence = top_genres[0]
    
    # Print main prediction
    print(f"\nPredicted genre: {main_genre}")
    if main_confidence is not None:
        print(f"Confidence: {main_confidence:.2%}")
    
    # Print all top genres
    print("\nTop 3 genre predictions:")
    for i, (genre, prob) in enumerate(top_genres, 1):
        if prob is not None:
            print(f"{i}. {genre}: {prob:.2%}")
        else:
            print(f"{i}. {genre}: probability not available")
    
    # Create genre prediction DataFrame
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
    
    # Save genre prediction to CSV
    prediction_df.to_csv('genre_prediction.csv', index=False)
    print("Genre prediction saved to genre_prediction.csv")
    
    # Predict BPM
    print("\nPredicting BPM...")
    bpm, beat_times = predict_bpm(audio_file_path)
    print(f"Estimated BPM: {bpm}")
    
    # Return to original directory
    os.chdir(current_dir)
    
    return main_genre, main_confidence, bpm

def generate_lyrics(genre, bpm):
    """Generate lyrics based on genre and BPM"""
    print("\nGenerating lyrics...")
    
    # Prepare paths for CSV files
    prediction_dir = os.path.join(parent_dir, 'prediction')
    genre_csv = os.path.join(prediction_dir, 'genre_prediction.csv')
    bpm_csv = os.path.join(prediction_dir, 'bpm_prediction.csv')
    
    # Check if needed CSVs exist
    if not os.path.exists(genre_csv) or not os.path.exists(bpm_csv):
        print("Error: Required prediction files not found.")
        return "Error: Could not generate lyrics due to missing prediction files.", ""
    
    # Build the prompt for reference
    prompt = build_prompt(predicted_genre=genre, bpm=bpm)
    
    # Import and use the generate_lyrics function from the module
    from lyric_generation.generate_lyrics import generate_lyrics as gen_lyrics
    
    # Generate lyrics using the module function
    lyrics = gen_lyrics(genre_csv, bpm_csv)
    
    if not lyrics:
        lyrics = "Error: Could not generate lyrics. Please check your Gemini API key."
    
    # Save the lyrics to a file in the output directory
    output_dir = os.path.join(parent_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    lyrics_path = os.path.join(output_dir, 'generated_lyrics.txt')
    
    with open(lyrics_path, 'w') as f:
        f.write(lyrics)
    
    print(f"Lyrics saved to {lyrics_path}")
    
    return lyrics, prompt

def analyze_lyrics_sentiment(lyrics):
    """Analyze the sentiment of the lyrics"""
    print("\nAnalyzing sentiment...")
    
    # Overall sentiment analysis
    sentiment_result = sentiment_analyzer(lyrics)
    print(f"Sentiment scores: {sentiment_result['raw_scores']}")
    print(f"Interpretation: {sentiment_result['interpretation']}")

    # Generate wordcloud
    wordcloud = generate_wordcloud(lyrics)
    
    # Analyze sentiment progression over time
    print("\nAnalyzing sentiment progression throughout the song...")
    sentiment_progression, progression_chart = analyze_sentiment_over_time(lyrics)
    
    return sentiment_result, wordcloud, sentiment_progression, progression_chart

def main():
    """Main function that orchestrates the entire workflow"""
    print("=== Lyricism - Music Analysis and Lyric Generation ===")
    
    # Check if audio file is provided as command line argument
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        if not os.path.exists(audio_file):
            print(f"Error: Audio file '{audio_file}' not found.")
            sys.exit(1)
    else:
        # Use a default audio file for demonstration
        audio_file = os.path.join(parent_dir, 'prediction', 'soft_spot_30_sec.wav')
        if not os.path.exists(audio_file):
            print(f"Default audio file not found at {audio_file}")
            print("Please provide an audio file path as a command line argument.")
            sys.exit(1)
        print(f"Using default audio file: {audio_file}")
    
    # Step 1: Check for models and train if needed
    check_and_train_models()
    
    # Step 2: Predict genre and BPM
    genre, confidence, bpm = predict_audio_features(audio_file)
    
    # Step 3: Generate lyrics
    lyrics, prompt = generate_lyrics(genre, bpm)
    
    # Step 4: Analyze sentiment
    sentiment_result, wordcloud, sentiment_progression, progression_chart = analyze_lyrics_sentiment(lyrics)
    
    # Step 5: Create HTML report
    report_path = create_html_report(
        audio_filename=audio_file,
        genre=genre,
        confidence=f"{confidence:.2%}" if confidence else "N/A",
        bpm=bpm,
        lyrics=lyrics,
        prompt=prompt,
        sentiment_result=sentiment_result,
        wordcloud_path=wordcloud,
        sentiment_progression=sentiment_progression,
        progression_chart_path=progression_chart
    )
    
    # Step 6: Display final summary
    print("\n=== Analysis Complete ===")
    print(f"Audio file: {os.path.basename(audio_file)}")
    print(f"Genre: {genre} (Confidence: {confidence:.2%})" if confidence else f"Genre: {genre}")
    print(f"BPM: {bpm}")
    print(f"Report saved to: {report_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
