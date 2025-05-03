import os
import sys
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lyric_generation.prompt_builder import build_prompt

def generate_lyrics(genre_csv=None, bpm_csv=None):
    """
    Generate lyrics based on genre and BPM data from CSV files
    
    Parameters:
    -----------
    genre_csv : str, optional
        Path to the genre prediction CSV file
    bpm_csv : str, optional
        Path to the BPM prediction CSV file
        
    Returns:
    --------
    str
        The generated lyrics
    """
    load_dotenv()
    
    # Set default paths if not provided
    if genre_csv is None:
        genre_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "prediction", "genre_prediction.csv")
    if bpm_csv is None:
        bpm_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "prediction", "bpm_prediction.csv")
    
    # Check if files exist
    if not os.path.exists(genre_csv):
        print(f"Error: Genre prediction file not found at {genre_csv}")
        return None
    if not os.path.exists(bpm_csv):
        print(f"Error: BPM prediction file not found at {bpm_csv}")
        return None
    
    # Read predicted genre from CSV
    try:
        genre_prediction = pd.read_csv(genre_csv)
        predicted_genre = genre_prediction["predicted_genre"].iloc[0]
    except Exception as e:
        print(f"Error reading genre prediction: {str(e)}")
        return None
    
    # Read BPM from CSV
    try:
        bpm_prediction = pd.read_csv(bpm_csv)
        bpm = float(bpm_prediction["estimated_tempo"].iloc[0])
    except Exception as e:
        print(f"Error reading BPM prediction: {str(e)}")
        return None
    
    # Build prompt for lyric generation
    prompt = build_prompt(predicted_genre=predicted_genre, bpm=bpm)
    
    # Check if API key is available
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment variables")
        return None
    
    # Set up Gemini API
    genai.configure(api_key=api_key)
    
    # Set up Gemini model
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Generate content
    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [prompt]}
            ],
            generation_config={
                "temperature": 0.8,
                "max_output_tokens": 300,
            }
        )
        
        generated_lyrics = response.text
        return generated_lyrics
    except Exception as e:
        print(f"Error during lyric generation: {str(e)}")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    genre_csv = None
    bpm_csv = None
    
    if len(sys.argv) > 1:
        genre_csv = sys.argv[1]
    if len(sys.argv) > 2:
        bpm_csv = sys.argv[2]
    
    # Generate lyrics
    lyrics = generate_lyrics(genre_csv, bpm_csv)
    
    if lyrics:
        print(lyrics)
        
        # Save the lyrics to a file
        output_file = "generated_lyrics.txt"
        with open(output_file, "w") as f:
            f.write(lyrics)
        print(f"Lyrics saved to {output_file}")
    else:
        print("Failed to generate lyrics.")
