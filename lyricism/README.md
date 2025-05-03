# Lyricism

A music analysis and lyric generation tool that analyses audio tracks to detect genre and tempo, then generates appropriate song lyrics based on those characteristics.

## Overview

Lyricism is an AI-powered tool that:

1. Analyses audio files to detect musical genre and tempo (BPM)
2. Generates original, genre-appropriate lyrics tailored to the detected tempo and style
3. Performs sentiment analysis and creates word clouds from the generated lyrics
4. Produces a beautiful HTML report containing all analysis results and visualisations

## Project Structure

- **app**: Main application orchestration
- **feature_extractor**: Audio feature extraction utilities
- **prediction**: Genre and BPM prediction models
- **training**: Model training scripts and data
- **lyric_generation**: Song lyrics generation with Google's Gemini API
- **sentiment_analysis**: Text analysis tools for evaluating lyric sentiment and generating word clouds
- **output**: Generated reports and files

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (install via pip):
  - librosa
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - google-generativeai
  - python-dotenv
  - nltk
  - wordcloud
  - matplotlib

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up API key:
   - Create a `.env` file in the project root
   - Add your Gemini API key: `GEMINI_API_KEY=your_api_key_here`

### Usage

To analyse an audio file and generate lyrics:

```
python app/main.py path/to/your/audio_file.wav
```

The application will:
1. Train or load existing genre prediction models
2. Analyse the audio file to detect genre and BPM
3. Generate appropriate lyrics based on the analysis
4. Perform sentiment analysis on the generated lyrics
5. Create a visual word cloud of key terms in the lyrics
6. Save all results to the `output` directory
7. Display a summary in the console

## Output

The application produces:
- A detailed HTML report with all analysis results
- A text file containing the generated lyrics
- A word cloud image generated from the lyrics
- CSV files with prediction data

## License

MIT 