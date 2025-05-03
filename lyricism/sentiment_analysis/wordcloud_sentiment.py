import nltk
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from collections import Counter
import os
import re
from textblob import TextBlob
import numpy as np
import seaborn as sns

# Download necessary NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    print("Warning: NLTK data download may have failed. Internet connection required for first-time setup.")

def clean_lyrics(text):
    """
    Cleans lyrics text by removing structural elements, metadata and other non-lyrical content
    
    Parameters:
    -----------
    text : str
        The raw lyrics text to clean
        
    Returns:
    --------
    str
        Cleaned lyrics text
    """
    # Remove title and metadata at the beginning
    text = re.sub(r'^.*?\*\*Title:.*?\*\*\n', '', text, flags=re.DOTALL)
    
    # Remove structural elements like [VERSE], [CHORUS], etc.
    text = re.sub(r'\[(Verse|Chorus|Pre-Chorus|Bridge|Intro|Outro|Hook|Refrain|Interlude)( \d+)?\]', '', text, flags=re.IGNORECASE)
    
    # Remove production notes, artist names, etc.
    text = re.sub(r'\((.*?)\)', '', text)
    
    # Remove other metadata markers
    text = re.sub(r'\*\*(.*?)\*\*', '', text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    
    return text

def analyze_sentiment(text):
    """
    Analyse the sentiment of the provided text with enhanced interpretation
    
    Parameters:
    -----------
    text : str
        The text to analyze
        
    Returns:
    --------
    dict
        Dictionary containing sentiment scores and interpretation
    """
    # Clean the lyrics first
    cleaned_text = clean_lyrics(text)
    
    # Use TextBlob for sentiment analysis
    blob = TextBlob(cleaned_text)
    
    # TextBlob returns polarity (-1 to 1) and subjectivity (0 to 1)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Convert TextBlob scores to a dictionary
    scores = {
        'polarity': polarity,
        'subjectivity': subjectivity
    }
    
    # Add sentiment interpretation
    interpretation = {}
    
    # Overall mood interpretation based on polarity
    if polarity >= 0.05:
        interpretation['overall_mood'] = 'Positive'
    elif polarity <= -0.05:
        interpretation['overall_mood'] = 'Negative'
    else:
        interpretation['overall_mood'] = 'Neutral'
    
    # Emotional intensity based on absolute polarity
    intensity = abs(polarity)
    if intensity >= 0.5:
        interpretation['emotional_intensity'] = 'High'
    elif intensity >= 0.2:
        interpretation['emotional_intensity'] = 'Moderate'
    else:
        interpretation['emotional_intensity'] = 'Low'
    
    # Subjectivity interpretation
    if subjectivity >= 0.7:
        interpretation['subjectivity'] = 'Highly Subjective'
    elif subjectivity >= 0.4:
        interpretation['subjectivity'] = 'Moderately Subjective'
    else:
        interpretation['subjectivity'] = 'Mostly Objective'
    
    # Create the enhanced result dictionary
    result = {
        'raw_scores': scores,
        'interpretation': interpretation
    }
    
    return result

def analyze_sentiment_over_time(text, save_path=None):
    """
    Analyse how sentiment changes throughout the lyrics
    
    Parameters:
    -----------
    text : str
        The lyrics to analyze
    save_path : str, optional
        Path to save the sentiment progression visualization
        
    Returns:
    --------
    tuple
        (Analysis results dictionary, Path to the saved visualization)
    """
    # Create output directory if it doesn't exist
    if save_path is None:
        # Default path in output directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(parent_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "sentiment_progression.png")
    
    # Clean the lyrics
    cleaned_text = clean_lyrics(text)
    
    # Extract sections using structural markers in the original text
    section_pattern = r'\[(Verse|Chorus|Pre-Chorus|Bridge|Intro|Outro|Hook|Refrain|Interlude)( \d+)?\](.*?)(?=\[|$)'
    sections = re.findall(section_pattern, text, re.DOTALL | re.IGNORECASE)
    
    # If no sections found, split by lines instead (fallback)
    if not sections:
        # Split into chunks of approximately equal size
        lines = cleaned_text.split('\n')
        chunks = []
        chunk_size = max(3, len(lines) // 8)  # Aim for around 8 chunks
        
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(('Line', str(i//chunk_size+1), chunk))
    else:
        chunks = sections
    
    # Analyze sentiment for each section
    sentiment_data = []
    
    for section_type, section_num, content in chunks:
        if not content.strip():
            continue
            
        # Analyze this section
        blob = TextBlob(content.strip())
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Create section name
        if section_num.strip():
            section_name = f"{section_type} {section_num.strip()}"
        else:
            section_name = section_type
            
        sentiment_data.append({
            'section': section_name,
            'content': content.strip(),
            'polarity': polarity,
            'subjectivity': subjectivity
        })
    
    # Create visualization
    if sentiment_data:
        plt.figure(figsize=(12, 8))
        
        # Set up the style
        sns.set_style("whitegrid")
        
        # Plot polarity
        ax1 = plt.subplot(2, 1, 1)
        section_names = [data['section'] for data in sentiment_data]
        polarity_values = [data['polarity'] for data in sentiment_data]
        
        bars = plt.bar(section_names, polarity_values, color=plt.cm.RdYlGn(np.array(polarity_values)/2 + 0.5))
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.title('Emotional Progression Throughout the Song', fontsize=16)
        plt.ylabel('Emotional Tone\n(Negative ↔ Positive)', fontsize=12)
        plt.ylim(-1.1, 1.1)
        
        # Add polarity labels
        for i, v in enumerate(polarity_values):
            if v >= 0:
                plt.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
            else:
                plt.text(i, v - 0.05, f"{v:.2f}", ha='center', va='top', fontsize=10)
        
        # Plot subjectivity
        ax2 = plt.subplot(2, 1, 2)
        subjectivity_values = [data['subjectivity'] for data in sentiment_data]
        
        plt.bar(section_names, subjectivity_values, color=plt.cm.Blues(np.array(subjectivity_values)))
        plt.title('Subjectivity Throughout the Song', fontsize=16)
        plt.ylabel('Subjectivity\n(Factual ↔ Emotional)', fontsize=12)
        plt.ylim(0, 1.1)
        
        # Add subjectivity labels
        for i, v in enumerate(subjectivity_values):
            plt.text(i, v + 0.05, f"{v:.2f}", ha='center', va='bottom', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Sentiment progression visualization saved to {save_path}")
    
    # Return both the data and the visualization path
    result = {
        'sentiment_progression': sentiment_data,
        'overall_sentiment': analyze_sentiment(cleaned_text)
    }
    
    return result, save_path

def generate_wordcloud(text, save_path=None):
    """
    Generate a wordcloud from the provided text and optionally save it to disk
    
    Parameters:
    -----------
    text : str
        The text to generate the wordcloud from
    save_path : str, optional
        Path to save the wordcloud image to
        
    Returns:
    --------
    str or None
        The path to the saved wordcloud image if save_path is provided,
        otherwise None after displaying the wordcloud
    """
    # Create output directory if it doesn't exist
    if save_path is None:
        # Default path in output directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(parent_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "wordcloud.png")
    
    # Clean the lyrics first
    cleaned_text = clean_lyrics(text)
    
    # Define custom stopwords specific to lyrics
    custom_stopwords = set(['verse', 'chorus', 'bridge', 'intro', 'outro', 
                          'lyrics', 'title', 'song', 'repeat', 'x2', 'x3', 'x4',
                          'pre', 'hook', 'refrain', 'interlude'])
    
    # Combine with standard stopwords
    stopwords = set(STOPWORDS).union(custom_stopwords)
    
    # Generate the wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=stopwords, 
        max_words=100,
        colormap='viridis',
        contour_width=1, 
        contour_color='steelblue',
        collocations=False  # Disable collocations to focus on individual words
    ).generate(cleaned_text)
    
    # Save to file
    wordcloud.to_file(save_path)
    print(f"Wordcloud saved to {save_path}")
    
    # Return the path
    return save_path

def save_wordcloud(text, output_path="wordcloud.png"):
    """
    Save a wordcloud from the provided text to the specified output path
    
    Parameters:
    -----------
    text : str
        The text to generate the wordcloud from
    output_path : str
        Path to save the wordcloud image to
        
    Returns:
    --------
    str
        The path to the saved wordcloud image
    """
    return generate_wordcloud(text, save_path=output_path)

def save_sentiment_scores(text, output_path="sentiment_scores.txt"):
    """
    Save sentiment scores for the provided text to the specified output path
    
    Parameters:
    -----------
    text : str
        The text to analyze
    output_path : str
        Path to save the sentiment scores to
        
    Returns:
    --------
    dict
        Dictionary of sentiment scores and interpretation
    """
    result = analyze_sentiment(text)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    
    # Format the sentiment analysis results for readability
    with open(output_path, 'w') as f:
        f.write("Sentiment Analysis Results\n")
        f.write("=========================\n\n")
        
        f.write("Raw Scores:\n")
        for key, value in result['raw_scores'].items():
            f.write(f"  {key}: {value:.4f}\n")
        
        f.write("\nInterpretation:\n")
        for key, value in result['interpretation'].items():
            f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
    
    return result
