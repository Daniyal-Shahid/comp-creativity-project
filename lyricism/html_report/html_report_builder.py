import os
from datetime import datetime

def create_html_report(audio_filename, genre, confidence, bpm, lyrics, prompt, sentiment_result, wordcloud_path, sentiment_progression=None, progression_chart_path=None):
    """Create a nice HTML report with all the results"""
    print("\nCreating HTML report...")
    
    # Get parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create a timestamp for the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format sentiment scores for display
    raw_scores_html = "<ul>"
    for key, value in sentiment_result['raw_scores'].items():
        raw_scores_html += f"<li><strong>{key.capitalize()}:</strong> {value:.4f}</li>"
    raw_scores_html += "</ul>"
    
    # Format sentiment interpretation for display
    interpretation_html = "<ul>"
    for key, value in sentiment_result['interpretation'].items():
        interpretation_html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
    interpretation_html += "</ul>"
    
    # Convert image paths to absolute URLs for the HTML file
    # This uses absolute paths with the file:// protocol to ensure images display properly
    wordcloud_abs_path = f"file://{os.path.abspath(wordcloud_path)}"
    
    # Create variable for emotional progression chart if provided
    progression_chart_html = ""
    if progression_chart_path:
        progression_chart_abs_path = f"file://{os.path.abspath(progression_chart_path)}"
        progression_chart_html = f"""
        <div class="section">
            <h2 class="section-title">Emotional Progression</h2>
            <p>This chart shows how the emotional tone and subjectivity change throughout different sections of the song.</p>
            <div class="visual-container">
                <img src="{progression_chart_abs_path}" alt="Emotional Progression Throughout the Song">
            </div>
        </div>
        """
    
    # Create the HTML content
    html_content = f"""
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lyricism - Song Analysis</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        :root {{
            --primary: #3a86ff;
            --accent: #8338ec;
            --light: #f8f9fa;
            --dark: #2b2d42;
            --shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background-color: #f9f9f9;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        /* Header Styles */
        header {{
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--primary);
            position: relative;
        }}
        
        h1 {{
            font-size: 2.5rem;
            color: var(--dark);
            margin-bottom: 0.5rem;
        }}
        
        .timestamp {{
            color: #666;
            font-size: 0.9rem;
        }}
        
        /* Container Styles */
        .container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: var(--shadow);
            padding: 25px;
            margin-bottom: 25px;
            transition: transform 0.3s ease;
        }}
        
        .container:hover {{
            transform: translateY(-3px);
        }}
        
        /* Section Styles */
        .section {{
            margin-bottom: 1.5rem;
        }}
        
        .section-title {{
            color: var(--primary);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #eee;
            position: relative;
        }}
        
        .section-title::after {{
            content: '';
            position: absolute;
            bottom: -1px;
            left: 0;
            width: 50px;
            height: 2px;
            background: var(--accent);
            transition: width 0.3s ease;
        }}
        
        .section:hover .section-title::after {{
            width: 100px;
        }}
        
        /* Info Box Styles */
        .info-box {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-bottom: 1rem;
        }}
        
        .info-item {{
            flex: 1;
            min-width: 200px;
            padding: 15px;
            background-color: var(--light);
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
        }}
        
        .info-item:hover {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .info-item-title {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: var(--accent);
        }}
        
        /* Lyrics Styles */
        .lyrics {{
            white-space: pre-line;
            line-height: 1.8;
            padding: 20px;
            background-color: var(--light);
            border-left: 3px solid var(--primary);
            border-radius: 0 6px 6px 0;
            margin: 1rem 0;
        }}
        
        /* Prompt Styles */
        .prompt {{
            white-space: pre-line;
            line-height: 1.6;
            padding: 20px;
            background-color: #f0f0f0;
            border-radius: 6px;
            font-size: 0.95rem;
            color: #444;
        }}
        
        /* Sentiment Analysis Styles */
        .sentiment-analysis {{
            padding: 20px;
            background-color: var(--light);
            border-radius: 6px;
        }}
        
        .sentiment-section {{
            margin-bottom: 1.5rem;
        }}
        
        .sentiment-section-title {{
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 0.75rem;
        }}
        
        /* Visual Container Styles */
        .visual-container {{
            text-align: center;
            margin: 1.5rem 0;
        }}
        
        .visual-container img {{
            max-width: 100%;
            border-radius: 6px;
            box-shadow: var(--shadow);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .visual-container img:hover {{
            transform: scale(1.01);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }}
        
        /* Footer Styles */
        footer {{
            text-align: center;
            margin-top: 2rem;
            padding: 1.5rem 0;
            color: #666;
            border-top: 1px solid #eee;
        }}
        
        /* Audio Visualizer */
        .music-visualizer {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 30px;
            margin: 1rem 0;
        }}
        
        .bar {{
            background: var(--primary);
            width: 3px;
            margin: 0 2px;
            border-radius: 3px;
            animation: sound-wave 1.5s ease-in-out infinite;
        }}
        
        .bar:nth-child(1) {{ height: 15px; animation-delay: 0.1s; }}
        .bar:nth-child(2) {{ height: 20px; animation-delay: 0.2s; }}
        .bar:nth-child(3) {{ height: 25px; animation-delay: 0.3s; }}
        .bar:nth-child(4) {{ height: 20px; animation-delay: 0.4s; }}
        .bar:nth-child(5) {{ height: 15px; animation-delay: 0.5s; }}
        
        @keyframes sound-wave {{
            0%, 100% {{ transform: scaleY(1); }}
            50% {{ transform: scaleY(0.7); }}
        }}
        
        /* Navigation */
        .nav {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 1rem 0;
        }}
        
        .nav a {{
            text-decoration: none;
            color: var(--dark);
            padding: 5px;
            position: relative;
        }}
        
        .nav a::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 2px;
            background: var(--primary);
            transition: width 0.3s ease;
        }}
        
        .nav a:hover::after {{
            width: 100%;
        }}
        
        /* Progress Bar */
        .progress-container {{
            width: 100%;
            height: 6px;
            background-color: #eee;
            border-radius: 3px;
            margin-top: 0.5rem;
            overflow: hidden;
        }}
        
        .progress-bar {{
            height: 100%;
            background: linear-gradient(to right, var(--primary), var(--accent));
            width: 75%;
            border-radius: 3px;
            animation: progress 1.5s ease-out;
        }}
        
        @keyframes progress {{
            from {{ width: 0; }}
            to {{ width: 75%; }}
        }}
        
        /* Button */
        .btn {{
            display: inline-block;
            padding: 8px 16px;
            background: var(--primary);
            color: white;
            border-radius: 20px;
            text-decoration: none;
            margin-top: 1rem;
            box-shadow: 0 3px 8px rgba(58, 134, 255, 0.3);
            cursor: pointer;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .info-item {{
                min-width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>Lyricism</h1>
        <p>Song Analysis & Lyric Generation</p>
        <p class="timestamp">Generated on {timestamp}</p>
        
        <div class="music-visualizer">
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
            <div class="bar"></div>
        </div>
        
        <div class="nav">
            <a href="#audio">Audio</a>
            <a href="#lyrics">Lyrics</a>
            <a href="#analysis">Analysis</a>
            <a href="#visuals">Visuals</a>
        </div>
    </header>
    
    <div class="container" id="audio">
        <div class="section">
            <h2 class="section-title">Audio Analysis</h2>
            <div class="info-box">
                <div class="info-item">
                    <div class="info-item-title">Audio File</div>
                    <div>{os.path.basename(audio_filename)}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-title">Genre</div>
                    <div>{genre}</div>
                </div>
                <div class="info-item">
                    <div class="info-item-title">Confidence</div>
                    <div>
                        {confidence}
                        <div class="progress-container">
                            <div class="progress-bar"></div>
                        </div>
                    </div>
                </div>
                <div class="info-item">
                    <div class="info-item-title">BPM</div>
                    <div>{bpm}</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container" id="lyrics">
        <div class="section">
            <h2 class="section-title">Generated Lyrics</h2>
            <div class="lyrics">{lyrics}</div>
            <div class="btn">Download Lyrics</div>
        </div>
    </div>
    
    <div class="container">
        <div class="section">
            <h2 class="section-title">Prompt Used for Generation</h2>
            <div class="prompt">{prompt}</div>
        </div>
    </div>

    <div class="container" id="analysis">
        <div class="section">
            <h2 class="section-title">Sentiment Analysis</h2>
            <div class="sentiment-analysis">
                <div class="sentiment-section">
                    <div class="sentiment-section-title">Interpretation</div>
                    {interpretation_html}
                </div>
                <div class="sentiment-section">
                    <div class="sentiment-section-title">Raw Scores</div>
                    {raw_scores_html}
                </div>
            </div>
        </div>
        
        {progression_chart_html}
    </div>
    
    <div class="container" id="visuals">
        <div class="section">
            <h2 class="section-title">Word Cloud</h2>
            <div class="visual-container">
                <img src="{wordcloud_abs_path}" alt="Word Cloud for Lyrics">
            </div>
        </div>
    </div>
    
    <footer>
        <p>Lyricism - A Music Analysis and Lyric Generation Tool</p>
        <p>© 2025</p>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Smooth scrolling for anchor links
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                anchor.addEventListener('click', function(e) {{
                    e.preventDefault();
                    document.querySelector(this.getAttribute('href')).scrollIntoView({{
                        behavior: 'smooth'
                    }});
                }});
            }});
            
            // Simple interactivity for download button
            const downloadBtn = document.querySelector('.btn');
            if (downloadBtn) {{
                downloadBtn.addEventListener('click', function() {{
                    alert('Lyrics downloaded successfully!');
                }});
            }}
        }});
    </script>
</body>
</html>
    """
    
    # Create output folder if it doesn't exist
    output_dir = os.path.join(parent_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a filename based on the audio file
    base_filename = os.path.splitext(os.path.basename(audio_filename))[0]
    report_filename = f"{base_filename}_analysis.html"
    report_path = os.path.join(output_dir, report_filename)
    
    # Write the HTML file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report saved to {report_path}")
    
    return report_path