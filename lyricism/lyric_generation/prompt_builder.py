import pandas as pd
import os

def build_prompt(predicted_genre, bpm):
    # 1. Set mood based on BPM
    if bpm > 150:
        mood = "energetic and intense"
        tone = "passionate and powerful"
    elif bpm > 120:
        mood = "upbeat and lively"
        tone = "confident and expressive"
    elif bpm > 90:
        mood = "moderate and emotive"
        tone = "thoughtful and resonant"
    elif bpm > 70:
        mood = "laid-back and reflective"
        tone = "introspective and genuine"
    else:
        mood = "slow and melancholic"
        tone = "intimate and vulnerable"
    
    # 2. Genre-specific guidance
    genre_guidance = {
        "pop": "catchy hooks, relatable themes, and contemporary language. Include repetitive, memorable choruses.",
        "rock": "powerful imagery, emotional depth, and some edge. Balance between intensity and vulnerability.",
        "hip hop": "clever wordplay, authentic storytelling, and rhythmic flow. Include some internal rhymes.",
        "r&b": "soulful themes, smooth delivery, and emotional honesty. Focus on relationships and feelings.",
        "country": "storytelling, authenticity, and relatability. Include references to everyday life experiences.",
        "electronic": "minimalist, atmospheric lyrics that complement the electronic production.",
        "jazz": "sophisticated, poetic lyrics with room for improvisation in delivery.",
        "folk": "narrative-driven, authentic storytelling with natural imagery and emotional depth.",
        "metal": "intense imagery, powerful themes, and dramatic language.",
        "reggae": "positive messages, cultural references, and rhythmic delivery.",
        "disco": "celebratory, upbeat themes focused on dancing and good times.",
        "indie": "quirky, personal perspectives with distinctive imagery and emotional honesty."
    }
    
    # Get genre guidance or use a default if genre not in dictionary
    specific_guidance = genre_guidance.get(
        predicted_genre.lower(), 
        "unique perspective, authentic emotion, and thoughtful imagery"
    )
    
    # 3. Build song structure guidance
    structure_guidance = (
        "Create complete lyrics with the following structure:\n"
        "- Verse 1: Introduce the central theme or story (4-8 lines)\n"
        "- Pre-Chorus: Build tension leading to the chorus (2-4 lines)\n"
        "- Chorus: Deliver the main message with memorable hooks (4-6 lines)\n"
        "- Verse 2: Expand on the theme with new details (4-8 lines)\n"
        "- Chorus: Repeat with slight variations if needed\n"
        "- Bridge: Change perspective or add contrast (2-4 lines)\n"
        "- Final Chorus: Intensify the emotional impact\n"
        "- Outro: Provide closure (2-4 lines)\n\n"
        "Clearly label each section in the lyrics (e.g., '[Verse 1]', '[Chorus]')."
    )
    
    # 4. Build final prompt
    prompt = (
        f"You are an experienced lyricist tasked with writing original, high-quality song lyrics.\n\n"
        f"GENRE: {predicted_genre}\n"
        f"MOOD: {mood}\n"
        f"TEMPO: {bpm} BPM\n"
        f"TONE: {tone}\n\n"
        f"STYLE GUIDANCE:\n"
        f"Write in the style of {predicted_genre} music, incorporating {specific_guidance}\n"
        f"The lyrics should reflect a {mood} atmosphere appropriate for {bpm} BPM.\n"
        f"Use a {tone} tone throughout.\n\n"
        f"STRUCTURE:\n{structure_guidance}\n\n"
        f"ADDITIONAL GUIDELINES:\n"
        f"- Create a clear title that captures the essence of the song\n"
        f"- Maintain consistent rhyme schemes appropriate for {predicted_genre}\n"
        f"- Use imagery and metaphors that resonate with fans of this genre\n"
        f"- Create an emotional arc across the song\n"
        f"- Make the chorus memorable and distinctive from verses\n\n"
        f"Please respond with only the complete, original lyrics formatted with clear section labels. Do not include any other text or comments."
    )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save prompt to HTML file
    prompt_df = pd.DataFrame({
        "prompt": [prompt]
    })
    
    
    prompt_html_path = os.path.join(output_dir, "prompt.html")
    prompt_df.to_html(prompt_html_path, index=False)
    
    return prompt
