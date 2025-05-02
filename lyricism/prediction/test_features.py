import os
import sys
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the safe feature extraction function
from prediction import safe_extract_features

def test_feature_extraction():
    """Test feature extraction on the test audio file and print diagnostics"""
    # Get the test audio file path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    audio_file = os.path.join(current_dir, 'soft_spot_30_sec.wav')
    
    print(f"Testing feature extraction on: {audio_file}")
    
    try:
        # Extract features
        features = safe_extract_features(audio_file)
        
        # Print information about the features
        print(f"Features shape: {features.shape}")
        print(f"Features type: {features.dtype}")
        print(f"Features size: {features.size}")
        
        # Check if the features are all finite (no NaN or Inf values)
        if np.all(np.isfinite(features)):
            print("All feature values are finite (no NaN or Inf)")
        else:
            print("WARNING: Features contain NaN or Inf values")
            # Print indices of non-finite values
            non_finite = np.where(~np.isfinite(features))[0]
            print(f"Non-finite values at indices: {non_finite}")
        
        # Print the first few and last few features
        print("\nFirst 5 features:")
        print(features[:5])
        print("\nLast 5 features:")
        print(features[-5:])
        
        print("\nFeature extraction test completed successfully")
        return True
    except Exception as e:
        print(f"Error during feature extraction test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_feature_extraction() 