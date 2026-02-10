import torch
import sys
import os

# Mock the clip module or import it if available
# Adjust path to find clip
sys.path.append(os.path.abspath('.'))

try:
    import model.clip.clip as clip
    
    template = "A photo of a {} person"
    composed_str = template.format("X")
    print(f"Composed string: '{composed_str}'")
    
    token_ids = clip.tokenize([composed_str])
    print(f"Token IDs: {token_ids[0].tolist()[:15]}")
    
    # Find X position logic
    prefix_str = composed_str[:composed_str.find("X")]
    print(f"Prefix string: '{prefix_str}'")
    prefix_ids = clip.tokenize([prefix_str])
    print(f"Prefix IDs: {prefix_ids[0].tolist()[:15]}")
    
    x_pos = prefix_ids.shape[1] - 1
    print(f"x_pos: {x_pos}")
    
    # Verify what's at x_pos
    # Note: clip.tokenize adds SOT at index 0
    print(f"Token at x_pos: {token_ids[0, x_pos].item()}")
    
    # Check if 'X' as a single character tokenizes to what we expect
    x_token = clip.tokenize(["X"])[0, 1].item()
    print(f"Expected 'X' token: {x_token}")
    
    if token_ids[0, x_pos].item() == x_token:
        print("SUCCESS: x_pos correctly identifies 'X' token.")
    else:
        print("FAILURE: x_pos does NOT point to 'X' token.")
        # Let's find where X is
        for i in range(token_ids.shape[1]):
            if token_ids[0, i].item() == x_token:
                print(f"Actual 'X' is at index: {i}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
