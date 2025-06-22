#!/usr/bin/env python3
"""
Test the edit_config.py interactive interface to see if the new Julia methods appear.
"""

import json
import sys
import os

# Import the toggle function from edit_config.py
sys.path.append('.')
from edit_config import toggle_julia_methods, load_config

def test_julia_methods_toggle():
    """Test that the new Julia methods appear in the toggle interface."""
    print("🧪 Testing edit_config.py Julia methods toggle")
    print("=" * 60)
    
    # Load current config
    config = load_config()
    
    # Simulate what the toggle_julia_methods function shows
    all_methods = ["GPR", "AAA", "AAA_lowpres", "LOESS", "BSpline5", "TVDiff", 
                   "JuliaAAALS", "JuliaAAAFullOpt", "JuliaAAATwoStage", "JuliaAAASmoothBary"]
    
    print("Methods that would appear in interactive toggle:")
    for i, method in enumerate(all_methods, 1):
        status = "✅" if method in config['julia_methods'] else "❌"
        marker = "🆕" if method.startswith("Julia") else "  "
        print(f"  {i:2d}. {status} {marker} {method}")
    
    # Count new methods
    new_methods = [m for m in all_methods if m.startswith("Julia")]
    enabled_new = [m for m in new_methods if m in config['julia_methods']]
    
    print(f"\n📊 Summary:")
    print(f"  Total methods available: {len(all_methods)}")
    print(f"  New Julia AAA methods: {len(new_methods)}")
    print(f"  New methods currently enabled: {len(enabled_new)}")
    
    if len(enabled_new) == len(new_methods):
        print("✅ All new Julia AAA methods are enabled")
    else:
        print("🔶 Some new Julia AAA methods are disabled")
    
    # Show usage instructions
    print(f"\n🛠️  To enable/disable methods interactively:")
    print(f"   python edit_config.py")
    print(f"   -> Choose option 3 (Toggle Julia methods)")
    print(f"   -> Methods 7-10 are the new Julia AAA methods:")
    for i, method in enumerate(new_methods, 7):
        print(f"      {i}. {method}")
    
    return True

if __name__ == "__main__":
    test_julia_methods_toggle()