#!/usr/bin/env python3
"""
Verify that edit_config.py now shows the new Julia AAA methods correctly.
"""

import subprocess
import sys

def test_edit_config_shows_julia_methods():
    """Test that edit_config.py --show includes the new Julia methods."""
    print("🧪 Verifying edit_config.py shows new Julia AAA methods")
    print("=" * 60)
    
    try:
        # Run edit_config.py --show
        result = subprocess.run(['python', 'edit_config.py', '--show'], 
                               capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print(f"❌ edit_config.py failed: {result.stderr}")
            return False
        
        output = result.stdout
        
        # Check for each new Julia method
        new_julia_methods = ["JuliaAAALS", "JuliaAAAFullOpt", "JuliaAAATwoStage", "JuliaAAASmoothBary"]
        
        all_found = True
        for method in new_julia_methods:
            if method in output:
                print(f"  ✅ {method} - Found in edit_config.py output")
            else:
                print(f"  ❌ {method} - NOT found in edit_config.py output")
                all_found = False
        
        # Check for the Julia Methods section
        if "📐 Julia Methods" in output:
            print(f"  ✅ Julia Methods section found")
            
            # Count total Julia methods shown
            julia_section = output.split("📐 Julia Methods")[1].split("🐍 Python Methods")[0]
            method_count = julia_section.count(". ")
            print(f"  📊 Total Julia methods shown: {method_count}")
            
        else:
            print(f"  ❌ Julia Methods section not found")
            all_found = False
        
        return all_found
        
    except subprocess.TimeoutExpired:
        print("❌ edit_config.py timed out")
        return False
    except Exception as e:
        print(f"❌ Error running edit_config.py: {e}")
        return False

def main():
    print("🔧 Edit Config Fix Verification")
    print("=" * 80)
    
    success = test_edit_config_shows_julia_methods()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ edit_config.py now shows all new Julia AAA methods")
        print(f"✅ Users can now enable/disable them interactively")
        print(f"\n📋 Usage:")
        print(f"   python edit_config.py")
        print(f"   -> Choose option 3 (Toggle Julia methods)")
        print(f"   -> Methods 7-10 are the new Julia AAA methods")
        return 0
    else:
        print(f"\n❌ FAILED!")
        print(f"   Some new Julia methods are not showing up in edit_config.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())