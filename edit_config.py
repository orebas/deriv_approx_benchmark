#!/usr/bin/env python3
"""
Simple configuration editor for the benchmark.
Allows you to quickly modify which methods, ODEs, and noise levels to test.
"""

import json
import sys
import argparse

# Import the new helper functions to dynamically get method names
from comprehensive_methods_library import get_base_method_names
from enhanced_gp_methods import get_enhanced_gp_method_names

def load_config():
    """Load the current configuration."""
    with open('benchmark_config.json', 'r') as f:
        return json.load(f)

def save_config(config):
    """Save the configuration."""
    with open('benchmark_config.json', 'w') as f:
        json.dump(config, f, indent=2)

def show_current_config(config):
    """Display current configuration."""
    print("\n📋 CURRENT CONFIGURATION")
    print("=" * 40)
    
    print(f"\n🧪 ODE Problems ({len(config['ode_problems'])}):")
    for i, ode in enumerate(config['ode_problems'], 1):
        print(f"  {i}. {ode}")
    
    print(f"\n🔊 Noise Levels ({len(config['noise_levels'])}):")
    print(f"  {config['noise_levels']}")
    
    print(f"\n📐 Julia Methods ({len(config['julia_methods'])}):")
    for i, method in enumerate(config['julia_methods'], 1):
        print(f"  {i}. {method}")
    
    base_count = len(config['python_methods']['base_methods'])
    gp_count = len(config['python_methods']['enhanced_gp_methods'])
    print(f"\n🐍 Python Methods ({base_count + gp_count}):")
    
    if base_count > 0:
        if base_count <= 5:
            print(f"  Base methods ({base_count}): {', '.join(config['python_methods']['base_methods'])}")
        else:
            first_few = ', '.join(config['python_methods']['base_methods'][:3])
            print(f"  Base methods ({base_count}): {first_few}... (+{base_count-3} more)")
    else:
        print(f"  Base methods: None selected")
    
    if gp_count > 0:
        print(f"  Enhanced GP ({gp_count}): {', '.join(config['python_methods']['enhanced_gp_methods'])}")
    else:
        print(f"  Enhanced GP: None selected")
    
    print(f"\n⚙️  Data Config:")
    print(f"  Data size: {config['data_config']['data_size']}")
    print(f"  Derivative orders: {config['data_config']['derivative_orders']}")

def list_all_available_methods():
    """Prints a clean, comprehensive list of all available methods."""
    print("📋 FULL LIST OF AVAILABLE METHODS")
    print("="*40)

    print("\n🐍 Python Methods")
    print("-" * 20)
    
    print("  📦 Base Methods:")
    base_methods = get_base_method_names()
    for method in sorted(base_methods):
        print(f"    - {method}")
    
    print("\n  🧠 Enhanced GP Methods:")
    gp_methods = get_enhanced_gp_method_names()
    for method in sorted(gp_methods):
        print(f"    - {method}")

    print("\n\n📐 Julia Methods")
    print("-" * 20)
    # NOTE: This list is currently hard-coded as it's not trivial to
    # read from the Julia source dynamically. If Julia methods change,
    # this list must be updated manually.
    julia_methods = ["GPR", "AAA", "AAA_lowpres", "LOESS", "BSpline5"]
    for method in sorted(julia_methods):
        print(f"    - {method}")
    print("\nNOTE: To use these methods, add their exact names to the lists")
    print("      in 'benchmark_config.json'.")

def quick_edit_menu(config):
    """Quick edit menu for common changes."""
    while True:
        print("\n🛠️  QUICK EDIT MENU")
        print("=" * 30)
        print("1. Toggle ODE problems")
        print("2. Edit noise levels")
        print("3. Toggle Julia methods")
        print("4. Toggle Python methods")
        print("5. Edit derivative orders")
        print("6. Quick presets")
        print("7. Show current config")
        print("8. Save and exit")
        print("9. Exit without saving")
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == '1':
            toggle_ode_problems(config)
        elif choice == '2':
            edit_noise_levels(config)
        elif choice == '3':
            toggle_julia_methods(config)
        elif choice == '4':
            toggle_python_methods(config)
        elif choice == '5':
            edit_derivative_orders(config)
        elif choice == '6':
            quick_presets(config)
        elif choice == '7':
            show_current_config(config)
        elif choice == '8':
            save_config(config)
            print("✅ Configuration saved!")
            break
        elif choice == '9':
            print("❌ Exiting without saving")
            break
        else:
            print("Invalid choice. Please try again.")

def toggle_ode_problems(config):
    """Toggle ODE problems on/off."""
    all_odes = ["lv_periodic", "vanderpol", "brusselator", "fitzhugh_nagumo", "seir"]
    
    print("\n🧪 Toggle ODE Problems:")
    for i, ode in enumerate(all_odes, 1):
        status = "✅" if ode in config['ode_problems'] else "❌"
        print(f"  {i}. {status} {ode}")
    
    print("\nEnter numbers to toggle (e.g., '1 3 5' or 'all' or 'none'):")
    selection = input().strip().lower()
    
    if selection == 'all':
        config['ode_problems'] = all_odes.copy()
    elif selection == 'none':
        config['ode_problems'] = []
    else:
        try:
            indices = [int(x) - 1 for x in selection.split()]
            for idx in indices:
                if 0 <= idx < len(all_odes):
                    ode = all_odes[idx]
                    if ode in config['ode_problems']:
                        config['ode_problems'].remove(ode)
                    else:
                        config['ode_problems'].append(ode)
        except ValueError:
            print("Invalid input. Use numbers separated by spaces.")
    
    print(f"✅ Updated ODE problems: {config['ode_problems']}")

def edit_noise_levels(config):
    """Edit noise levels."""
    print("\n🔊 Current noise levels:")
    print(f"  {config['noise_levels']}")
    
    print("\nPresets:")
    print("1. Minimal: [0.0, 1e-3, 1e-2]")
    print("2. Standard: [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]") 
    print("3. Full: [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]")
    print("4. Custom")
    
    choice = input("Choose preset (1-4): ").strip()
    
    if choice == '1':
        config['noise_levels'] = [0.0, 1e-3, 1e-2]
    elif choice == '2':
        config['noise_levels'] = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    elif choice == '3':
        config['noise_levels'] = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    elif choice == '4':
        print("Enter noise levels as space-separated numbers (e.g., '0.0 0.001 0.01'):")
        try:
            levels_str = input().strip()
            config['noise_levels'] = [float(x) for x in levels_str.split()]
        except ValueError:
            print("Invalid input. Keeping current values.")
    
    print(f"✅ Updated noise levels: {config['noise_levels']}")

def edit_derivative_orders(config):
    """Edit the maximum derivative order to compute."""
    print(f"\n📊 Current derivative orders: {config['data_config']['derivative_orders']}")
    print("\nNote: Higher orders require more computation but provide more detailed analysis.")
    print("Recommended ranges:")
    print("  • Quick testing: 1-2")
    print("  • Standard analysis: 3-4") 
    print("  • Comprehensive analysis: 5-7")
    print("  • Maximum supported: 7")
    
    try:
        new_orders = input("\nEnter new derivative orders (1-7): ").strip()
        new_orders_int = int(new_orders)
        
        if 1 <= new_orders_int <= 7:
            config['data_config']['derivative_orders'] = new_orders_int
            print(f"✅ Updated derivative orders: {new_orders_int}")
        else:
            print("❌ Invalid range. Must be between 1 and 7.")
    except ValueError:
        print("❌ Invalid input. Must be a number between 1 and 7.")

def toggle_julia_methods(config):
    """Toggle Julia methods on/off."""
    all_methods = ["GPR", "AAA", "AAA_lowpres", "LOESS", "BSpline5", "TVDiff", 
                   "JuliaAAALS", "JuliaAAAFullOpt", "JuliaAAATwoStage", "JuliaAAASmoothBary"]
    
    print("\n📐 Toggle Julia Methods:")
    for i, method in enumerate(all_methods, 1):
        status = "✅" if method in config['julia_methods'] else "❌"
        print(f"  {i}. {status} {method}")
    
    print("\nEnter numbers to toggle (e.g., '1 3 5' or 'all' or 'none'):")
    selection = input().strip().lower()
    
    if selection == 'all':
        config['julia_methods'] = all_methods.copy()
    elif selection == 'none':
        config['julia_methods'] = []
    else:
        try:
            indices = [int(x) - 1 for x in selection.split()]
            for idx in indices:
                if 0 <= idx < len(all_methods):
                    method = all_methods[idx]
                    if method in config['julia_methods']:
                        config['julia_methods'].remove(method)
                    else:
                        config['julia_methods'].append(method)
        except ValueError:
            print("Invalid input. Use numbers separated by spaces.")
    
    print(f"✅ Updated Julia methods: {config['julia_methods']}")

def toggle_python_methods(config):
    """Toggle Python methods on/off."""
    print("\n🐍 Toggle Python Methods:")
    print("\n📦 Base Methods:")
    
    base_methods = config['python_methods']['base_methods']
    # Dynamically get the list of all available base methods
    all_base = sorted(get_base_method_names())
    
    for i, method in enumerate(all_base, 1):
        status = "✅" if method in base_methods else "❌"
        print(f"  {i:2}. {status} {method}")
    
    print("\n🧠 Enhanced GP Methods:")
    gp_methods = config['python_methods']['enhanced_gp_methods']
    # Dynamically get the list of all available GP methods
    all_gp = sorted(get_enhanced_gp_method_names())
    
    for i, method in enumerate(all_gp, len(all_base) + 1):
        status = "✅" if method in gp_methods else "❌"
        print(f"  {i:2}. {status} {method}")
    
    print("\nOptions:")
    print("• Enter numbers to toggle (e.g., '1 3 5')")
    print("• 'base-all' / 'base-none' for all/no base methods")
    print("• 'gp-all' / 'gp-none' for all/no GP methods")
    print("• 'all' / 'none' for everything")
    
    selection = input("Selection: ").strip().lower()
    
    if selection == 'all':
        config['python_methods']['base_methods'] = all_base.copy()
        config['python_methods']['enhanced_gp_methods'] = all_gp.copy()
    elif selection == 'none':
        config['python_methods']['base_methods'] = []
        config['python_methods']['enhanced_gp_methods'] = []
    elif selection == 'base-all':
        config['python_methods']['base_methods'] = all_base.copy()
    elif selection == 'base-none':
        config['python_methods']['base_methods'] = []
    elif selection == 'gp-all':
        config['python_methods']['enhanced_gp_methods'] = all_gp.copy()
    elif selection == 'gp-none':
        config['python_methods']['enhanced_gp_methods'] = []
    else:
        try:
            indices = [int(x) - 1 for x in selection.split()]
            for idx in indices:
                if 0 <= idx < len(all_base):
                    # Base method
                    method = all_base[idx]
                    if method in base_methods:
                        base_methods.remove(method)
                    else:
                        base_methods.append(method)
                elif len(all_base) <= idx < len(all_base) + len(all_gp):
                    # GP method
                    gp_idx = idx - len(all_base)
                    method = all_gp[gp_idx]
                    if method in gp_methods:
                        gp_methods.remove(method)
                    else:
                        gp_methods.append(method)
        except ValueError:
            print("Invalid input. Use numbers separated by spaces.")
            return
    
    total_methods = len(base_methods) + len(gp_methods)
    print(f"✅ Updated Python methods: {total_methods} total ({len(base_methods)} base + {len(gp_methods)} GP)")

def quick_presets(config):
    """Apply quick presets for common testing scenarios."""
    print("\n⚡ Quick Presets:")
    print("1. Fast test: 1 ODE, minimal noise, few methods")
    print("2. Method comparison: 1 ODE, standard noise, all methods")
    print("3. Noise robustness: all ODEs, full noise range, core methods")
    print("4. Full benchmark: all ODEs, full noise, all methods")
    
    choice = input("Choose preset (1-4): ").strip()
    
    if choice == '1':
        config['ode_problems'] = ['lv_periodic']
        config['noise_levels'] = [0.0, 1e-3, 1e-2]
        config['julia_methods'] = ['GPR', 'AAA']
        config['python_methods']['base_methods'] = ['CubicSpline', 'GP_RBF', 'SavitzkyGolay']
        config['python_methods']['enhanced_gp_methods'] = []
        config['data_config']['data_size'] = 101
        config['data_config']['derivative_orders'] = 2
    elif choice == '2':
        config['ode_problems'] = ['lv_periodic']
        config['noise_levels'] = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
        config['julia_methods'] = ["GPR", "AAA", "AAA_lowpres", "LOESS", "BSpline5", "TVDiff"]
        # All Python methods for comparison
        config['python_methods']['base_methods'] = [
            "CubicSpline", "SmoothingSpline", "RBF_ThinPlate", "RBF_Multiquadric",
            "GP_RBF", "GP_Matern", "Chebyshev", "Polynomial", "SavitzkyGolay", 
            "Butterworth", "RandomForest", "SVR", "Fourier", "FiniteDiff", 
            "AAA_LS", "AAA_FullOpt", "KalmanGrad"
        ]
        config['python_methods']['enhanced_gp_methods'] = ["GP_RBF_Iso", "GP_Matern_1.5", "GP_Matern_2.5", "GP_Periodic"]
        config['data_config']['data_size'] = 201
        config['data_config']['derivative_orders'] = 4
    elif choice == '3':
        config['ode_problems'] = ["lv_periodic", "vanderpol", "brusselator", "fitzhugh_nagumo", "seir"]
        config['noise_levels'] = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        config['julia_methods'] = ['GPR', 'AAA', 'LOESS']
        # Core robust methods
        config['python_methods']['base_methods'] = ['CubicSpline', 'SmoothingSpline', 'GP_RBF', 'GP_Matern', 'SavitzkyGolay']
        config['python_methods']['enhanced_gp_methods'] = ['GP_Matern_2.5']
        config['data_config']['data_size'] = 201
        config['data_config']['derivative_orders'] = 3
    elif choice == '4':
        config['ode_problems'] = ["lv_periodic", "vanderpol", "brusselator", "fitzhugh_nagumo", "seir"]
        config['noise_levels'] = [0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        config['julia_methods'] = ["GPR", "AAA", "AAA_lowpres", "LOESS", "BSpline5", "TVDiff"]
        # All methods
        config['python_methods']['base_methods'] = [
            "CubicSpline", "SmoothingSpline", "RBF_ThinPlate", "RBF_Multiquadric",
            "GP_RBF", "GP_Matern", "Chebyshev", "Polynomial", "SavitzkyGolay", 
            "Butterworth", "RandomForest", "SVR", "Fourier", "FiniteDiff", 
            "AAA_LS", "AAA_FullOpt", "KalmanGrad"
        ]
        config['python_methods']['enhanced_gp_methods'] = ["GP_RBF_Iso", "GP_Matern_1.5", "GP_Matern_2.5", "GP_Periodic"]
        config['data_config']['data_size'] = 201
        config['data_config']['derivative_orders'] = 4
    
    print(f"✅ Applied preset {choice}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="A CLI tool to view and edit 'benchmark_config.json'.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--list-methods',
        action='store_true',
        help='List all available Julia and Python methods and exit.'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show the current configuration and exit.'
    )
    args = parser.parse_args()

    if args.list_methods:
        list_all_available_methods()
        return 0

    try:
        config = load_config()
    except FileNotFoundError:
        print("❌ benchmark_config.json not found!")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in config file: {e}")
        return 1
    
    if args.show:
        show_current_config(config)
        return 0
    
    show_current_config(config)
    quick_edit_menu(config)
    return 0

if __name__ == "__main__":
    sys.exit(main())