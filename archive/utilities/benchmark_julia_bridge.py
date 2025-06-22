#!/usr/bin/env python3
"""
Benchmark the Python-Julia bridge latency for AAA barycentric evaluation.
This will help determine if the bridge overhead is acceptable.
"""

import numpy as np
import time
import subprocess
import sys

def test_julia_bridge_latency():
    """Test the latency of calling Julia AAA functions from Python"""
    print("Benchmarking Python-Julia Bridge Latency")
    print("="*60)
    
    # Check if Julia is available
    try:
        result = subprocess.run(['julia', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Julia found: {result.stdout.strip()}")
        else:
            print("❌ Julia not found in PATH")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Julia not available or timed out")
        return False
    
    # Try to import juliacall or PyJulia
    julia_available = False
    bridge_method = None
    
    try:
        from juliacall import Main as jl
        julia_available = True
        bridge_method = "juliacall"
        print(f"✅ Using juliacall for Python-Julia bridge")
    except ImportError:
        try:
            import julia
            from julia import Main as jl
            julia_available = True
            bridge_method = "PyJulia"
            print(f"✅ Using PyJulia for Python-Julia bridge")
        except ImportError:
            print("❌ Neither juliacall nor PyJulia available")
            print("   Install with: pip install juliacall  OR  pip install julia")
            return False
    
    if not julia_available:
        return False
    
    # Test Julia evaluation
    print(f"\n📊 Testing {bridge_method} latency...")
    
    # Simple test: basic arithmetic
    print("\n1. Basic Julia arithmetic test:")
    times = []
    for i in range(10):
        start = time.perf_counter()
        result = jl.eval("2 + 2")
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    print(f"   Mean latency: {np.mean(times):.3f} ms")
    print(f"   Std dev: {np.std(times):.3f} ms")
    print(f"   Min/Max: {np.min(times):.3f} / {np.max(times):.3f} ms")
    
    # Test array passing
    print("\n2. Array passing test:")
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
    
    times = []
    for i in range(10):
        start = time.perf_counter()
        # Test passing numpy arrays and getting result back
        result = jl.eval("x -> sum(x)")(x)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    print(f"   Mean latency: {np.mean(times):.3f} ms")
    print(f"   Std dev: {np.std(times):.3f} ms")
    print(f"   Min/Max: {np.min(times):.3f} / {np.max(times):.3f} ms")
    
    # Test if we can call existing barycentric function
    print("\n3. Barycentric evaluation test:")
    try:
        # Load the existing Julia module if available
        jl.eval('include("src/approximation_methods.jl")')
        jl.eval('using .DerivativeApproximationBenchmark')
        
        # Test data for barycentric evaluation
        t = np.linspace(0, 2*np.pi, 10)
        y = np.sin(t)
        
        times = []
        success_count = 0
        for i in range(5):
            try:
                start = time.perf_counter()
                
                # This is a placeholder - we'd need to call the actual Julia AAA function
                # For now, just test the overhead of calling with real data
                result = jl.eval("t -> length(t)")(t)
                
                end = time.perf_counter()
                times.append((end - start) * 1000)
                success_count += 1
            except Exception as e:
                print(f"   Error in iteration {i}: {e}")
        
        if success_count > 0:
            print(f"   Successful calls: {success_count}/5")
            print(f"   Mean latency: {np.mean(times):.3f} ms")
            print(f"   Std dev: {np.std(times):.3f} ms")
        else:
            print("   ❌ All Julia function calls failed")
            
    except Exception as e:
        print(f"   ⚠️  Could not load Julia module: {e}")
        print("   This is expected if Julia AAA code isn't set up yet")
    
    # Analysis and recommendations
    print(f"\n📈 Analysis and Recommendations:")
    basic_latency = np.mean(times) if times else float('inf')
    
    if basic_latency < 1.0:
        print("   🎉 EXCELLENT: <1ms latency - Direct integration viable")
        print("   ✅ Recommendation: Use direct juliacall/PyJulia in main Python process")
    elif basic_latency < 10.0:
        print("   ✅ GOOD: 1-10ms latency - Acceptable for most use cases")
        print("   ✅ Recommendation: Direct integration fine, consider batching for high-volume")
    elif basic_latency < 100.0:
        print("   ⚠️  MODERATE: 10-100ms latency - May need optimization")
        print("   💡 Recommendation: Consider batching calls or microservice architecture")
    else:
        print("   ❌ HIGH: >100ms latency - Needs optimization or different approach")
        print("   💡 Recommendation: Investigate microservice with connection pooling")
    
    return True

def test_deployment_strategies():
    """Outline deployment strategies based on latency results"""
    print(f"\n🚀 Deployment Strategy Options:")
    print(f"="*60)
    
    print("1. DIRECT INTEGRATION (Recommended if <10ms latency):")
    print("   - Julia code runs in same process as Python")
    print("   - Use juliacall or PyJulia")
    print("   - Simplest deployment - single container")
    print("   - Best for development and moderate scale")
    
    print("\n2. MICROSERVICE ARCHITECTURE (If >10ms latency or scaling needs):")
    print("   - Julia service in separate container")
    print("   - gRPC or HTTP API interface")
    print("   - Independent scaling and deployment")
    print("   - Better for high-volume production")
    
    print("\n3. BATCH PROCESSING (If high latency but batch-friendly workload):")
    print("   - Process multiple evaluations in single Julia call")
    print("   - Amortize startup/communication costs")
    print("   - Good for offline analysis or batch inference")

if __name__ == "__main__":
    print("🔬 AAA Algorithm Architecture: Julia Bridge Latency Benchmark")
    print("=" * 80)
    
    success = test_julia_bridge_latency()
    
    if success:
        test_deployment_strategies()
        
        print(f"\n🎯 Next Steps:")
        print(f"   1. Run this benchmark on your target deployment environment")
        print(f"   2. If latency is acceptable, proceed with Julia implementation")
        print(f"   3. Create containerized deployment with both Python and Julia")
        print(f"   4. Migrate AAA methods one by one to validate approach")
        
    else:
        print(f"\n❌ Bridge setup required before proceeding")
        print(f"   Install: pip install juliacall")
        print(f"   Then re-run this benchmark")