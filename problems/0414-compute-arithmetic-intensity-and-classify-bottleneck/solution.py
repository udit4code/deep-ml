def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    # 1. Arithmetic Intensity (FLOPs per byte)
    arithmetic_intensity = flops / bytes_accessed
    
    # 2. Ridge Point (The balance point of the hardware)
    ridge_point = peak_performance / peak_bandwidth
    
    # 3. Achieved Performance (bounded by either memory or compute)
    memory_bound_limit = peak_bandwidth * arithmetic_intensity
    achieved_performance = min(peak_performance, memory_bound_limit)
    
    # 4. Bottleneck determination
    if arithmetic_intensity < ridge_point:
        bottleneck = "memory-bound"
    else:
        bottleneck = "compute-bound"
        
    # 5. Compute Utilization Percent
    utilization_percent = (achieved_performance / peak_performance) * 100
    
    return {
        "arithmetic_intensity": round(arithmetic_intensity, 4),
        "ridge_point": round(ridge_point, 4),
        "bottleneck": bottleneck,
        "achieved_performance": round(achieved_performance, 4),
        "utilization_percent": round(utilization_percent, 2)
    }