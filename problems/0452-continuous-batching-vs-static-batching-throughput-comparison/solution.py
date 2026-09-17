def get_continuous_batching_wall_time(requests: list, max_batch_size : int, time_per_step: float) -> float: 
    current_batch = [ ]
    continuous_steps = 0
    # We create a new copy from requests 
    queue = list(requests)

    while queue or current_batch:
        # Step 1 : Fill available slots up to max_batch_size,
        while len(current_batch) < max_batch_size:
            if queue:
                popped_request = queue.pop(0)
                current_batch.append(popped_request)
            else:
                break 
        # Step 2 : If there are active requests, run exactly 1 step 
        if current_batch:
            continuous_steps += 1
            # Step 2.1 : Decrement 1 step for every request in the current batch 
            current_batch = [step - 1 for step in current_batch]
            # Step 2.2 : Evict requests that have finished (with 0 steps)
            current_batch = [step for step in current_batch if step > 0]
        
    return continuous_steps * time_per_step


def get_static_batching_wall_time(requests: list, max_batch_size : int, time_per_step: float) -> float: 
    static_steps = 0
    for i in range(0, len(requests), max_batch_size):
        batch_requests = requests[i : i + max_batch_size]
        max_request_in_batch = max(batch_requests)
        static_steps += max_request_in_batch
    return static_steps * time_per_step


def compare_batching(requests: list, max_batch_size: int, time_per_step: float) -> dict:
    """
    Simulate and compare static vs continuous batching strategies for LLM serving.
    
    Args:
        requests: List of integers, each representing decode steps for a request
        max_batch_size: Maximum concurrent requests the GPU can handle
        time_per_step: Time in milliseconds for one decode step
    
    Returns:
        Dictionary with performance metrics for both strategies
    """
    sum_of_req_steps = (sum(requests)) * time_per_step
    req_count = len(requests)
    # Step 1 : Get static batching wall time 
    static_total_time = get_static_batching_wall_time(requests, max_batch_size, time_per_step)
    static_throughput = (req_count / static_total_time) * 1000 # because static_total_time is in milliseconds
    static_gpu_capacity = max_batch_size * static_total_time 
    static_gpu_utilization = sum_of_req_steps / static_gpu_capacity
    # Step 2 : Get continuous batching wall time 
    continuous_total_time = get_continuous_batching_wall_time(requests, max_batch_size, time_per_step)
    continuous_throughput = (req_count / continuous_total_time) * 1000 # because continuous_total_time is in milliseconds
    continuous_gpu_capacity = max_batch_size * continuous_total_time
    continuous_gpu_utilization = sum_of_req_steps / continuous_gpu_capacity 

    speedup = static_total_time / continuous_total_time
    return {
        "static_total_time" : round(static_total_time, 4), 
        "continuous_total_time" : round(continuous_total_time, 4),
        "static_throughput" : round(static_throughput, 4),
        "continuous_throughput" : round(continuous_throughput, 4),
        "static_gpu_utilization" : round(static_gpu_utilization, 4),
        "continuous_gpu_utilization" : round(continuous_gpu_utilization, 4),
        "speedup" : round(speedup, 4),
    }