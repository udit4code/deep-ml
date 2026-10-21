from collections import deque

# This is an implicit approach, where we do not create the graph upfront. 
# We create the Graph on the fly. 
# Time Complexity : O(N^2) in the worst case, when each node may check its distance against all other nodes during the BFS traversal. 
# Space Complexity : O(N) to maintain the visited set and the BFS queue.
def graph_reachable(points, start, target, r):
    """
    Determines whether the target point is reachable from the start point 
    within a distance threshold r using BFS.
    """
    if start == target:
        return True
    
    n = len(points)
    if not (0 <= start < n and 0 <= target < n):
        return False
        
    r_sq = r * r  # Use squared distance to avoid math.sqrt() overhead
    visited = {start}
    queue = deque([start])
    
    while queue:
        curr = queue.popleft()
        
        # If we reached the target, return True
        if curr == target:
            return True
            
        x1, y1 = points[curr]
        
        # Check all unvisited nodes to see if they are directly connected
        for next_idx in range(n):
            if next_idx not in visited:
                x2, y2 = points[next_idx]
                dist_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
                
                if dist_sq <= r_sq:
                    visited.add(next_idx)
                    queue.append(next_idx)
                    
    return False
