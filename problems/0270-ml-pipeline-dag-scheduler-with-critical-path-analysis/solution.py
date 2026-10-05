import heapq as hq 
from copy import deepcopy

class CriticalPathAnalysis:
    
    def __init__(self, data: list = None, schema: dict = None):
        self.nodes = []
        self.edges = []
        self.DAG = dict()
        self.indegree = dict()
        self.data = data
        self.schema = schema
        
    def build_DAG(self):
        """
            Build the Directed Acyclic Graph (DAG) from the task data.
            Each task is a node, and dependencies are directed edges.
            Also computes the indegree for each node.
            
            Returns:
                None
        """
        for task in self.data:
            if task['id'] not in self.DAG.keys():
                self.DAG[task['id']] = [ ]
                self.indegree[task['id']] = 0
            for dependency in task['dependencies']:
                if dependency not in self.DAG.keys():
                    self.DAG[dependency] = [ ]
                    self.indegree[dependency] = 0  
              