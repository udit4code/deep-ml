class BatchInferenceEngine:

    def __init__(self, max_batch_size: int, max_wait_time: float):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.batches_so_far = []
        self.current_batch = []

    def add_request(self, request: dict):
        if len(self.current_batch) == self.max_batch_size:
            # CASE 1 : When we have reached the max_batch_size
            # In this case, we simply add the current_batch to batches_so_far
            self.batches_so_far.append(self.current_batch[::])
            # initialise the current_batch as an empty batch for the next batch
            self.current_batch = []
        else:
            # CASE 2 : When the incoming request arrives beyond 1st request of current_batch + max_wait_time
            if len(self.current_batch) > 0:
                if (
                    self.current_batch[0]["timestamp"] + self.max_wait_time
                    < request["timestamp"]
                ):
   