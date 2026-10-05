import io
import torch
import torch.nn as nn


# io.BytesIO is a built-in Python class that creates a file-like object directly in the system RAM. 

# Instead of reading and writing data to a slow physical storage drive (like an SSD or HDD), it reads and writes raw binary data (bytes) to an allocated block of volatile computer memory.

# Behavior: It exposes the exact same methods as a regular file (.write(), .read(), .seek()). 

# Why use it here?: Serializing model weights to disk involves slow hardware input/output (I/O) operations. Using io.BytesIO tricks PyTorch into running its serialization pipelines completely inside memory at lightning-fast speeds.

# This process of copy-ing weights from src to dst is needed during model checkpointing of long training pipelines, production deployment (where, once complete, saved weights are transferred to a production environment and exposed via a web server or mobile device to run live inference), transfer-learning and most importantly, in distributed training. During Distributed Training, in multi-GPU clusters, master nodes serialize and broadcase authorative model state parameters to worker nodes across local memory pipelines.  

# Why do we set weights-only flag to true?
# Because, PyTorch's torch.save used standard Python pickling under the hood. The pickle format is inherently insecure; loading an untrusted checkpoint file downloaded from the internet could execute malicious shell commands disguised inside the file. 

# By setting weights_only to True, we ensure security, as we parse only basic structural datatypes (tensors, dicts, lists, strings) from the buffer. 

def copy_weights(src: nn.Module, dst: nn.Module) -> nn.Module:
    # Step 1 : Create an in-memory byte buffer
    buffer = io.BytesIO()
    
    # Step 2 : Serialize and save the source model's state dictionary to the buffer
    torch.save(src.state_dict(), buffer)
    
    # Step 3 : Rewind the buffer's file pointer to the beginning for reading. 
    buffer.seek(0)
    
    # Step 4 : Load the weights from the buffer into the destination model.
    # Setting weights_only=True forces PyTorch's loading mechanism to strictly deserialize structural tensor data, completely blocking the execution of arbitrary Python code during the loading process.
    dst.load_state_dict(torch.load(buffer, weights_only=True))
    
    return dst
