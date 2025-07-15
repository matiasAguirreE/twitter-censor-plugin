"""
Device Management Utils for Twitter Censor Plugin
Optimized for Apple Silicon (MPS), NVIDIA (CUDA), and CPU fallback
"""

import torch
import platform

def get_optimal_device():
    """
    Returns the optimal device for training/inference with priority:
    1. MPS (Apple Silicon/Mac)
    2. CUDA (NVIDIA GPU) 
    3. CPU (fallback)
    
    Returns:
        torch.device: The optimal available device
        str: Device name/description
    """
    
    # Check for MPS (Apple Silicon - M1, M2, etc.)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            return torch.device("mps"), "Apple Silicon GPU (Metal Performance Shaders)"
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"NVIDIA GPU: {gpu_name}"
    
    # Fallback to CPU
    cpu_info = f"CPU: {platform.processor() or 'Unknown'}"
    return torch.device("cpu"), cpu_info

def print_device_info():
    """Print comprehensive device information"""
    print("🔧 Device Information:")
    print("=" * 50)
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Platform info
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # MPS info (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    else:
        print("MPS: Not available (requires PyTorch 1.12+ on macOS)")
    
    # CUDA info
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    # Selected device
    device, device_name = get_optimal_device()
    print(f"\n🎯 Selected device: {device}")
    print(f"📝 Description: {device_name}")
    print("=" * 50)
    
    return device, device_name

def optimize_for_device(model, device):
    """
    Apply device-specific optimizations
    
    Args:
        model: PyTorch model
        device: torch.device
    
    Returns:
        model: Optimized model
    """
    model = model.to(device)
    
    # Apple Silicon optimizations
    if device.type == "mps":
        print("🍎 Applying Apple Silicon (MPS) optimizations...")
        # MPS works best with certain data types and optimizations
        # Ensure model is in the right precision
        model = model.float()  # MPS works better with float32
        
    # CUDA optimizations  
    elif device.type == "cuda":
        print("🔥 Applying NVIDIA CUDA optimizations...")
        # Enable cuDNN benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        # Mixed precision can be beneficial for CUDA
        
    # CPU optimizations
    elif device.type == "cpu":
        print("💻 Applying CPU optimizations...")
        # Set number of threads for optimal CPU performance
        torch.set_num_threads(torch.get_num_threads())
        
    return model

def get_recommended_batch_size(device, base_batch_size=8):
    """
    Get recommended batch size based on device capabilities
    
    Args:
        device: torch.device
        base_batch_size: int, base batch size
        
    Returns:
        int: Recommended batch size
    """
    if device.type == "mps":
        # Apple Silicon has unified memory, can handle larger batches efficiently
        # but be conservative to avoid memory issues
        return min(base_batch_size * 2, 32)
    elif device.type == "cuda":
        # CUDA batch size depends on GPU memory, keep base
        return base_batch_size
    else:
        # CPU: smaller batches are usually better
        return max(base_batch_size // 2, 4)

def move_batch_to_device(batch, device):
    """
    Efficiently move batch data to device with proper error handling
    
    Args:
        batch: Dictionary containing batch data
        device: torch.device
        
    Returns:
        dict: Batch data moved to device
    """
    try:
        moved_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved_batch[key] = value.to(device, non_blocking=True)
            else:
                moved_batch[key] = value
        return moved_batch
    except Exception as e:
        print(f"⚠️  Error moving batch to device {device}: {e}")
        print("🔄 Falling back to synchronous transfer...")
        return {key: value.to(device) if torch.is_tensor(value) else value 
                for key, value in batch.items()} 