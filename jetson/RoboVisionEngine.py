import tensorrt as trt
from cuda.bindings import runtime as cudart
import numpy as np


class RoboVisionEngine:
    def __init__(self, engine_path: str):
        
        self.host_buffers = {}
        self.device_buffers = {}
        self.logger = trt.Logger()

        trt.init_libnvinfer_plugins(self.logger, "")
        with open(engine_path,'rb') as f:
            self.runtime = trt.Runtime(self.logger)
            engine_data = f.read()
            self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        
        self.context = self.engine.create_execution_context()
        
        self.tensor_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]



        for name in self.tensor_names:
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            self.host_buffers[name] = np.empty(shape,dtype=dtype)
            
            self.device_buffers[name] = check_cuda(cudart.cudaMalloc(size))
            self.context.set_tensor_address(name, self.device_buffers[name])

        self.stream = check_cuda(cudart.cudaStreamCreate())
    
    def infer(self, input_tensor: np.ndarray) -> dict:
        
        input_name = 'images' 

        np.copyto(self.host_buffers[input_name],input_tensor)

        check_cuda(
            cudart.cudaMemcpyAsync(
                self.device_buffers[input_name],
                self.host_buffers[input_name].ctypes.data,
                self.host_buffers[input_name].nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self.stream
            ))
        
        
        self.context.execute_async_v3(stream_handle=self.stream)

        outputs = {}

        for name in self.tensor_names:
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            
            check_cuda(cudart.cudaMemcpyAsync(
                self.host_buffers[name].ctypes.data,
                self.device_buffers[name],
                self.host_buffers[name].nbytes,
                  cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                  self.stream
            ))

            outputs[name] = self.host_buffers[name]
        
        check_cuda(cudart.cudaStreamSynchronize(self.stream))
        return outputs

    def __del__(self):
        
        for d_ptr in self.device_buffers.values():
            cudart.cudaFree(d_ptr)
        if hasattr(self,'stream'):
            cudart.cudaStreamDestroy(self.stream)


        
def check_cuda(err):
    """Wrap cuda-python calls that return (err, result) tuples."""
    if isinstance(err, tuple):
        err, *result = err
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return result[0] if len(result) == 1 else tuple(result)

