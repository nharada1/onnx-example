import torch
import onnx
from onnxconverter_common import float16
import onnxruntime as ort
import numpy as np
import time

input_shape = (1, 3, 224, 224)

def benchmark_onnx():
    # Load the ONNX model
    model_path = 'model.onnx'
    onnx_model = onnx.load(model_path)
    og = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    og.eval()

    # Create an ONNX runtime session
    session = ort.InferenceSession(model_path)

    # Run the model N times and calculate the average time
    num_runs = 50
    total_time, total_time_og, total_diff = 0, 0, 0
    for _ in range(num_runs):
        # Generate a random input image of size 224x224
        input_data = np.random.randn(*input_shape).astype(np.float32)

        start_time = time.time()
        outputs = session.run(None, {'input': input_data})
        end_time = time.time()

        # Calculate the elapsed time
        with torch.no_grad():
            elapsed_time = end_time - start_time
            total_time += elapsed_time

            start_time = time.time()
            output_og = og(torch.Tensor(input_data))
            end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        total_time_og += elapsed_time

        # Calc diffs
        diff = (output_og - outputs[0]).abs().sum()
        total_diff += diff

    # Calculate the average time
    average_time = total_time / num_runs
    average_time_og = total_time_og / num_runs

    print(f"Benchmarking complete over {num_runs} runs")
    print(f"Average inference time: {average_time * 1000} ms")
    print(f"Average inference time original: {average_time_og * 1000} ms")
    print(f"Average sum numerical difference: {total_diff / num_runs}")


def torchscript():
    dino = torch.hub.load('dinov2', 'dinov2_vits14', source='local')
    dino.eval()

    example = torch.rand(*input_shape)
    traced_script_module = torch.jit.trace(dino, example)

def run_onnx():
    class xyz_model(torch.nn.Module): 
        def __init__(self, model): 
            super().__init__() 
            self.model = model  
        
        def forward(self, tensor): 
            ff = self.model(tensor) 
            return ff 
        
    dino = torch.hub.load('dinov2', 'dinov2_vits14', source='local')
    mm = xyz_model(dino).to('cpu') 

    mm.eval()

    dynamic_axes = {'input': {0: 'batch', 2: 'height', 3: 'width'}}
    input_data = torch.randn(*input_shape).to('cpu') 
    output = mm(input_data) 
    
    torch.onnx.export(mm, input_data, 'model.onnx', input_names = ['input'], export_params=True, do_constant_folding=True, dynamic_axes=dynamic_axes)


if __name__ == '__main__':
    run_onnx()
    benchmark_onnx()