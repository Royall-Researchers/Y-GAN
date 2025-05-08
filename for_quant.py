import os
import numpy as np
import onnx
import pyxir
from pyxir.contrib.tools import quantize, build
from pyxir.target_registry import TargetRegistry

# Define paths
ONNX_MODEL_PATH = 'YGAN_generator_new.onnx'
QUANTIZED_MODEL_PATH = 'YGAN_generator_new_quantized.onnx'
OUTPUT_DIR = './compiled_model'

# Define input and output node names
INPUTS = ['input.1', 'input.13']
OUTPUT = '37'

# Register the ZCU104 target
TARGET = 'DPUCZDX8G-zcu104'
target_registry = TargetRegistry()

if not target_registry.is_registered(TARGET):
    target_registry.register(TARGET, 'pyxir')
print(f'Target registered: {TARGET}')

# Load ONNX model to verify structure
print('Loading ONNX model...')
model = onnx.load(ONNX_MODEL_PATH)
print('Model inputs:', [node.name for node in model.graph.input])
print('Model outputs:', [node.name for node in model.graph.output])

# Dummy calibration data (Replace with real calibration data)
print('Generating dummy calibration data...')
calib_data = {
    'input.1': [np.random.randn(1, 3, 224, 224).astype(np.float32)],
    'input.13': [np.random.randn(1, 3, 224, 224).astype(np.float32)]
}

# Quantize the model
print('Quantizing model...')
quantized_model = quantize(
    model_path=ONNX_MODEL_PATH,
    inputs=INPUTS,
    output=OUTPUT,
    calib_data=calib_data,
    target=TARGET,
    quant_mode='int8'
)

# Save the quantized model
onnx.save(quantized_model, QUANTIZED_MODEL_PATH)
print(f'Quantized model saved at {QUANTIZED_MODEL_PATH}')

# Compile the model to .xmodel
print('Compiling model to .xmodel...')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

build(
    model_path=QUANTIZED_MODEL_PATH,
    target=TARGET,
    output_dir=OUTPUT_DIR
)

print(f'Model compiled to {OUTPUT_DIR}')
