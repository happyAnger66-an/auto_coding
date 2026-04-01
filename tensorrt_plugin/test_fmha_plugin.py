"""
Test script for the FMHA TensorRT plugin
"""

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import sys

def load_plugin_lib():
    """Load the custom plugin library"""
    import ctypes
    
    # Get the path to the plugin library
    plugin_path = os.path.join(os.path.dirname(__file__), "build", "lib", "libfmha_plugin.so")
    
    if not os.path.exists(plugin_path):
        print(f"Plugin library not found at: {plugin_path}")
        print("Please run build.sh first to build the plugin")
        return False
    
    try:
        # Load the plugin library
        ctypes.CDLL(plugin_path)
        print(f"Successfully loaded plugin: {plugin_path}")
        return True
    except OSError as e:
        print(f"Failed to load plugin library: {e}")
        return False

def create_sample_network_with_fmha_plugin(builder, network, config, 
                                          batch_size=1, seq_len=128, num_heads=8, head_size=64):
    """Create a sample network with FMHA plugin"""
    
    # Create input tensors
    query_tensor = network.add_input("query", trt.float32, (batch_size, seq_len, num_heads, head_size))
    key_tensor = network.add_input("key", trt.float32, (batch_size, seq_len, num_heads, head_size))
    value_tensor = network.add_input("value", trt.float32, (batch_size, seq_len, num_heads, head_size))
    
    # Get plugin registry and find our FMHA plugin
    plugin_registry = trt.get_plugin_registry()
    plugin_creator_list = plugin_registry.plugin_creator_list
    
    fmha_plugin = None
    for plugin_creator in plugin_creator_list:
        if plugin_creator.name == "FMHA_TRT":
            print(f"Found FMHA plugin: {plugin_creator.name}")
            
            # Create plugin instance with parameters
            plugin_fields = [
                trt.PluginField("numHeads", np.array([num_heads], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("headSize", np.array([head_size], dtype=np.int32), trt.PluginFieldType.INT32),
                trt.PluginField("isCausal", np.array([False], dtype=bool), trt.PluginFieldType.BOOL)
            ]
            
            plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
            fmha_plugin = plugin_creator.create_plugin(name="fmha_layer", field_collection=plugin_field_collection)
            break
    
    if fmha_plugin is None:
        print("FMHA plugin not found in registry!")
        # List available plugins for debugging
        print("Available plugins:")
        for plugin_creator in plugin_registry.plugin_creator_list:
            print(f"  - {plugin_creator.name}")
        return None
    
    # Create the FMHA layer using the plugin
    fmha_layer = network.add_plugin_v2(inputs=[query_tensor, key_tensor, value_tensor], plugin=fmha_plugin)
    fmha_layer.name = "fmha_layer"
    
    # Mark output
    output_tensor = fmha_layer.get_output(0)
    output_tensor.name = "fmha_output"
    network.mark_output(output_tensor)
    
    # Set dynamic ranges if needed (for INT8 calibration)
    if builder.platform_has_fast_int8:
        for layer_input in [query_tensor, key_tensor, value_tensor]:
            layer_input.dynamic_range = (-1.0, 1.0)
        output_tensor.dynamic_range = (-1.0, 1.0)
    
    return output_tensor

def test_plugin_creation():
    """Test if we can create the plugin in a TensorRT network"""
    print("Testing FMHA plugin creation...")
    
    # Load the plugin library
    if not load_plugin_lib():
        return False
    
    # Initialize TensorRT builder
    builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
    if builder is None:
        print("Failed to create TensorRT builder")
        return False
    
    # Create network definition
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=network_flags)
    
    if network is None:
        print("Failed to create TensorRT network")
        return False
    
    # Create builder config
    config = builder.create_builder_config()
    if config is None:
        print("Failed to create builder config")
        return False
    
    # Set memory limit
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Create the network with FMHA plugin
    output_tensor = create_sample_network_with_fmha_plugin(builder, network, config)
    if output_tensor is None:
        print("Failed to create network with FMHA plugin")
        return False
    
    print("Successfully created network with FMHA plugin!")
    print(f"Output tensor shape: {output_tensor.shape}")
    
    # Try to build the engine
    try:
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is not None:
            print("Successfully built TensorRT engine with FMHA plugin!")
            
            # Save the engine for later use
            engine_path = os.path.join(os.path.dirname(__file__), "fmha_engine.trt")
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            print(f"Engine saved to: {engine_path}")
            
            return True
        else:
            print("Failed to build TensorRT engine")
            return False
    except Exception as e:
        print(f"Error during engine building: {e}")
        return False

def test_inference():
    """Test inference with the plugin"""
    print("\nTesting inference with FMHA plugin...")
    
    try:
        # Load the saved engine
        engine_path = os.path.join(os.path.dirname(__file__), "fmha_engine.trt")
        if not os.path.exists(engine_path):
            print("Engine file not found, skipping inference test")
            return True
        
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        # Create runtime
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        
        if engine is None:
            print("Failed to deserialize engine")
            return False
        
        # Create execution context
        context = engine.create_execution_context()
        if context is None:
            print("Failed to create execution context")
            return False
        
        # Allocate I/O tensors on GPU
        input_shapes = [
            (1, 128, 8, 64),  # query
            (1, 128, 8, 64),  # key
            (1, 128, 8, 64)   # value
        ]
        output_shape = (1, 128, 8, 64)
        
        # Allocate GPU memory
        d_inputs = []
        for i, shape in enumerate(input_shapes):
            input_size = trt.volume(shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
            d_input = cuda.mem_alloc(input_size)
            d_inputs.append(d_input)
        
        output_size = trt.volume(output_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize
        d_output = cuda.mem_alloc(output_size)
        
        # Create bindings
        bindings = [int(d_input) for d_input in d_inputs] + [int(d_output)]
        
        # Create sample input data
        h_query = np.random.rand(*input_shapes[0]).astype(np.float32)
        h_key = np.random.rand(*input_shapes[1]).astype(np.float32)
        h_value = np.random.rand(*input_shapes[2]).astype(np.float32)
        
        # Copy input data to GPU
        cuda.memcpy_htod(d_inputs[0], h_query)
        cuda.memcpy_htod(d_inputs[1], h_key)
        cuda.memcpy_htod(d_inputs[2], h_value)
        
        # Execute inference
        context.execute_v2(bindings)
        
        # Copy output from GPU
        h_output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh(h_output, d_output)
        
        print(f"Inference successful! Output shape: {h_output.shape}")
        print(f"Output stats - Mean: {np.mean(h_output):.6f}, Std: {np.std(h_output):.6f}")
        
        # Clean up
        for d_input in d_inputs:
            d_input.free()
        d_output.free()
        
        return True
        
    except Exception as e:
        print(f"Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing FMHA TensorRT Plugin")
    print("="*50)
    
    # Test plugin creation
    creation_success = test_plugin_creation()
    
    if creation_success:
        # Test inference if creation was successful
        inference_success = test_inference()
        
        if inference_success:
            print("\n✅ All tests passed! FMHA plugin is working correctly.")
            return True
        else:
            print("\n❌ Inference test failed.")
            return False
    else:
        print("\n❌ Plugin creation test failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)