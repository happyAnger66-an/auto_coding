# FMHA TensorRT Plugin - Quick Start Guide

## 1. Clone and Setup

```bash
cd /home/zhangxa/codes/auto_coding/tensorrt_plugin
```

## 2. Build the Plugin

```bash
chmod +x build.sh
./build.sh
```

Wait for compilation to complete. The plugin will be built at `build/lib/libfmha_plugin.so`.

## 3. Test the Plugin

```bash
python test_fmha_plugin.py
```

## 4. Integrate into Your TensorRT Application

### Load the Plugin

```python
import ctypes
# Load the plugin library before creating any TensorRT objects
ctypes.CDLL('./build/lib/libfmha_plugin.so')
```

### Use in Network Definition

```python
import tensorrt as trt
import numpy as np

def add_fmha_layer(network, query_tensor, key_tensor, value_tensor, num_heads=8, head_size=64, is_causal=False):
    # Get plugin creator
    plugin_registry = trt.get_plugin_registry()
    fmha_plugin_creator = None
    
    for plugin_creator in plugin_registry.plugin_creator_list:
        if plugin_creator.name == "FMHA_TRT":
            fmha_plugin_creator = plugin_creator
            break
    
    if fmha_plugin_creator is None:
        raise RuntimeError("FMHA plugin not found")
    
    # Create plugin with parameters
    plugin_fields = [
        trt.PluginField("numHeads", np.array([num_heads], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("headSize", np.array([head_size], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("isCausal", np.array([is_causal], dtype=bool), trt.PluginFieldType.BOOL)
    ]
    
    plugin_field_collection = trt.PluginFieldCollection(plugin_fields)
    fmha_plugin = fmha_plugin_creator.create_plugin(name="fmha_layer", field_collection=plugin_field_collection)
    
    # Add plugin layer to network
    fmha_layer = network.add_plugin_v2(inputs=[query_tensor, key_tensor, value_tensor], plugin=fmha_plugin)
    return fmha_layer.get_output(0)
```

## 5. Example Usage

```python
import tensorrt as trt
import ctypes

# Load plugin
ctypes.CDLL('./build/lib/libfmha_plugin.so')

# Create builder and network
builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Define inputs
batch_size, seq_len, num_heads, head_size = 1, 128, 8, 64
query = network.add_input('query', trt.float32, (batch_size, seq_len, num_heads, head_size))
key = network.add_input('key', trt.float32, (batch_size, seq_len, num_heads, head_size))
value = network.add_input('value', trt.float32, (batch_size, seq_len, num_heads, head_size))

# Add FMHA layer
output = add_fmha_layer(network, query, key, value, num_heads=num_heads, head_size=head_size)

# Mark output
network.mark_output(output)

# Build engine
config = builder.create_builder_config()
serialized_engine = builder.build_serialized_network(network, config)
```

## 6. Verify Installation

Check that the plugin is properly registered:

```python
import tensorrt as trt
import ctypes

ctypes.CDLL('./build/lib/libfmha_plugin.so')

registry = trt.get_plugin_registry()
plugin_names = [creator.name for creator in registry.plugin_creator_list]
print("Available plugins:", plugin_names)
print("FMHA plugin available:", "FMHA_TRT" in plugin_names)
```

That's it! You now have a working FMHA TensorRT plugin that can be integrated into your inference pipeline.