#include "fmha_plugin.h"
#include <cstring>
#include <iostream>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

// Helper function to get data type size
size_t FMHAPlugin::getTypeSize(DataType dtype) const {
    switch (dtype) {
        case DataType::kFLOAT: return sizeof(float);
        case DataType::kHALF: return sizeof(half);
        case DataType::kINT8: return sizeof(int8_t);
        case DataType::kINT32: return sizeof(int32_t);
        case DataType::kBOOL: return sizeof(bool);
        default: return 0;
    }
}

// Constructor
FMHAPlugin::FMHAPlugin(int numHeads, int headSize, bool isCausal)
    : mNumHeads(numHeads), mHeadSize(headSize), mIsCausal(isCausal) {
    mScale = 1.0f / sqrtf(static_cast<float>(headSize));
}

// Deserialization constructor
FMHAPlugin::FMHAPlugin(const void* data, size_t length) {
    const char* d = static_cast<const char*>(data);
    const char* a = d;
    
    mNumHeads = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    mHeadSize = *reinterpret_cast<const int*>(d);
    d += sizeof(int);
    mScale = *reinterpret_cast<const float*>(d);
    d += sizeof(float);
    mIsCausal = *reinterpret_cast<const bool*>(d);
    d += sizeof(bool);
    
    assert(d == a + length);
}

// Destructor
FMHAPlugin::~FMHAPlugin() {}

// IPluginV2 methods
const char* FMHAPlugin::getPluginType() const {
    return "FMHA_TRT";
}

const char* FMHAPlugin::getPluginVersion() const {
    return "1_0";
}

int FMHAPlugin::getNbOutputs() const {
    return 1;
}

int FMHAPlugin::initialize() {
    return 0;
}

void FMHAPlugin::terminate() {}

size_t FMHAPlugin::getSerializationSize() const {
    return sizeof(int) * 2 + sizeof(float) + sizeof(bool);
}

void FMHAPlugin::serialize(void* buffer) const {
    char* d = static_cast<char*>(buffer);
    const char* a = d;
    
    *reinterpret_cast<int*>(d) = mNumHeads;
    d += sizeof(int);
    *reinterpret_cast<int*>(d) = mHeadSize;
    d += sizeof(int);
    *reinterpret_cast<float*>(d) = mScale;
    d += sizeof(float);
    *reinterpret_cast<bool*>(d) = mIsCausal;
    d += sizeof(bool);
    
    assert(d == a + getSerializationSize());
}

void FMHAPlugin::destroy() {
    delete this;
}

IPluginV2* FMHAPlugin::clone() const {
    return new FMHAPlugin(mNumHeads, mHeadSize, mIsCausal);
}

void FMHAPlugin::setPluginNamespace(const char* pluginNamespace) {
    mLayerName = pluginNamespace;
}

const char* FMHAPlugin::getPluginNamespace() const {
    return mLayerName.c_str();
}

// IPluginV2Ext methods
DataType FMHAPlugin::getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const {
    // Output type matches input type
    return inputTypes[0];
}

bool FMHAPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const {
    return false;
}

bool FMHAPlugin::canBroadcastInputAcrossBatch(int inputIndex) const {
    return false;
}

void FMHAPlugin::attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* allocator) {}

void FMHAPlugin::detachFromContext() {}

// IPluginV2DynamicExt methods
DimsExprs FMHAPlugin::getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) {
    // Output has same dimensions as input[0] (query)
    return inputs[0];
}

bool FMHAPlugin::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) {
    const PluginTensorDesc& desc = inOut[pos];
    
    if (desc.type == DataType::kFLOAT) {
        if (desc.format != TensorFormat::kLINEAR) {
            return false;
        }
    } else if (desc.type == DataType::kHALF) {
        if (desc.format != TensorFormat::kLINEAR) {
            return false;
        }
    } else {
        return false;
    }
    
    if (pos == 0) {
        // Input 0 (query) - should be float or half
        return (desc.type == DataType::kFLOAT || desc.type == DataType::kHALF);
    } else if (pos == 1) {
        // Input 1 (key) - should match input 0
        return (desc.type == inOut[0].type);
    } else if (pos == 2) {
        // Input 2 (value) - should match input 0
        return (desc.type == inOut[0].type);
    } else if (pos == 3) {
        // Output - should match input 0
        return (desc.type == inOut[0].type);
    }
    
    return false;
}

void FMHAPlugin::configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) {
    // Validate input dimensions
    assert(nbInputs == 3);  // Query, Key, Value
    assert(nbOutputs == 1); // Output
    
    const auto& q_desc = in[0].desc;
    const auto& k_desc = in[1].desc;
    const auto& v_desc = in[2].desc;
    
    // All inputs should have the same dimensions except for the sequence length
    assert(q_desc.dims.nbDims == 4); // [batch, seq_len, num_heads, head_size]
    assert(k_desc.dims.nbDims == 4);
    assert(v_desc.dims.nbDims == 4);
    
    // Dimensions should match: batch, num_heads, head_size
    assert(q_desc.dims.d[0] == k_desc.dims.d[0]); // batch
    assert(q_desc.dims.d[0] == v_desc.dims.d[0]); // batch
    assert(q_desc.dims.d[2] == k_desc.dims.d[2]); // num_heads
    assert(q_desc.dims.d[2] == v_desc.dims.d[2]); // num_heads
    assert(q_desc.dims.d[3] == k_desc.dims.d[3]); // head_size
    assert(q_desc.dims.d[3] == v_desc.dims.d[3]); // head_size
}

size_t FMHAPlugin::getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const {
    // No additional workspace needed for this implementation
    return 0;
}

// External function declaration for the CUDA kernel
extern "C" {
    void launch_fmha_kernel(
        const void* query,
        const void* key, 
        const void* value,
        void* output,
        int batch_size,
        int seq_len,
        int num_heads,
        int head_size,
        float scale,
        bool is_causal,
        cudaStream_t stream,
        bool use_half_precision
    );
}

int FMHAPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) {
    const void* query = inputs[0];
    const void* key = inputs[1]; 
    const void* value = inputs[2];
    void* output = outputs[0];
    
    // Extract dimensions from input descriptor
    const auto& q_dims = inputDesc[0].dims;
    int batch_size = q_dims.d[0];
    int seq_len = q_dims.d[1];
    int num_heads = q_dims.d[2];
    int head_size = q_dims.d[3];
    
    // Determine if using half precision
    bool use_half_precision = (inputDesc[0].type == DataType::kHALF);
    
    // Launch the CUDA kernel
    launch_fmha_kernel(
        query, key, value, output,
        batch_size, seq_len, num_heads, head_size,
        mScale, mIsCausal, stream, use_half_precision
    );
    
    return 0;
}

// Plugin Creator Implementation
PluginFieldCollection FMHAPluginCreator::mFC{};
std::vector<PluginField> FMHAPluginCreator::mPluginAttributes;

FMHAPluginCreator::FMHAPluginCreator() {
    mPluginAttributes.clear();
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FMHAPluginCreator::getPluginName() const {
    return "FMHA_TRT";
}

const char* FMHAPluginCreator::getPluginVersion() const {
    return "1_0";
}

const PluginFieldCollection* FMHAPluginCreator::getFieldNames() {
    return &mFC;
}

IPluginV2* FMHAPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) {
    int numHeads = 8;  // default
    int headSize = 64; // default
    bool isCausal = false; // default
    
    for (int i = 0; i < fc->nbFields; i++) {
        if (!strcmp(fc->fields[i].name, "numHeads")) {
            numHeads = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(fc->fields[i].name, "headSize")) {
            headSize = *static_cast<const int*>(fc->fields[i].data);
        } else if (!strcmp(fc->fields[i].name, "isCausal")) {
            isCausal = *static_cast<const bool*>(fc->fields[i].data);
        }
    }
    
    FMHAPlugin* plugin = new FMHAPlugin(numHeads, headSize, isCausal);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

IPluginV2* FMHAPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) {
    auto plugin = new FMHAPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void FMHAPluginCreator::setPluginNamespace(const char* libNamespace) {
    mNamespace = libNamespace;
}

const char* FMHAPluginCreator::getPluginNamespace() const {
    return mNamespace.c_str();
}

// Register the plugin creator
REGISTER_TENSORRT_PLUGIN(FMHAPluginCreator);