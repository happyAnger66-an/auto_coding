#ifndef FMHA_PLUGIN_H
#define FMHA_PLUGIN_H

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <vector>

namespace nvinfer1 {
namespace plugin {

class FMHAPlugin : public IPluginV2DynamicExt {
private:
    std::string mLayerName;
    int mNumHeads;
    int mHeadSize;
    float mScale;
    bool mIsCausal;

public:
    FMHAPlugin(int numHeads, int headSize, bool isCausal = false);
    
    FMHAPlugin(const void* data, size_t length);

    ~FMHAPlugin();

    // IPluginV2 methods
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    int initialize() override;
    void terminate() override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void destroy() override;
    IPluginV2* clone() const override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

    // IPluginV2Ext methods
    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override;
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
    bool canBroadcastInputAcrossBatch(int inputIndex) const override;
    void attachToContext(cudnnContext* cudnn, cublasContext* cublas, IGpuAllocator* allocator) override;
    void detachFromContext() override;

    // IPluginV2DynamicExt methods
    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) override;
    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override;
    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) override;
    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const override;
    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) override;

private:
    size_t getTypeSize(DataType dtype) const;
};

class FMHAPluginCreator : public IPluginCreator {
private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;

public:
    FMHAPluginCreator();
    ~FMHAPluginCreator() override = default;

    const char* getPluginName() const override;
    const char* getPluginVersion() const override;
    const PluginFieldCollection* getFieldNames() override;
    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override;
    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    void setPluginNamespace(const char* libNamespace) override;
    const char* getPluginNamespace() const override;

private:
    std::string mNamespace;
};

} // namespace plugin
} // namespace nvinfer1

#endif // FMHA_PLUGIN_H