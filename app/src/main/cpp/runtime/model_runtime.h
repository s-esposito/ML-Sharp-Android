// model_runtime.h — minimal backend-neutral runtime surface used by sharp_jni.cpp.
//
// Phase 1 (CPU fallback): both qnn::HtpRuntime and ort::OrtRuntime implement this so
// the JNI stage functions (PE/IE/REST/AnyCalib) are written once. Phase 2 will grow
// this into the full InferenceBackend interface (probe/load/run/release/name).
//
// Conventions at this boundary:
//   - Caller tensors are float32, NCHW, in the layout of the exported ONNX graph.
//   - TensorInfo::dataType uses QNN_DATATYPE_* codes (FLOAT_32 for ORT models) so the
//     existing (de)quantisation helpers keep working unchanged.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sharp {

// Input/output tensor exchanged with a runtime
struct Tensor {
    std::string name;
    std::vector<uint32_t> dims;    // dims (NCHW, model-dependent)
    float* data = nullptr;         // float32 by default; raw quantised bytes (uint16/uint8) when quantized=true
    size_t count = 0;              // element count
    bool quantized = false;        // data points to raw quantised data (HTP only)
};

// Static metadata of a model input/output
struct TensorInfo {
    std::string name;
    std::vector<uint32_t> dims;
    uint32_t dataType = 0;       // QNN_DATATYPE_* code
    float scale = 1.0f;          // quantisation scale (1 for float)
    int32_t offset = 0;          // quantisation offset (0 for float)
    uint32_t quantEncoding = 0;  // raw quantisation encoding (0 = undefined)
    uint32_t bitwidth = 0;       // BW_SCALE_OFFSET bitwidth
};

class ModelRuntime {
public:
    virtual ~ModelRuntime() = default;

    // Short backend label for logs/UI ("HTP", "ORT-CPU", ...)
    virtual const char* backendName() const = 0;

    // Initialised and able to load a model
    virtual bool isReady() const = 0;

    // Whether a model/graph is currently loaded
    virtual bool hasModel() const = 0;

    virtual const std::vector<TensorInfo>& getInputInfos() const = 0;
    virtual const std::vector<TensorInfo>& getOutputInfos() const = 0;

    // Runs inference.
    // keepOutputRaw=false: every outputs[i].data must point to count floats; results are written there.
    // keepOutputRaw=true : outputs[i].data may be null; the results stay in a runtime-owned buffer
    //                      in the tensor's native dtype (see TensorInfo::dataType), read with
    //                      getRawOutputData(i) and released with releaseRawOutput(i).
    virtual int execute(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs,
                        bool keepOutputRaw = false) = 0;

    virtual const uint8_t* getRawOutputData(size_t i) const = 0;
    virtual void releaseRawOutput(size_t i) = 0;

    // Frees the loaded model/graph (runtime stays usable for the next load)
    virtual void freeGraph() = 0;
};

} // namespace sharp
