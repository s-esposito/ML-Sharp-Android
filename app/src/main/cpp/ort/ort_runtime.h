// ort_runtime.h — ONNX Runtime (CPU EP) implementation of sharp::ModelRuntime.
// Phase 1 CPU fallback for devices without a usable Qualcomm HTP (e.g. Exynos).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model_runtime.h"

namespace Ort { struct Session; struct Value; }

namespace ortrt {

struct OrtOptions {
    int intraOpThreads = 0;      // 0 = let the caller's default apply (big-core count)
    bool useXnnpack = false;     // Phase 3 experiment; CPU EP (MLAS) by default
};

class OrtRuntime : public sharp::ModelRuntime {
public:
    OrtRuntime();
    ~OrtRuntime() override;

    // Marks the runtime usable (ORT env is created lazily, process-wide)
    int init(const OrtOptions& opts);

    // Loads an .onnx model (external-data files resolved relative to the model path).
    // Returns 0 on success.
    int load(const std::string& onnxPath);

    const char* backendName() const override { return m_opts.useXnnpack ? "ORT-XNNPACK" : "ORT-CPU"; }
    bool isReady() const override { return m_ready; }
    bool hasModel() const override { return m_session != nullptr; }

    const std::vector<sharp::TensorInfo>& getInputInfos() const override { return m_inputInfos; }
    const std::vector<sharp::TensorInfo>& getOutputInfos() const override { return m_outputInfos; }

    int execute(const std::vector<sharp::Tensor>& inputs, std::vector<sharp::Tensor>& outputs,
                bool keepOutputRaw = false) override;

    const uint8_t* getRawOutputData(size_t i) const override;
    void releaseRawOutput(size_t i) override;

    void freeGraph() override;

private:
    OrtOptions m_opts;
    bool m_ready = false;
    std::unique_ptr<Ort::Session> m_session;
    std::vector<sharp::TensorInfo> m_inputInfos;
    std::vector<sharp::TensorInfo> m_outputInfos;
    std::vector<std::vector<int64_t>> m_inputShapes;
    std::vector<std::vector<int64_t>> m_outputShapes;
    // keep-raw mode: ORT-allocated outputs, indexed like the caller's outputs vector
    std::vector<std::unique_ptr<Ort::Value>> m_rawOutputs;
};

} // namespace ortrt
