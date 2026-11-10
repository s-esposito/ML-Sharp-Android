// ort_runtime.h — ONNX Runtime (CPU EP) implementation of sharp::ModelRuntime.
// Phase 1 CPU fallback for devices without a usable Qualcomm HTP (e.g. Exynos).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model_runtime.h"

namespace Ort { struct Session; struct Value; }

namespace ortrt {

class OrtRuntime : public sharp::ModelRuntime {
public:
    // intraOpThreads: ORT intra-op thread count (> 0)
    explicit OrtRuntime(int intraOpThreads);
    ~OrtRuntime() override;

    // Loads an .onnx model (external-data files resolved relative to the model path).
    // useArena: keep ORT's CPU BFC arena. Worth it only for models run many times
    // (PE, 35 runs); for single-run models it pins the peak until the session is
    // freed, so streamed-out outputs never give memory back. Returns 0 on success.
    int load(const std::string& onnxPath, bool useArena);

    const char* backendName() const override { return "ORT-CPU"; }
    bool isReady() const override { return true; }
    bool hasModel() const override { return m_session != nullptr; }

    const std::vector<sharp::TensorInfo>& getInputInfos() const override { return m_inputInfos; }
    const std::vector<sharp::TensorInfo>& getOutputInfos() const override { return m_outputInfos; }

    int execute(const std::vector<sharp::Tensor>& inputs, std::vector<sharp::Tensor>& outputs,
                bool keepOutputRaw = false) override;

    const uint8_t* getRawOutputData(size_t i) const override;
    void releaseRawOutput(size_t i) override;

    void freeGraph() override;

    int threads() const { return m_threads; }

private:
    int m_threads;
    std::unique_ptr<Ort::Session> m_session;
    std::vector<sharp::TensorInfo> m_inputInfos;
    std::vector<sharp::TensorInfo> m_outputInfos;
    // keep-raw mode: ORT-allocated outputs, indexed like the caller's outputs vector
    std::vector<std::unique_ptr<Ort::Value>> m_rawOutputs;
};

} // namespace ortrt
