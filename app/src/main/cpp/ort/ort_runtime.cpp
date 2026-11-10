// ort_runtime.cpp — ONNX Runtime (CPU EP) implementation of sharp::ModelRuntime.
#include "ort_runtime.h"

#include <android/log.h>
#include <chrono>
#include <mutex>

#include "onnxruntime_cxx_api.h"
#include "proc_stats.h"
#include "QnnTypes.h"   // QNN_DATATYPE_FLOAT_32: TensorInfo dtype codes are shared with the HTP path

#define LOG_TAG "SHARP_ORT"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

namespace ortrt {

namespace {

using sharp::selfVmRSS_kB;
using sharp::TensorInfo;

// One ORT environment per process (ORT requires a single Env)
Ort::Env& env() {
    static std::once_flag once;
    static std::unique_ptr<Ort::Env> e;
    std::call_once(once, [] {
        e = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sharp");
    });
    return *e;
}

std::vector<int64_t> toShape(const std::vector<uint32_t>& dims) {
    return std::vector<int64_t>(dims.begin(), dims.end());
}

size_t elemCount(const std::vector<uint32_t>& dims) {
    size_t n = 1;
    for (auto d : dims) n *= d;
    return n;
}

// Index of the tensor named `name`, or infos.size() if absent
size_t indexOf(const std::vector<TensorInfo>& infos, const std::string& name) {
    for (size_t j = 0; j < infos.size(); j++)
        if (infos[j].name == name) return j;
    return infos.size();
}

long msSince(std::chrono::steady_clock::time_point t0) {
    return (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

// Reads input/output metadata; returns false if a tensor is not float32
bool readInfos(Ort::Session& s, bool inputs, std::vector<TensorInfo>& infos) {
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t n = inputs ? s.GetInputCount() : s.GetOutputCount();
    infos.clear();
    for (size_t i = 0; i < n; i++) {
        auto name = inputs ? s.GetInputNameAllocated(i, alloc) : s.GetOutputNameAllocated(i, alloc);
        auto ti = inputs ? s.GetInputTypeInfo(i) : s.GetOutputTypeInfo(i);
        auto tsi = ti.GetTensorTypeAndShapeInfo();
        if (tsi.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            LOGE("%s[%zu] '%s' is not float32 (elem type %d); unsupported",
                 inputs ? "input" : "output", i, name.get(), (int)tsi.GetElementType());
            return false;
        }
        TensorInfo info;
        info.name = name.get();
        info.dataType = QNN_DATATYPE_FLOAT_32;
        std::string dimStr;
        for (int64_t d : tsi.GetShape()) {
            // dynamic dim (batch) -> 1; the SHARP graphs are static
            info.dims.push_back(d > 0 ? (uint32_t)d : 1u);
            dimStr += (dimStr.empty() ? "" : "x") + std::to_string(d);
        }
        LOGI("  %s[%zu] name='%s' dims=[%s]", inputs ? "Input" : "Output", i,
             info.name.c_str(), dimStr.c_str());
        infos.push_back(std::move(info));
    }
    return true;
}

} // namespace

OrtRuntime::OrtRuntime(int intraOpThreads) : m_threads(intraOpThreads) {}
OrtRuntime::~OrtRuntime() { freeGraph(); }

int OrtRuntime::load(const std::string& onnxPath, bool useArena) {
    if (m_session) freeGraph();

    auto t0 = std::chrono::steady_clock::now();
    size_t rss0 = selfVmRSS_kB();
    try {
        Ort::SessionOptions so;
        so.SetIntraOpNumThreads(m_threads);
        so.SetInterOpNumThreads(1);
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (useArena) so.EnableCpuMemArena(); else so.DisableCpuMemArena();
        so.EnableMemPattern();
        m_session = std::make_unique<Ort::Session>(env(), onnxPath.c_str(), so);
        if (!readInfos(*m_session, true, m_inputInfos) || !readInfos(*m_session, false, m_outputInfos)) {
            freeGraph();
            return -3;
        }
    } catch (const Ort::Exception& e) {
        LOGE("ORT load failed for %s: %s", onnxPath.c_str(), e.what());
        freeGraph();
        return -2;
    } catch (const std::bad_alloc&) {
        LOGE("ORT load OOM for %s (RSS=%zuKB)", onnxPath.c_str(), selfVmRSS_kB());
        freeGraph();
        return -0xE0000001;
    }
    LOGI("ORT %s loaded %s in %ldms (%zu in, %zu out, threads=%d, arena=%d), RSS %zu -> %zu KB",
         Ort::GetVersionString().c_str(), onnxPath.c_str(), msSince(t0), m_inputInfos.size(),
         m_outputInfos.size(), m_threads, (int)useArena, rss0, selfVmRSS_kB());
    return 0;
}

int OrtRuntime::execute(const std::vector<sharp::Tensor>& inputs,
                        std::vector<sharp::Tensor>& outputs, bool keepOutputRaw) {
    if (!m_session) return -1;
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Inputs: wrap caller buffers (no copy), matched by name
        std::vector<const char*> inNames;
        std::vector<Ort::Value> inVals;
        for (const auto& in : inputs) {
            size_t idx = indexOf(m_inputInfos, in.name);
            if (idx == m_inputInfos.size()) {
                LOGE("execute: unknown input '%s'", in.name.c_str());
                return -3;
            }
            if (in.quantized) {
                LOGE("execute: input '%s' flagged quantized; ORT path expects float32", in.name.c_str());
                return -4;
            }
            const auto& info = m_inputInfos[idx];
            const size_t expect = elemCount(info.dims);
            if (in.count < expect) {
                LOGE("execute: input '%s' has %zu elems, model expects %zu", in.name.c_str(), in.count, expect);
                return -5;
            }
            auto shape = toShape(info.dims);
            inNames.push_back(info.name.c_str());
            inVals.push_back(Ort::Value::CreateTensor<float>(mem, in.data, expect, shape.data(), shape.size()));
        }

        std::vector<const char*> outNames;
        std::vector<size_t> outIdx;
        for (const auto& out : outputs) {
            size_t idx = indexOf(m_outputInfos, out.name);
            if (idx == m_outputInfos.size()) {
                LOGE("execute: unknown output '%s'", out.name.c_str());
                return -6;
            }
            outNames.push_back(m_outputInfos[idx].name.c_str());
            outIdx.push_back(idx);
        }

        auto t0 = std::chrono::steady_clock::now();
        size_t rss0 = selfVmRSS_kB();
        Ort::RunOptions ro;
        if (!keepOutputRaw) {
            // Bind outputs straight into the caller's float buffers
            std::vector<Ort::Value> outVals;
            for (size_t k = 0; k < outputs.size(); k++) {
                const auto& info = m_outputInfos[outIdx[k]];
                const size_t n = elemCount(info.dims);
                if (!outputs[k].data || outputs[k].count < n) {
                    LOGE("execute: output '%s' buffer too small (%zu < %zu)",
                         outputs[k].name.c_str(), outputs[k].count, n);
                    return -7;
                }
                auto shape = toShape(info.dims);
                outVals.push_back(Ort::Value::CreateTensor<float>(mem, outputs[k].data, n, shape.data(), shape.size()));
            }
            m_session->Run(ro, inNames.data(), inVals.data(), inVals.size(),
                           outNames.data(), outVals.data(), outVals.size());
        } else {
            // Let ORT allocate; caller streams them out via getRawOutputData()
            auto res = m_session->Run(ro, inNames.data(), inVals.data(), inVals.size(),
                                      outNames.data(), outNames.size());
            m_rawOutputs.clear();
            m_rawOutputs.resize(outputs.size());
            for (size_t k = 0; k < res.size(); k++)
                m_rawOutputs[k] = std::make_unique<Ort::Value>(std::move(res[k]));
        }
        LOGI("ORT run %ldms, RSS %zu -> %zu KB", msSince(t0), rss0, selfVmRSS_kB());
    } catch (const Ort::Exception& e) {
        LOGE("ORT execute failed: %s (RSS=%zuKB)", e.what(), selfVmRSS_kB());
        return -2;
    } catch (const std::bad_alloc&) {
        LOGE("ORT execute OOM (RSS=%zuKB)", selfVmRSS_kB());
        return -0xE0000002;
    }
    return 0;
}

const uint8_t* OrtRuntime::getRawOutputData(size_t i) const {
    if (i >= m_rawOutputs.size() || !m_rawOutputs[i]) return nullptr;
    return reinterpret_cast<const uint8_t*>(m_rawOutputs[i]->GetTensorData<float>());
}

void OrtRuntime::releaseRawOutput(size_t i) {
    if (i < m_rawOutputs.size()) m_rawOutputs[i].reset();
}

void OrtRuntime::freeGraph() {
    m_rawOutputs.clear();
    m_session.reset();
    m_inputInfos.clear();
    m_outputInfos.clear();
}

} // namespace ortrt
