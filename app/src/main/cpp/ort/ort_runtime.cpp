// ort_runtime.cpp — ONNX Runtime (CPU EP) implementation of sharp::ModelRuntime.
#include "ort_runtime.h"

#include <android/log.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "onnxruntime_cxx_api.h"
#include "QnnTypes.h"   // QNN_DATATYPE_FLOAT_32: TensorInfo dtype codes are shared with the HTP path

#define LOG_TAG "SHARP_ORT"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)

namespace ortrt {

namespace {

size_t selfVmRSS_kB() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char line[256];
    size_t rss = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line + 6, "%zu", &rss); break; }
    }
    fclose(f);
    return rss;
}

// One ORT environment per process (ORT requires a single Env)
Ort::Env& env() {
    static std::once_flag once;
    static std::unique_ptr<Ort::Env> e;
    std::call_once(once, [] {
        e = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "sharp");
    });
    return *e;
}

std::string shapeStr(const std::vector<int64_t>& s) {
    std::string r;
    for (size_t i = 0; i < s.size(); i++) { if (i) r += "x"; r += std::to_string(s[i]); }
    return r;
}

long msSince(std::chrono::steady_clock::time_point t0) {
    return (long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();
}

} // namespace

OrtRuntime::OrtRuntime() = default;
OrtRuntime::~OrtRuntime() { freeGraph(); }

int OrtRuntime::init(const OrtOptions& opts) {
    m_opts = opts;
    try {
        (void)env();
    } catch (const Ort::Exception& e) {
        LOGE("ORT env creation failed: %s", e.what());
        return -1;
    }
    m_ready = true;
    LOGI("ORT runtime ready (threads=%d, xnnpack=%d, ort=%s)",
         m_opts.intraOpThreads, (int)m_opts.useXnnpack, Ort::GetVersionString().c_str());
    return 0;
}

int OrtRuntime::load(const std::string& onnxPath) {
    if (!m_ready) return -1;
    if (m_session) freeGraph();

    auto t0 = std::chrono::steady_clock::now();
    size_t rss0 = selfVmRSS_kB();
    try {
        Ort::SessionOptions so;
        if (m_opts.intraOpThreads > 0) so.SetIntraOpNumThreads(m_opts.intraOpThreads);
        so.SetInterOpNumThreads(1);
        so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        so.EnableCpuMemArena();
        so.EnableMemPattern();
        if (m_opts.useXnnpack) {
            so.AppendExecutionProvider("XNNPACK",
                {{"intra_op_num_threads", std::to_string(m_opts.intraOpThreads)}});
        }
        m_session = std::make_unique<Ort::Session>(env(), onnxPath.c_str(), so);

        Ort::AllocatorWithDefaultOptions alloc;
        auto readInfos = [&](bool inputs) -> int {
            size_t n = inputs ? m_session->GetInputCount() : m_session->GetOutputCount();
            auto& infos = inputs ? m_inputInfos : m_outputInfos;
            auto& shapes = inputs ? m_inputShapes : m_outputShapes;
            infos.clear(); shapes.clear();
            for (size_t i = 0; i < n; i++) {
                auto name = inputs ? m_session->GetInputNameAllocated(i, alloc)
                                   : m_session->GetOutputNameAllocated(i, alloc);
                auto ti = inputs ? m_session->GetInputTypeInfo(i) : m_session->GetOutputTypeInfo(i);
                auto tsi = ti.GetTensorTypeAndShapeInfo();
                if (tsi.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    LOGE("%s[%zu] '%s' is not float32 (elem type %d); unsupported",
                         inputs ? "input" : "output", i, name.get(), (int)tsi.GetElementType());
                    return -1;
                }
                std::vector<int64_t> shape = tsi.GetShape();
                sharp::TensorInfo info;
                info.name = name.get();
                info.dataType = QNN_DATATYPE_FLOAT_32;
                for (auto& d : shape) {
                    if (d <= 0) d = 1;  // dynamic dim (batch) -> 1; the SHARP graphs are static
                    info.dims.push_back((uint32_t)d);
                }
                LOGI("  %s[%zu] name='%s' dims=[%s]", inputs ? "Input" : "Output", i,
                     info.name.c_str(), shapeStr(shape).c_str());
                infos.push_back(std::move(info));
                shapes.push_back(std::move(shape));
            }
            return 0;
        };
        if (readInfos(true) != 0 || readInfos(false) != 0) {
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
    LOGI("ORT loaded %s in %ldms (%zu in, %zu out), RSS %zu -> %zu KB",
         onnxPath.c_str(), msSince(t0), m_inputInfos.size(), m_outputInfos.size(),
         rss0, selfVmRSS_kB());
    return 0;
}

int OrtRuntime::execute(const std::vector<sharp::Tensor>& inputs,
                        std::vector<sharp::Tensor>& outputs, bool keepOutputRaw) {
    if (!m_session) return -1;
    try {
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Inputs: wrap caller buffers (no copy), ordered as given, matched by name
        std::vector<const char*> inNames;
        std::vector<Ort::Value> inVals;
        for (const auto& in : inputs) {
            size_t idx = m_inputInfos.size();
            for (size_t j = 0; j < m_inputInfos.size(); j++)
                if (m_inputInfos[j].name == in.name) { idx = j; break; }
            if (idx == m_inputInfos.size()) {
                LOGE("execute: unknown input '%s'", in.name.c_str());
                return -3;
            }
            if (in.quantized) {
                LOGE("execute: input '%s' flagged quantized; ORT path expects float32", in.name.c_str());
                return -4;
            }
            size_t expect = 1;
            for (auto d : m_inputShapes[idx]) expect *= (size_t)d;
            if (in.count < expect) {
                LOGE("execute: input '%s' has %zu elems, model expects %zu", in.name.c_str(), in.count, expect);
                return -5;
            }
            inNames.push_back(m_inputInfos[idx].name.c_str());
            inVals.push_back(Ort::Value::CreateTensor<float>(
                mem, in.data, expect, m_inputShapes[idx].data(), m_inputShapes[idx].size()));
        }

        std::vector<const char*> outNames;
        std::vector<size_t> outIdx;
        for (const auto& out : outputs) {
            size_t idx = m_outputInfos.size();
            for (size_t j = 0; j < m_outputInfos.size(); j++)
                if (m_outputInfos[j].name == out.name) { idx = j; break; }
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
                const auto& shp = m_outputShapes[outIdx[k]];
                size_t n = 1; for (auto d : shp) n *= (size_t)d;
                if (!outputs[k].data || outputs[k].count < n) {
                    LOGE("execute: output '%s' buffer too small (%zu < %zu)",
                         outputs[k].name.c_str(), outputs[k].count, n);
                    return -7;
                }
                outVals.push_back(Ort::Value::CreateTensor<float>(
                    mem, outputs[k].data, n, shp.data(), shp.size()));
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
    m_inputShapes.clear();
    m_outputShapes.clear();
}

} // namespace ortrt
