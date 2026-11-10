// proc_stats.h — tiny /proc readers shared by the JNI layer and both runtimes.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace sharp {

// Reads a "<key> <value> kB" line from a /proc file; 0 if missing/unreadable.
inline size_t procKb(const char* path, const char* key) {
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    const size_t keyLen = strlen(key);
    char line[256];
    size_t v = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, key, keyLen) == 0) {
            sscanf(line + keyLen, "%zu", &v);
            break;
        }
    }
    fclose(f);
    return v;
}

inline size_t selfVmRSS_kB() { return procKb("/proc/self/status", "VmRSS:"); }    // current RSS
inline size_t selfVmHWM_kB() { return procKb("/proc/self/status", "VmHWM:"); }    // peak RSS
inline size_t memAvailable_kB() { return procKb("/proc/meminfo", "MemAvailable:"); }

} // namespace sharp
