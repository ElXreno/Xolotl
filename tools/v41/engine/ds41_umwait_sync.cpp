#include <cuda_runtime.h>

#include <cpuid.h>
#include <x86intrin.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);

namespace {

using write_value_fn = int (*)(cudaStream_t, unsigned long long, unsigned int, unsigned int);

constexpr uint64_t kWaitCycles  = 100000;
constexpr uint64_t kCheckCycles = 3000000;

struct UmwaitSync {
    bool           on    = false;
    unsigned       state = 0;
    write_value_fn write = nullptr;
};

const UmwaitSync & umwait_sync() {
    static const UmwaitSync cfg = [] {
        UmwaitSync r;
        const char * e = std::getenv("XOLOTL_UMWAIT_SYNC");
        if (e == nullptr || e[0] == '\0' || e[0] == '0') {
            return r;
        }
        unsigned a = 0, b = 0, c = 0, d = 0;
        if (!__get_cpuid_count(7, 0, &a, &b, &c, &d) || !(c & (1u << 5))) {
            std::fprintf(stderr, "ds41_generate: XOLOTL_UMWAIT_SYNC: this CPU has no WAITPKG, CUDA syncs spin as before\n");
            return r;
        }
        void *                          fn = nullptr;
        cudaDriverEntryPointQueryResult q  = cudaDriverEntryPointSymbolNotFound;
        if (cudaGetDriverEntryPointByVersion("cuStreamWriteValue32", &fn, 12000, cudaEnableDefault, &q) != cudaSuccess ||
            q != cudaDriverEntryPointSuccess || fn == nullptr) {
            std::fprintf(stderr, "ds41_generate: XOLOTL_UMWAIT_SYNC: no cuStreamWriteValue32, CUDA syncs spin as before\n");
            return r;
        }
        r.write = reinterpret_cast<write_value_fn>(fn);
        r.state = std::strcmp(e, "c01") == 0 ? 1u : 0u;
        r.on    = true;
        std::fprintf(stderr, "ds41_generate: XOLOTL_UMWAIT_SYNC=%s: CUDA stream syncs sleep in UMWAIT C0.%u\n", e,
                     r.state ? 1u : 2u);
        return r;
    }();
    return cfg;
}

struct Slot {
    volatile uint32_t * host = nullptr;
    unsigned long long  dev  = 0;
    uint32_t            seq  = 0;
    bool                bad  = false;
};

thread_local Slot slot;

bool slot_ready() {
    if (slot.host != nullptr || slot.bad) {
        return !slot.bad;
    }
    void * p = nullptr;
    if (cudaHostAlloc(&p, 64, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
        slot.bad = true;
        return false;
    }
    void * dp = nullptr;
    if (cudaHostGetDevicePointer(&dp, p, 0) != cudaSuccess) {
        cudaFreeHost(p);
        slot.bad = true;
        return false;
    }
    slot.host  = static_cast<volatile uint32_t *>(p);
    *slot.host = 0;
    slot.dev   = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(dp));
    return true;
}

inline void umonitor(volatile void * addr) {
    asm volatile(".byte 0xf3, 0x0f, 0xae, 0xf0" ::"a"(addr) : "memory");
}

inline void umwait(unsigned state, uint64_t deadline) {
    asm volatile(".byte 0xf2, 0x0f, 0xae, 0xf1" ::"c"(state), "a"(static_cast<uint32_t>(deadline)),
                 "d"(static_cast<uint32_t>(deadline >> 32))
                 : "memory", "cc");
}

}  // namespace

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
    const UmwaitSync & cfg = umwait_sync();
    if (!cfg.on) {
        return __real_cudaStreamSynchronize(stream);
    }
    const cudaError_t q = cudaStreamQuery(stream);
    if (q == cudaSuccess) {
        return cudaSuccess;
    }
    if (q != cudaErrorNotReady || !slot_ready()) {
        return __real_cudaStreamSynchronize(stream);
    }
    const uint32_t want = ++slot.seq;
    if (cfg.write(stream, slot.dev, want, 0) != 0) {
        return __real_cudaStreamSynchronize(stream);
    }
    uint64_t check = __rdtsc() + kCheckCycles;
    while (*slot.host != want) {
        umonitor(slot.host);
        if (*slot.host == want) {
            break;
        }
        const uint64_t now = __rdtsc();
        if (now >= check) {
            if (cudaStreamQuery(stream) != cudaErrorNotReady) {
                break;
            }
            check = now + kCheckCycles;
        }
        umwait(cfg.state, now + kWaitCycles);
    }
    return __real_cudaStreamSynchronize(stream);
}
