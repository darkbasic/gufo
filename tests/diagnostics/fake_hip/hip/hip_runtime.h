#pragma once

#include "tests/cache/fake_hip/hip/hip_runtime.h"

// Additional host-runtime entry points used by the native resource probe.
hipError_t hipInit(unsigned);
hipError_t hipSetDevice(int);
hipError_t hipRuntimeGetVersion(int*);
hipError_t hipDriverGetVersion(int*);
hipError_t hipMalloc(void**, std::size_t);
hipError_t hipFree(void*);
