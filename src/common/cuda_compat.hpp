#pragma once

#if defined(__CUDACC__)
#define NEUROEVOLUTION_HOST_DEVICE __host__ __device__
#else
#define NEUROEVOLUTION_HOST_DEVICE
#endif
