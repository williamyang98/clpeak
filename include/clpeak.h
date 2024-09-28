#ifndef CLPEAK_HPP
#define CLPEAK_HPP

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <string.h>
#include <sstream>
#include <common.h>
#include <logger.h>

#define BUILD_OPTIONS " -cl-mad-enable "

using namespace std;

class clPeak
{
public:
  bool forcePlatform, forcePlatformName, forceDevice, forceDeviceName, forceTest, useEventTimer;
  bool isGlobalBW, isComputeHP, isComputeSP, isComputeDP, isComputeIntFast, isComputeInt, isTransferBW, isKernelLatency;
  bool isUseHostMemory;
  ulong specifiedPlatform, specifiedDevice;
  char *specifiedPlatformName;
  char *specifiedDeviceName;
  char *specifiedTestName;
  logger *log;

  clPeak();
  ~clPeak();

  inline cl_mem_flags get_read_mem_flags() {
    return isUseHostMemory ? CL_MEM_HOST_READ_ONLY : CL_MEM_READ_ONLY;
  }
  inline cl_mem_flags get_write_mem_flags() {
    return isUseHostMemory ? CL_MEM_HOST_WRITE_ONLY : CL_MEM_WRITE_ONLY;
  }

  int parseArgs(int argc, char **argv);

  // Return avg time in us
  float run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, uint iters);

  int runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeSP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeHP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeDP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeInteger(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeIntFast(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeChar(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeShort(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runComputeLong(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

  int runAll();
};

#endif // CLPEAK_HPP
