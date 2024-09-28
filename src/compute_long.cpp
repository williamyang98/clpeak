#include <clpeak.h>

int clPeak::runComputeLong(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  float timed, gflops;
  cl_uint workPerWI;
  cl::NDRange globalSize, localSize;
  cl_long A = 4;
  uint iters = devInfo.computeIters;

  if (!isComputeLong)
    return 0;

  try
  {
    log->print(NEWLINE TAB TAB "Long compute (GIOPS)" NEWLINE);
    log->xmlOpenTag("long_compute");
    log->xmlAppendAttribs("unit", "giops");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint64_t globalWIs = (devInfo.numCUs) * (devInfo.computeWgsPerCU) * (devInfo.maxWGSize);
    uint64_t t = std::min((globalWIs * sizeof(cl_long)), devInfo.maxAllocSize) / sizeof(cl_long);
    globalWIs = roundToMultipleOf(t, devInfo.maxWGSize);

    cl::Buffer outputBuf = cl::Buffer(ctx, get_write_mem_flags(), (globalWIs * sizeof(cl_long)));

    globalSize = globalWIs;
    localSize = devInfo.maxWGSize;

    cl::Kernel kernel_v1(prog, "compute_long_v1");
    kernel_v1.setArg(0, outputBuf), kernel_v1.setArg(1, A);

    cl::Kernel kernel_v2(prog, "compute_long_v2");
    kernel_v2.setArg(0, outputBuf), kernel_v2.setArg(1, A);

    cl::Kernel kernel_v4(prog, "compute_long_v4");
    kernel_v4.setArg(0, outputBuf), kernel_v4.setArg(1, A);

    cl::Kernel kernel_v8(prog, "compute_long_v8");
    kernel_v8.setArg(0, outputBuf), kernel_v8.setArg(1, A);

    cl::Kernel kernel_v16(prog, "compute_long_v16");
    kernel_v16.setArg(0, outputBuf), kernel_v16.setArg(1, A);

    ///////////////////////////////////////////////////////////////////////////
    // Vector width 1
    if (!forceTest || strcmp(specifiedTestName, "int") == 0)
    {
      log->print(TAB TAB TAB "long  : ");

      workPerWI = 2048/4; // Indicates long operations executed per work-item

      timed = run_kernel(queue, kernel_v1, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("long", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 2
    if (!forceTest || strcmp(specifiedTestName, "int2") == 0)
    {
      log->print(TAB TAB TAB "long2 : ");

      workPerWI = 2048/4;

      timed = run_kernel(queue, kernel_v2, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("long2", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 4
    if (!forceTest || strcmp(specifiedTestName, "int4") == 0)
    {
      log->print(TAB TAB TAB "long4 : ");

      workPerWI = 2048/4;

      timed = run_kernel(queue, kernel_v4, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("long4", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 8
    if (!forceTest || strcmp(specifiedTestName, "long8") == 0)
    {
      log->print(TAB TAB TAB "long8 : ");

      workPerWI = 2048/4;

      timed = run_kernel(queue, kernel_v8, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("long8", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 16
    if (!forceTest || strcmp(specifiedTestName, "long16") == 0)
    {
      log->print(TAB TAB TAB "long16: ");

      workPerWI = 2048/4;

      timed = run_kernel(queue, kernel_v16, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("long16", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////
    log->xmlCloseTag(); // long_compute
  }
  catch (cl::Error &error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    return -1;
  }

  return 0;
}
