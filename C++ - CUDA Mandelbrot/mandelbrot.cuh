#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH

#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime.h>

void mandelbrot(
  int* iterations,
  const double x_start,
  const double y_fin,
  const double dx,
  const double dy,
  const int width,
  const int height
);

#endif
