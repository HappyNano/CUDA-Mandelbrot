#include "mandelbrot.cuh"

#include <cmath>

#include "mandelbrot.cuh"

__global__ void mandelbrot_kernel(
  int* iterations,
  const double x_start,
  const double y_fin,
  const double dx,
  const double dy,
  const int width,
  const int height
)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height)
  {
    const int limit = 100000;

    double real = x_start + x * dx;
    double imag = y_fin - y * dy;
    double zReal = real;
    double zImag = imag;

    int i = 0;

    for (i = 0; i < limit; ++i)
    {
      double r2 = zReal * zReal;
      double i2 = zImag * zImag;

      if (r2 + i2 > 4.0)
        break;

      zImag = 2.0 * zReal * zImag + imag;
      zReal = r2 - i2 + real;
    }

    iterations[x * height + y] = i;
  }
}

void mandelbrot(
  int* iterations,
  const double x_start,
  const double y_fin,
  const double dx,
  const double dy,
  const int width,
  const int height
)
{
  dim3 threads(32, 32);
  dim3 blocks((width + threads.x - 1) / threads.x, (height + threads.y - 1) / threads.y);

  //dim3 threads_per_block(32, 32);
  //dim3 blocks_per_grid(1, 1);
  //blocks_per_grid.x = std::ceil(static_cast<double>(width) /
  //                              static_cast<double>(threads_per_block.x));
  //blocks_per_grid.y = std::ceil(static_cast<double>(height) /
  //                              static_cast<double>(threads_per_block.y));

  mandelbrot_kernel<<<blocks, threads>>>(iterations, x_start, y_fin, dx, dy, width, height);
}
