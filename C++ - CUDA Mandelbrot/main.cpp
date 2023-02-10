#pragma comment(lib, "SDL2test.lib")

#include <iostream>
#include <complex>
#include <cmath>
#include <utility>
#include <SDL.h>

#include "mandelbrot.cuh"

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
#define RMASK 0xff000000
#define GMASK 0x00ff0000
#define BMASK 0x0000ff00
#define AMASK 0x000000ff
#else
#define RMASK 0x000000ff
#define GMASK 0x0000ff00
#define BMASK 0x00ff0000
#define AMASK 0xff000000
#endif

void set_pixel(SDL_Surface* surface, int x, int y, Uint32 pixel)
{
  Uint32* const target_pixel
    = reinterpret_cast<Uint32*>((static_cast<Uint8*>(surface->pixels)) + y * surface->pitch + x * surface->format->BytesPerPixel);
  *target_pixel = pixel;
}

Uint32 get_color(SDL_Surface* surface, SDL_Color color)
{
  return SDL_MapRGBA(surface->format, color.r, color.g, color.b, color.a);
}

const auto width = 1000;
const auto height = 1000;

double x_start = -2.25;
double x_fin = 0.75;
double y_start = -1.5;
double y_fin = 1.5;

double scale_pos = 1;
const double scaling = 2; // можно так сделать

int* iterations_device, * iterations_host;

void recreateSurface(SDL_Surface* surface)
{
  double dx = (x_fin - x_start) / (width - 1);
  double dy = (y_fin - y_start) / (height - 1);
  mandelbrot(iterations_device, x_start, y_fin, dx, dy, width, height);
  cudaDeviceSynchronize();

  cudaMemcpy(iterations_host, iterations_device, width * height * sizeof(int), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < height; i++)
  {
    for (int j = 0; j < width; j++)
    {
      int value = iterations_host[i * width + j];

      if (value == 100)
      {
        set_pixel(surface, i, j, get_color(surface, { 0, 0, 0, 255 }));
      }
      else
      {
        //Uint8 r = 10.0 / std::min(255, std::max(value, 0));
        //Uint8 g = 100.0 / std::min(255, std::max(value, 0));
        //Uint8 b = 1000.0 / std::min(255, std::max(value, 0));
        Uint8 r = value % 8 * 32;
        Uint8 g = value % 16 * 16;
        Uint8 b = value % 32 * 8;

        set_pixel(surface, i, j, get_color(surface, { r, g, b, 255 }));
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
  {
    std::cout << "Bad initialization!";
    return 1;
  }
  std::cout << "Initialized.";

  // Cuda

  cudaMalloc((void**) &iterations_device, width * height * sizeof(int));
  iterations_host = new int[width * height];

  // End Cuda


  SDL_Window* window = SDL_CreateWindow("Mandelbrot", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_SHOWN);
  SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

  // Code

  SDL_Surface* surface = SDL_CreateRGBSurface(0, width, height, 32, RMASK, GMASK, BMASK, AMASK);

  recreateSurface(surface);

  SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);

  SDL_Event event;
  bool running = true;
  while (running)
  {
    if (SDL_PollEvent(&event))
    {
      switch (event.type)
      {
      case SDL_QUIT:
        running = false;
        break;
      case SDL_MOUSEBUTTONDOWN:
        if (event.button.button == SDL_BUTTON_LEFT)
        {
          int mouseX, mouseY;
          SDL_GetMouseState(&mouseX, &mouseY);

          double x[2]{ x_start, x_fin };
          double y[2]{ y_start, y_fin };

          double delX = (mouseX - 1.f * width / 2.0) / width * (x[1] - x[0]);
          double delY = (mouseY - 1.f * height / 2.0) / height * (y[1] - y[0]);
          x_start += delX;
          x_fin += delX;
          y_start -= delY;
          y_fin -= delY;

          double scaleX = ((x_fin - x_start) - (x_fin - x_start) / scaling) / 2.0;
          double scaleY = ((y_fin - y_start) - (y_fin - y_start) / scaling) / 2.0;

          x_start += scaleX;
          y_start += scaleY;
          x_fin -= scaleX;
          y_fin -= scaleY;

          scale_pos *= scaling;
          recreateSurface(surface);
          SDL_DestroyTexture(texture);
          texture = SDL_CreateTextureFromSurface(renderer, surface);
        }
        break;
      }
    }

    SDL_RenderClear(renderer);

    SDL_Rect rect = { 0, 0, width, height };
    SDL_RenderCopy(renderer, texture, NULL, &rect);

    SDL_RenderPresent(renderer);
  }

  // End code

  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  SDL_Quit();


  // Free the device memory

  cudaFree(iterations_device);

  // Clean up
  delete[] iterations_host;

  return 0;
}
