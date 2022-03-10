
// This file is free software. You can use it, redistribute it, and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vectorization.h"

// Separate function to compute the daxpy from three given pointers
void compute_daxpy(const unsigned int N,
                   const double       a,
                   const double *     x,
                   const double *     y,
                   double *           z)
{
  constexpr unsigned int simd_width = VectorizedArray<double>::size();
  // round N down to nearest multiple of SIMD width
  const unsigned int N_regular = N / simd_width * simd_width;
  for (unsigned int i = 0; i < N_regular; i += simd_width)
    {
      VectorizedArray<double> x_vec, y_vec;
      x_vec.load(x + i);
      y_vec.load(y + i);
      const VectorizedArray<double> z_vec = a * x_vec + y_vec;
      z_vec.store(z + i);
    }

  // remainder
  for (unsigned int i = N_regular; i < N; ++i)
    z[i] = a * x[i] + y[i];
}


// Run the actual benchmark
void benchmark_daxpy(const unsigned int N)
{
  std::vector<double> v1(N), v2(N), v3(N);
  for (unsigned int i = 0; i < N; ++i)
    v1[i] = static_cast<double>(rand()) / RAND_MAX;
  for (unsigned int i = 0; i < N; ++i)
    v2[i] = static_cast<double>(rand()) / RAND_MAX;

  const unsigned int n_tests  = 20;
  const unsigned int n_repeat = std::max(1U, 10000000U / N);
  double             best = 1e10, worst = 0, avg = 0;
  for (unsigned int t = 0; t < n_tests; ++t)
    {
      // type of t1: std::chrono::steady_clock::time_point
      const auto t1 = std::chrono::steady_clock::now();

      // declare variable volatile to prevent the compiler from removing the
      // unused result and the associated loop altogether
      for (unsigned int rep = 0; rep < n_repeat; ++rep)
        compute_daxpy(N, 13, v1.data(), v2.data(), v3.data());
      const volatile double result = v3[0] + v3.back();

      // prevent compiler to report about an unused variable
      (void)result;

      // measure the time by taking the difference between the time point
      // before starting and now
      const double time =
        std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - t1)
          .count();

      best  = std::min(best, time / n_repeat);
      worst = std::max(worst, time / n_repeat);
      avg += time / n_repeat;
    }

  std::cout << "daxpy of size " << std::setw(8) << N
            << " : min/avg/max: " << std::setw(11) << best << " "
            << std::setw(11) << avg / n_tests << " " << std::setw(11) << worst
            << " seconds or " << std::setw(8) << 1e-6 * N / best
            << " MUPD/s or " << std::setw(8) << 1e-9 * 3 * 8 * N / best
            << " GB/s" << std::endl;
}

int main(int argc, char **argv)
{
  int N = -1;
  if (argc > 1)
    N = std::atoi(argv[1]);

  const unsigned int n_vect_doubles = VectorizedArray<double>::size();
  const unsigned int n_vect_bits    = 8 * sizeof(double) * n_vect_doubles;

  std::cout << "Explicit vectorization over " << n_vect_doubles
            << " doubles = " << n_vect_bits << " bits" << std::endl;

  if (N == -1)
    {
      for (unsigned int n = 10; n < 100000000; n = n * 1.2)
        benchmark_daxpy(n);
    }
  else if (N > 0)
    benchmark_daxpy(N);

  return 0;
}
