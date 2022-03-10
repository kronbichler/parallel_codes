
// This file is free software. You can use it, redistribute it, and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.

// Example code that implements the stream triad function
// Compile this e.g. with
// gcc -O3 -march=native -fopenmp stream_triad_parallel.cc -o
// stream_triad_parallel and run as "./stream_triad_parallel 100 0

#include <omp.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

// Separate function to compute the daxpy from three given pointers
void compute_daxpy(const unsigned int N,
                   const double       a,
                   const double *     x,
                   const double *     y,
                   double *           z)
{
#pragma omp simd
  for (unsigned int i = 0; i < N; ++i)
    z[i] = a * x[i] + y[i];
}


// Allocate memory and create a memory address divisible by 64 bytes (8
// doubles). Assign memory to different NUMA domains by the first touch on
// right thread
double *get_aligned_vector_pointer(const unsigned int N, std::vector<double> &x)
{
  const unsigned int alignment = 8;
  x.reserve(N + alignment - 1);

  // allocate entries on NUMA nodes
#pragma omp parallel for
  for (unsigned int i = 0; i < N + alignment - 1; ++i)
    x[i] = 0;
  x.resize(N + alignment - 1);

  const std::size_t alignment_offset =
    reinterpret_cast<std::size_t>(x.data()) % (alignment * sizeof(double));
  return x.data() + (alignment - alignment_offset / sizeof(double)) % alignment;
}



// Run the actual benchmark
void benchmark_daxpy(const unsigned int N_wish,
                     const bool         test_multiple_vectors)
{
#ifdef _OPENMP
  const unsigned int n_threads = omp_get_max_threads();
#else
  const unsigned int n_threads = 1;
#endif
  const unsigned int local_size =
    ((N_wish + n_threads - 1) / n_threads + 7) / 8 * 8;
  const unsigned int N = local_size * n_threads;

  const double random = static_cast<double>(rand()) / RAND_MAX;

  const unsigned int n_tests  = 20;
  const unsigned int n_repeat = std::max(1U, 100000000U / N);
  double             best = 1e10, worst = 0, avg = 0;

  if (test_multiple_vectors == false)
    {
      // Case where we work on a single long vector. This is pretty slow on
      // multiple threads because we need to start up loop of small size and
      // because we have NUMA conflicts when different cores work on nearby
      // memory regions (same memory page)
      std::vector<double> v1_vec, v2_vec, v3_vec;
      double *            v1 = get_aligned_vector_pointer(N, v1_vec);
      double *            v2 = get_aligned_vector_pointer(N, v2_vec);
      double *            v3 = get_aligned_vector_pointer(N, v3_vec);

#pragma omp parallel for
      for (unsigned int i = 0; i < N; ++i)
        v1[i] = 13. * i - 32. * random;
#pragma omp parallel for
      for (unsigned int i = 0; i < N; ++i)
        v2[i] = random * i + 1.24;

      for (unsigned int t = 0; t < n_tests; ++t)
        {
          // type of t1: std::chrono::steady_clock::time_point
          const auto t1 = std::chrono::steady_clock::now();

          // declare variable volatile to prevent the compiler from removing the
          // unused result and the associated loop altogether
          if (N < 512)
            {
              for (unsigned int rep = 0; rep < n_repeat; ++rep)
                compute_daxpy(N, 13, v1, v2, v3);
              const volatile double result = v3[0];
              // prevent compiler to report about an unused variable
              (void)result;
            }
          else
            {
#pragma omp parallel for
              for (unsigned int i = 0; i < n_threads; ++i)
                {
                  for (unsigned int rep = 0; rep < n_repeat; ++rep)
                    compute_daxpy(local_size,
                                  13,
                                  v1 + i * local_size,
                                  v2 + i * local_size,
                                  v3 + i * local_size);
                  const volatile double result = v3[0];
                  // prevent compiler to report about an unused variable
                  (void)result;
                }
            }

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
    }
  else
    {
      // Allocate different vectors on different threads - this runs much
      // quicker because we do not synchronize threads at the level of
      // individual updates but run repetitions
#pragma omp parallel
      {
        std::vector<double> v1_vec, v2_vec, v3_vec;
        double *            v1 = get_aligned_vector_pointer(local_size, v1_vec);
        double *            v2 = get_aligned_vector_pointer(local_size, v2_vec);
        double *            v3 = get_aligned_vector_pointer(local_size, v3_vec);

        for (unsigned int i = 0; i < local_size; ++i)
          v1[i] = 13. * i - 32. * random;
        for (unsigned int i = 0; i < local_size; ++i)
          v2[i] = random * i + 1.24;

        for (unsigned int t = 0; t < n_tests; ++t)
          {
            // type of t1: std::chrono::steady_clock::time_point
            std::chrono::steady_clock::time_point t1;
#pragma omp barrier
#pragma omp master
            t1 = std::chrono::steady_clock::now();

            // declare variable volatile to prevent the compiler from removing
            // the unused result and the associated loop altogether
            for (unsigned int rep = 0; rep < n_repeat; ++rep)
              compute_daxpy(local_size, 13, v1, v2, v3);
            const volatile double result = v3[0] + v3[N - 1];

            // prevent compiler to report about an unused variable
            (void)result;

#pragma omp barrier

#pragma omp master
            {
              // measure the time by taking the difference between the time
              // point before starting and now
              const double time =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                  std::chrono::steady_clock::now() - t1)
                  .count();

              best  = std::min(best, time / n_repeat);
              worst = std::max(worst, time / n_repeat);
              avg += time / n_repeat;
            }
          }
      }
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
  int  N                     = -1;
  bool test_multiple_vectors = false;
  if (argc > 1 &&
      (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))
    {
      std::cout << "Usage: ./stream_triad_parallel 1000 0" << std::endl
                << "  First arguments gives vector size (-1 -> test range) "
                << std::endl
                << "  Second arguments specifies whether to test on multiple "
                << "  vectors (1/true) or on a single vector (0/false)"
                << std::endl;
      return 0;
    }
  else if (argc > 1)
    N = std::atoi(argv[1]);
  if (argc > 2)
    test_multiple_vectors = std::atoi(argv[2]);

  if (N == -1)
    {
      for (unsigned int n = 10; n < 100000000; n = n * 1.12)
        benchmark_daxpy(n, test_multiple_vectors);
    }
  else if (N > 0)
    benchmark_daxpy(N, test_multiple_vectors);

  return 0;
}
