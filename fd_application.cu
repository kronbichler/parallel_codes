
// This file is free software. You can use it, redistribute it, and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

#include "vector.h"
#include "finite_difference.h"
#include "sparse_matrix.h"
#include "conjugate_gradient.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

template <typename Number>
void run_test(const long long N_x_given,
              const long long N_y_given,
              const long long N_z_given,
              const long long n_repeat,
              const bool do_sparse)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  const unsigned int n_mpi_ranks = get_n_mpi_ranks(communicator);
  const unsigned int my_mpi_rank = get_my_mpi_rank(communicator);

  // find subdivision of domain into suitable sub-blocks close to a cube. to
  // this end we find the prime factors of the number of ranks
  std::array<unsigned int,3> domain_partitions;
  {
    std::vector<unsigned int> prime_factors;
    unsigned int n = n_mpi_ranks;
    while (n%2 == 0)
      {
        prime_factors.push_back(2);
        n /= 2;
      }
    for (unsigned int i=3; i<=std::sqrt(n); i+=2)
        while (n%i == 0)
          {
            prime_factors.push_back(i);
            n /= i;
          }

    if (n > 2)
      prime_factors.push_back(n);

    for (unsigned int d=0; d<3; ++d)
      domain_partitions[d] = 1;
    for (unsigned int i=0; i<prime_factors.size(); ++i)
      domain_partitions[i%3] *= prime_factors[i];
  }

  const std::size_t N_x_local = N_x_given / domain_partitions[0];
  const std::size_t N_y_local = (N_y_given > 0 ? N_y_given :
                                 N_x_given) / domain_partitions[1];
  const std::size_t N_z_local = (N_z_given > 0 ? N_z_given :
                                 N_x_given) / domain_partitions[2];

  const std::size_t N_x = N_x_local * domain_partitions[0];
  const std::size_t N_y = N_y_local * domain_partitions[1];
  const std::size_t N_z = N_z_local * domain_partitions[2];

#ifdef _OPENMP
  const unsigned int n_threads = omp_get_max_threads();
#else
  const unsigned int n_threads = 1;
#endif

  if (my_mpi_rank == 0)
    std::cout << "Computing on a " << N_x << " x "
              << N_y << " x " << N_z << " domain with "
              << domain_partitions[0] << " x " << domain_partitions[1]
              << " x " << domain_partitions[2] << " MPI processes and "
              << n_threads << " OpenMP threads" << std::endl;

#ifdef HAVE_MPI
  MPI_Barrier(communicator);
#endif

  if (N_x * N_y * N_z == 0)
    {
      if (my_mpi_rank == 0)
        std::cout << "Domain size is zero due to partitioning, increase Nx/Ny/Nz"
                  << std::endl;
      return;
    }

  std::vector<double> lower_left({0., 0., 0.});
  std::vector<double> upper_right({1., 1., 1.});
  std::array<std::size_t,3> points_per_dim_global({N_x, N_y, N_z});
  DifferenceOperator<3, Number> difference_stencil(lower_left, upper_right,
                                                   points_per_dim_global,
                                                   domain_partitions,
                                                   communicator);

  const std::size_t local_size = N_x_local * N_y_local * N_z_local;
  Vector<Number> src(N_x * N_y * N_z,
                     std::make_pair(my_mpi_rank * local_size,
                                    (my_mpi_rank+1) * local_size),
                     MemorySpace::Host,
                     communicator);
  Vector<Number> dst(src), result(src);

  constexpr double PI = 3.14159265358979323846;

  const std::array<unsigned int,3> my_mpi_domain_position =
    difference_stencil.get_my_domain_position();

  const std::size_t ix_offset = my_mpi_domain_position[0] * N_x_local;
  const std::size_t iy_offset = my_mpi_domain_position[1] * N_y_local;
  const std::size_t iz_offset = my_mpi_domain_position[2] * N_z_local;

#pragma omp parallel for collapse(2)
  for (unsigned int iz=0; iz<N_z_local; ++iz)
    for (unsigned int iy=0; iy<N_y_local; ++iy)
      for (unsigned int ix=0; ix<N_x_local; ++ix)
        src((iz*N_y_local+iy)*N_x_local+ix) =
          std::sin(PI * static_cast<double>(ix_offset + ix + 1) / (N_x+1)) *
          std::sin(PI * static_cast<double>(iy_offset + iy + 1) / (N_y+1)) *
          std::sin(PI * static_cast<double>(iz_offset + iz + 1) / (N_z+1));

  difference_stencil.apply(src, dst);

  double error = 0;
#pragma omp parallel for
  for (unsigned int iz=0; iz<N_z_local; ++iz)
    {
      double my_error = 0;
      for (unsigned int iy=0; iy<N_y_local; ++iy)
        for (unsigned int ix=0; ix<N_x_local; ++ix)
          {
            double local_error = PI * PI * 3 *
              std::sin(PI * static_cast<double>(ix_offset + ix + 1) / (N_x+1)) *
              std::sin(PI * static_cast<double>(iy_offset + iy + 1) / (N_y+1)) *
              std::sin(PI * static_cast<double>(iz_offset + iz + 1) / (N_z+1)) -
              dst(iz*N_y_local*N_x_local+iy*N_x_local+ix);
            my_error += local_error * local_error / ((N_x+1) * (N_y+1) * (N_z+1));
          }
#pragma omp critical
      error += my_error;
    }
  {
    const double global_error = std::sqrt(mpi_sum(error,communicator));
    if (my_mpi_rank == 0)
      std::cout << "Discretization error FD stencil: " << global_error
                << std::endl;
  }

  {
    const auto t1 = std::chrono::steady_clock::now();
    for (unsigned long long rep=0; rep<n_repeat; ++rep)
      difference_stencil.apply(src, dst);

    const double time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - t1).count();

    if (my_mpi_rank == 0)
      std::cout << "FD stencil of size " << src.size()
                << ": " << time/n_repeat << " seconds or "
                << std::setw(8) << 1e-6 * src.size() * n_repeat / time
                << " MUPD/s" << std::endl;
  }

  {
    // somewhat modify the initial condition to not let CG converge
    // immediately
    if (my_mpi_rank == 0)
      {
        result(0) = 1.;
        result(1) = 0.8;
      }

    const auto t1 = std::chrono::steady_clock::now();
    const auto info =
      solve_with_conjugate_gradient(500, 1e-12, difference_stencil,
                                    dst, result);

    const double time =
      std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::steady_clock::now() - t1).count();

    if (my_mpi_rank == 0)
      std::cout << "Conjugate gradient solve of size " << src.size()
                << " in " << info.first << " iterations: " << time << " seconds or "
                << std::setw(8) << 1e-6 * N_x * N_y * N_z * info.first / time
                << " MUPD/s/it" << std::endl;
    result.add(-1., src);
    const double l2_norm = result.l2_norm();
    if (my_mpi_rank == 0)
      std::cout << "Error conjugate gradient solve: " << l2_norm << std::endl;
  }

  if (do_sparse)
    {
      const SparseMatrix<Number> sparse_matrix_host = difference_stencil.fill_sparse_matrix();
      // create matrix on device
      SparseMatrix<Number> sparse_matrix = sparse_matrix_host.copy_to_device();
      Vector<Number> src_device = src.copy_to_device();
      Vector<Number> dst_device = dst.copy_to_device();

      sparse_matrix.apply(src_device, dst_device);

      dst = dst_device.copy_to_host();

      double error = 0;
#pragma omp parallel for
      for (unsigned int iz=0; iz<N_z_local; ++iz)
        {
          double my_error = 0;
          for (unsigned int iy=0; iy<N_y_local; ++iy)
            for (unsigned int ix=0; ix<N_x_local; ++ix)
              {
                double local_error = PI * PI * 3 *
                  std::sin(PI * static_cast<double>(ix_offset + ix + 1) / (N_x+1)) *
                  std::sin(PI * static_cast<double>(iy_offset + iy + 1) / (N_y+1)) *
                  std::sin(PI * static_cast<double>(iz_offset + iz + 1) / (N_z+1)) -
                  dst(iz*N_y_local*N_x_local+iy*N_x_local+ix);
                my_error += local_error * local_error / ((N_x+1) * (N_y+1) * (N_z+1));
              }
#pragma omp critical
          error += my_error;
        }

      {
        const double global_error = std::sqrt(mpi_sum(error,communicator));
        if (my_mpi_rank == 0)
          std::cout << "Discretization error sparse matrix: "
                    << global_error << std::endl;
      }

      {
        const auto t1 = std::chrono::steady_clock::now();
        for (unsigned long long rep=0; rep<n_repeat; ++rep)
          sparse_matrix.apply(src, dst);

        // make sure to finish all GPU kernels before measuring again
        cudaDeviceSynchronize();

        const double time =
          std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t1).count();

        if (my_mpi_rank == 0)
          std::cout << "Sparse matrix of size " << src.size()
                    << " : " << time/n_repeat << " seconds or "
                    << std::setw(8) << 1e-6 * src.size() * n_repeat / time
                    << " MUPD/s or "
                    << 1e-9 * n_repeat * (sparse_matrix.n_nonzero_entries() * 12 +
                                          dst.size() * 4 * 8) / time
                    << " GB/s" << std::endl;
      }

      {
        // somewhat modify the initial condition to not let CG converge
        // immediately
        result = 0;
        if (my_mpi_rank == 0)
          {
            result(0) = 1.;
            result(1) = 0.8;
          }
        Vector<Number> result_device = result.copy_to_device();

        const auto t1 = std::chrono::steady_clock::now();
        const auto info =
          solve_with_conjugate_gradient(500, 1e-12, sparse_matrix,
                                        dst_device, result_device);

        const double time =
          std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t1).count();

        if (my_mpi_rank == 0)
          std::cout << "Conjugate gradient solve of size " << src.size()
                    << " in " << info.first << " iterations: "
                    << time << " seconds or "
                    << std::setw(8) << 1e-6 * src.size() * info.first / time
                    << " MUPD/s/it" << std::endl;
        result = result_device.copy_to_host();
        result.add(-1., src);
        const double l2_norm = result.l2_norm();

        if (my_mpi_rank == 0)
          std::cout << "Error conjugate gradient solve: " << l2_norm << std::endl;
      }
    }
}


int main(int argc, char **argv)
{
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  long long N_x = -1, N_y = -1, N_z = -1;
  long long n_repeat = -1;
  std::string number = "double";
  bool do_sparse = false;
  const unsigned int my_mpi_rank = get_my_mpi_rank(MPI_COMM_WORLD);

  if (argc % 2 == 0)
    {
      if (my_mpi_rank == 0)
        std::cout << "Error, expected odd number of common line arguments"
                  << std::endl
                  << "Expected line of the form"
                  << std::endl
                  << "-Nx 100 -Ny 100 -Nz 100 -repeat 100 -number double -sparse 0"
                  << std::endl;
      std::abort();
    }

  // parse from the command line
  for (unsigned l=1; l<argc; l+=2)
    {
      std::string option = argv[l];
      if (option == "-Nx")
        N_x = std::atoll(argv[l+1]);
      else if (option == "-Ny")
        N_y = std::atoll(argv[l+1]);
      else if (option == "-Nz")
        N_z = std::atoll(argv[l+1]);
      else if (option == "-repeat")
        n_repeat = std::atoll(argv[l+1]);
      else if (option == "-number")
        number = argv[l+1];
      else if (option == "-sparse")
        do_sparse = std::atoi(argv[l+1]);
      else if (my_mpi_rank == 0)
        std::cout << "Unknown option " << option << " - ignored!" << std::endl;
    }

  if (N_x == -1)
    for (unsigned long long NN=24; NN<500; NN+=24)
      {
        n_repeat = std::max(2ULL, 10000000000ULL/(NN*NN*NN));
        run_test<double>(NN, NN, NN, n_repeat, do_sparse);
      }
  else
    {
      run_test<double>(N_x, N_y, N_z, n_repeat, do_sparse);
    }

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
}
