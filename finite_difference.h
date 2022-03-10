
// This file is free software. You can use it, redistribute it, and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.

#ifndef finite_difference_h
#define finite_difference_h

#include <vector>

#include "sparse_matrix.h"
#include "vector.h"

template <int dim, typename Number>
class DifferenceOperator
{
public:
  static_assert(dim > 0 && dim < 4, "Only dimensions 1 to 3 implemented");

  DifferenceOperator(const std::vector<Number> &          lower_left,
                     const std::vector<Number> &          upper_right,
                     const std::array<std::size_t, dim> & n_points_per_dim,
                     const std::array<unsigned int, dim> &domain_partitions,
                     const MPI_Comm                       communicator) :
    communicator(communicator),
    n_mpi_ranks(get_n_mpi_ranks(communicator)),
    my_mpi_rank(get_my_mpi_rank(communicator))
  {
    if (lower_left.size() != dim || upper_right.size() != dim)
      {
        std::cout << "Invalid size" << std::endl;
        std::abort();
      }
    for (unsigned int d = 0; d < dim; ++d)
      {
        this->domain_partitions[d] = domain_partitions[d];
        sizes[d]                   = n_points_per_dim[d] / domain_partitions[d];
        h2_inv[d] =
          (n_points_per_dim[d] + 1) * (n_points_per_dim[d] + 1) /
          ((upper_right[d] - lower_left[d]) * (upper_right[d] - lower_left[d]));
      }
    for (unsigned int d = dim; d < 3; ++d)
      {
        sizes[d]                   = 1;
        h2_inv[d]                  = 0;
        this->domain_partitions[d] = 1;
      }
    strides[0] = 1;
    for (unsigned int d = 1; d < 3; ++d)
      strides[d] = strides[d - 1] * sizes[d - 1];

    // initialize the fields for data to send and receive as ghost data
    std::array<unsigned int, 2 *dim> mpi_neighbors = get_mpi_neighbors();
    unsigned int                     count         = 0;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int side = 0; side < 2; ++side)
        if (mpi_neighbors[2 * d + side] < n_mpi_ranks)
          {
            std::size_t n_values = 1;
            for (unsigned int e = 0; e < dim; ++e)
              if (d != e)
                n_values *= sizes[e];
            data_to_send[2 * d + side].resize(n_values);
            ghost_data[2 * d + side].resize(n_values);
            ++count;
          }
  }

  std::array<unsigned int, dim> get_my_domain_position() const
  {
    std::array<unsigned int, dim> my_pos              = {};
    unsigned int                  n_domains_lower_dim = 1;
    for (unsigned int d = 0; d < dim; ++d)
      {
        my_pos[d] = (my_mpi_rank / n_domains_lower_dim) % domain_partitions[d];
        n_domains_lower_dim *= domain_partitions[d];
      }
    return my_pos;
  }

  std::array<unsigned int, 2 * dim> get_mpi_neighbors() const
  {
    const std::array<unsigned int, dim> position = get_my_domain_position();
    std::array<unsigned int, 2 * dim>   neighbors;
    unsigned int                        stride = 1;
    for (unsigned int d = 0; d < dim; ++d)
      {
        // check for neighbors on the left of the current proc
        if (position[d] > 0)
          neighbors[2 * d] = my_mpi_rank - stride;
        else
          neighbors[2 * d] = static_cast<unsigned int>(-1);

        // check for neighbors on the right of the current proc
        if (position[d] + 1 < domain_partitions[d])
          neighbors[2 * d + 1] = my_mpi_rank + stride;
        else
          neighbors[2 * d + 1] = static_cast<unsigned int>(-1);

        stride *= domain_partitions[d];
      }
    return neighbors;
  }

  void apply(const Vector<Number> &src, Vector<Number> &dst) const
  {
#ifdef HAVE_MPI
    std::array<unsigned int, 2 *dim> mpi_neighbors = get_mpi_neighbors();

    unsigned int count = 0;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int side = 0; side < 2; ++side)
        if (mpi_neighbors[2 * d + side] < n_mpi_ranks)
          ++count;

    mpi_requests.resize(2 * count);
    MPI_Request *mpi_requests_ptr = mpi_requests.data();

    // start non-blocking MPI receives
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int side = 0; side < 2; ++side)
        if (mpi_neighbors[2 * d + side] < n_mpi_ranks)
          MPI_Irecv(ghost_data[2 * d + side].data(),
                    ghost_data[2 * d + side].size() * sizeof(Number),
                    MPI_BYTE,
                    mpi_neighbors[2 * d + side],
                    /*mpi_tag*/ 23,
                    communicator,
                    mpi_requests_ptr++);
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int side = 0; side < 2; ++side)
        if (mpi_neighbors[2 * d + side] < n_mpi_ranks)
          {
            // pack data before sending
            unsigned int my_size[2];
            unsigned int my_stride[2];
            for (unsigned int e = 0, count = 0; e < 3; ++e)
              if (d != e)
                {
                  my_size[count]   = sizes[e];
                  my_stride[count] = 1;
                  for (unsigned int f = 0; f < e; ++f)
                    my_stride[count] *= sizes[f];
                  ++count;
                }
            if (my_size[0] * my_size[1] != data_to_send[2 * d + side].size())
              {
                std::cout << "Internal error with dimension mismatch!"
                          << std::endl;
                std::abort();
              }
            unsigned int offset = side * (sizes[d] - 1);
            for (unsigned int f = 0; f < d; ++f)
              offset *= sizes[f];
#  pragma omp parallel for collapse(2)
            for (unsigned int iy = 0; iy < (dim > 2 ? my_size[1] : 1); ++iy)
              for (unsigned int ix = 0; ix < (dim > 1 ? my_size[0] : 1); ++ix)
                data_to_send[2 * d + side][iy * my_size[0] + ix] =
                  src(iy * my_stride[1] + ix * my_stride[0] + offset);

            // send the packed data via MPI
            MPI_Isend(data_to_send[2 * d + side].data(),
                      data_to_send[2 * d + side].size() * sizeof(Number),
                      MPI_BYTE,
                      mpi_neighbors[2 * d + side],
                      /*mpi_tag*/ 23,
                      communicator,
                      mpi_requests_ptr++);
          }
#endif

          // apply the finite difference stencil on the local domain (without
          // considering the ghost data yet)
#pragma omp parallel for collapse(2)
    for (unsigned int iz = 0; iz < (dim > 2 ? sizes[2] : 1); ++iz)
      for (unsigned int iy = 0; iy < (dim > 1 ? sizes[1] : 1); ++iy)
        {
          std::size_t  base_idx      = iz * sizes[0] * sizes[1] + iy * sizes[0];
          const Number hx            = h2_inv[0];
          const Number hy            = h2_inv[1];
          const Number hz            = h2_inv[2];
          const Number hdiag         = 2. * (hx + hy + hz);
          Number *__restrict dst_ptr = &dst(base_idx);

          // peel off the first loop iteration in x
          dst_ptr[0] = hx * (2. * src(base_idx) - src(base_idx + 1));
          if (dim > 1)
            dst_ptr[0] +=
              hy *
              (2. * src(base_idx) - (iy > 0 ? src(base_idx - strides[1]) : 0.) -
               (iy < sizes[1] - 1 ? src(base_idx + strides[1]) : 0.));
          if (dim > 2)
            dst_ptr[0] +=
              hz *
              (2. * src(base_idx) - (iz > 0 ? src(base_idx - strides[2]) : 0.) -
               (iz < sizes[2] - 1 ? src(base_idx + strides[2]) : 0.));

          // regular loop
          if ((dim < 3 || (iz > 0 && iz < sizes[2] - 1)) &&
              (dim < 2 || (iy > 0 && iy < sizes[1] - 1)))
#pragma omp simd
            for (unsigned int i = base_idx + 1; i < base_idx + sizes[0] - 1;
                 ++i)
              {
                dst(i) = hdiag * src(i) - hx * (src(i + 1) + src(i - 1));
                if (dim > 1)
                  dst(i) -= hy * (src(i + strides[1]) + src(i - strides[1]));
                if (dim > 2)
                  dst(i) -= hz * (src(i + strides[2]) + src(i - strides[2]));
              }

          // loop where some part is cut away due to Dirichlet boundary
          // conditions. we move it out of the regular loop to avoid overhead
          // of if statements and make the loop vectorizable
          else
            {
              for (unsigned int ix = 1; ix < sizes[0] - 1; ++ix)
                dst_ptr[ix] =
                  hx * (2 * src(base_idx + ix) - src(base_idx + ix + 1) -
                        src(base_idx + ix - 1));
              if (dim >= 2 && (iy == 0 || iy == sizes[1] - 1))
                {
                  const long long stride = (iy == 0) ? strides[1] : -strides[1];
                  for (unsigned int ix = 1; ix < sizes[0] - 1; ++ix)
                    dst_ptr[ix] += hy * (2. * src(base_idx + ix) -
                                         src(base_idx + ix + stride));
                }
              else if (dim >= 2)
                for (unsigned int ix = 1; ix < sizes[0] - 1; ++ix)
                  dst_ptr[ix] += hy * (2. * src(base_idx + ix) -
                                       src(base_idx + ix + strides[1]) -
                                       src(base_idx + ix - strides[1]));
              if (dim == 3 && (iz == 0 || iz == sizes[2] - 1))
                {
                  const long long stride = (iz == 0) ? strides[2] : -strides[2];
                  for (unsigned int ix = 1; ix < sizes[0] - 1; ++ix)
                    dst_ptr[ix] += hz * (2. * src(base_idx + ix) -
                                         src(base_idx + ix + stride));
                }
              else if (dim == 3)
                for (unsigned int ix = 1; ix < sizes[0] - 1; ++ix)
                  dst_ptr[ix] += hz * (2. * src(base_idx + ix) -
                                       src(base_idx + ix + strides[2]) -
                                       src(base_idx + ix - strides[2]));
            }

          // peel off the last loop iteration in x
          const unsigned int ix = sizes[0] - 1;
          dst_ptr[ix] = hx * (2. * src(base_idx + ix) - src(base_idx - 1 + ix));
          if (dim > 1)
            dst_ptr[ix] +=
              hy * (2. * src(base_idx + ix) -
                    (iy > 0 ? src(base_idx - strides[1] + ix) : 0.) -
                    (iy < sizes[1] - 1 ? src(base_idx + strides[1] + ix) : 0.));
          if (dim > 2)
            dst_ptr[ix] +=
              hz * (2. * src(base_idx + ix) -
                    (iz > 0 ? src(base_idx - strides[2] + ix) : 0.) -
                    (iz < sizes[2] - 1 ? src(base_idx + strides[2] + ix) : 0.));
        }

#ifdef HAVE_MPI

    // finalize the MPI data exchange, i.e., wait for the non-blocking sends
    // to complete
    MPI_Waitall(mpi_requests.size(), mpi_requests.data(), MPI_STATUSES_IGNORE);

    // apply the remaining part of the difference stencil on the ghost data
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int side = 0; side < 2; ++side)
        if (mpi_neighbors[2 * d + side] < n_mpi_ranks)
          {
            unsigned int my_size[2];
            unsigned int my_stride[2];
            for (unsigned int e = 0, count = 0; e < 3; ++e)
              if (d != e)
                {
                  my_size[count]   = sizes[e];
                  my_stride[count] = 1;
                  for (unsigned int f = 0; f < e; ++f)
                    my_stride[count] *= sizes[f];
                  ++count;
                }
            unsigned int offset = side * (sizes[d] - 1);
            for (unsigned int f = 0; f < d; ++f)
              offset *= sizes[f];
#  pragma omp parallel for collapse(2)
            for (unsigned int iy = 0; iy < (dim > 2 ? my_size[1] : 1); ++iy)
              for (unsigned int ix = 0; ix < (dim > 1 ? my_size[0] : 1); ++ix)
                dst(iy * my_stride[1] + ix * my_stride[0] + offset) -=
                  h2_inv[d] * ghost_data[2 * d + side][iy * my_size[0] + ix];
          }
#endif
  }

  SparseMatrix<Number> fill_sparse_matrix() const
  {
    std::vector<unsigned int> row_lengths(sizes[0] * sizes[1] * sizes[2]);
#pragma omp parallel for collapse(2)
    for (unsigned int iz = 0; iz < (dim > 2 ? sizes[2] : 1); ++iz)
      for (unsigned int iy = 0; iy < (dim > 1 ? sizes[1] : 1); ++iy)
        for (unsigned int ix = 0; ix < sizes[0]; ++ix)
          row_lengths[(iz * sizes[1] + iy) * sizes[0] + ix] =
            4 + (ix > 0 && ix < sizes[0] - 1) + (iy > 0 && iy < sizes[1] - 1) +
            (iz > 0 && iz < sizes[2] - 1);

    SparseMatrix<Number> sparse(row_lengths, MemorySpace::Host, communicator);
#pragma omp parallel
    {
      std::vector<unsigned int> col_indices;
      std::vector<Number>       values;
#pragma omp for collapse(2)
      for (unsigned int iz = 0; iz < (dim > 2 ? sizes[2] : 1); ++iz)
        for (unsigned int iy = 0; iy < (dim > 1 ? sizes[1] : 1); ++iy)
          for (unsigned int ix = 0; ix < sizes[0]; ++ix)
            {
              const std::size_t index = (iz * sizes[1] + iy) * sizes[0] + ix;
              col_indices.resize(row_lengths[index]);
              values.resize(row_lengths[index]);
              unsigned int count = 0;
              if (iz > 0)
                {
                  col_indices[count] = index - strides[2];
                  values[count++]    = -h2_inv[2];
                }
              if (iy > 0)
                {
                  col_indices[count] = index - strides[1];
                  values[count++]    = -h2_inv[1];
                }
              if (ix > 0)
                {
                  col_indices[count] = index - 1;
                  values[count++]    = -h2_inv[0];
                }
              col_indices[count] = index;
              values[count++] =
                2. * h2_inv[0] +
                (dim > 1 ? (2. * h2_inv[1] + (dim > 2 ? 2. * h2_inv[2] : 0.)) :
                           0.);
              if (ix < sizes[0] - 1)
                {
                  col_indices[count] = index + 1;
                  values[count++]    = -h2_inv[0];
                }
              if (iy < sizes[1] - 1)
                {
                  col_indices[count] = index + strides[1];
                  values[count++]    = -h2_inv[1];
                }
              if (iz < sizes[2] - 1)
                {
                  col_indices[count] = index + strides[2];
                  values[count++]    = -h2_inv[2];
                }
              sparse.add_row(index, col_indices, values);
            }
    }

    // for the off-processor part, we need to specify the vector entries we
    // want to send to individual processors; here we again assume symmetry
    // between what gets sent and what is received
    std::size_t n_ghost_entries = 0;
    for (unsigned int d = 0; d < 2 * dim; ++d)
      n_ghost_entries += data_to_send[d].size();
    sparse.allocate_ghost_data_memory(n_ghost_entries);
    std::array<unsigned int, 2 *dim> mpi_neighbors = get_mpi_neighbors();
    n_ghost_entries                                = 0;
    std::vector<std::pair<unsigned int, std::vector<unsigned int>>>
                                                       send_indices;
    std::vector<std::pair<unsigned int, unsigned int>> receive_indices;
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int side = 0; side < 2; ++side)
        if (mpi_neighbors[2 * d + side] < n_mpi_ranks)
          {
            send_indices.push_back({});
            send_indices.back().first = mpi_neighbors[2 * d + side];
            receive_indices.emplace_back(mpi_neighbors[2 * d + side],
                                         data_to_send[2 * d + side].size());
            unsigned int my_size[2];
            unsigned int my_stride[2];
            for (unsigned int e = 0, count = 0; e < 3; ++e)
              if (d != e)
                {
                  my_size[count]   = sizes[e];
                  my_stride[count] = 1;
                  for (unsigned int f = 0; f < e; ++f)
                    my_stride[count] *= sizes[f];
                  ++count;
                }
            unsigned int offset = side * (sizes[d] - 1);
            for (unsigned int f = 0; f < d; ++f)
              offset *= sizes[f];

            for (unsigned int iy = 0; iy < (dim > 2 ? my_size[1] : 1); ++iy)
              for (unsigned int ix = 0; ix < (dim > 1 ? my_size[0] : 1); ++ix)
                {
                  send_indices.back().second.push_back(
                    iy * my_stride[1] + ix * my_stride[0] + offset);
                  // utilize that each off-proc entry is accessed exactly once
                  sparse.add_ghost_entry(iy * my_stride[1] + ix * my_stride[0] +
                                           offset,
                                         n_ghost_entries++,
                                         -h2_inv[d]);
                }
          }
    sparse.set_send_and_receive_information(send_indices, receive_indices);

    return sparse;
  }

private:
  const MPI_Comm     communicator;
  const unsigned int n_mpi_ranks;
  const unsigned int my_mpi_rank;
  Number             h2_inv[dim];
  std::size_t        sizes[3];
  std::size_t        strides[3];
  unsigned int       domain_partitions[3];

  mutable std::array<std::vector<Number>, 2 * dim> data_to_send;
  mutable std::array<std::vector<Number>, 2 * dim> ghost_data;
#ifdef HAVE_MPI
  mutable std::vector<MPI_Request> mpi_requests;
#endif
};

#endif
