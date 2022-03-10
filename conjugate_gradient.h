
// This file is free software. You can use it, redistribute it, and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.

#ifndef conjugate_gradient_h
#define conjugate_gradient_h

#include <utility>

template <typename Matrix, typename Vector>
std::pair<unsigned int, double>
solve_with_conjugate_gradient(const unsigned int n_iterations,
                              const double relative_tolerance,
                              const Matrix &A,
                              const Vector &b,
                              Vector &x)
{
  Vector r(x);

  A.apply(x, r);
  r.sadd(-1., 1., b);

  Vector p(r), q(r);

  double residual_norm_square = r.norm_square();
  const double initial_residual = std::sqrt(residual_norm_square);
  if (initial_residual < 1e-16)
    return std::make_pair(0U, initial_residual);

  unsigned int it = 0;
  while (it<n_iterations)
    {
      ++it;
      A.apply(p, q);
      const double alpha = residual_norm_square / (p.dot(q));
      x.add(alpha, p);
      r.add(-alpha, q);
      double new_residual_norm_square = r.norm_square();
      if (std::sqrt(new_residual_norm_square) <
          relative_tolerance * initial_residual)
        break;

      const double beta = new_residual_norm_square / residual_norm_square;
      residual_norm_square = new_residual_norm_square;
      p.sadd(beta, 1., r);
    }
  return std::make_pair(it, std::sqrt(residual_norm_square));
}


#endif
