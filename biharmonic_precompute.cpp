#include "biharmonic_precompute.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>

void biharmonic_precompute(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::VectorXi &b,
    igl::min_quad_with_fixed_data<double> &data)
{
  Eigen::SparseMatrix<double> L, M, Minv, Q, Aeq;

  // set L:
  igl::cotmatrix(V, F, L);
  // Set M:
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);
  // invert M
  igl::invert_diag(M, Minv);
  // set Q
  Q = L.transpose() * Minv * L;

  igl::min_quad_with_fixed_precompute(Q, b, Aeq, false, data);
}
