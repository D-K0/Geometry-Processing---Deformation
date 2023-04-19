#include "../include/arap_single_iteration.h"
#include <igl/polar_svd3x3.h>
#include <igl/min_quad_with_fixed.h>

void arap_single_iteration(
    const igl::min_quad_with_fixed_data<double> &data,
    const Eigen::SparseMatrix<double> &K,
    const Eigen::MatrixXd &bc,
    Eigen::MatrixXd &U)
{
  // set C and R
  Eigen::MatrixXd C = K.transpose() * U;
  Eigen::MatrixXd R(3 * data.n, 3);
  Eigen::MatrixXd Beq;

  // fill R
  for (int k = 0; k < data.n; k++)
  {
    Eigen::Matrix3d Rk;
    Eigen::Matrix3d Ck = C.block(3 * k, 0, 3, 3);
    igl::polar_svd3x3(Ck, Rk);
    R.block(3 * k, 0, 3, 3) = Rk;
  }
  igl::min_quad_with_fixed_solve(data, K * R, bc, Beq, U);
}
