#include "biharmonic_solve.h"
#include <igl/min_quad_with_fixed.h>

void biharmonic_solve(
    const igl::min_quad_with_fixed_data<double> &data,
    const Eigen::MatrixXd &bc,
    Eigen::MatrixXd &D)
{
  // set Beq, B
  Eigen::VectorXd Beq, B;

  B = Eigen::VectorXd::Zero(data.n);
  igl::min_quad_with_fixed_solve(data, B, bc, Beq, D);
}
