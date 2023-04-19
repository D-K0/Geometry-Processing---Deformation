#include "../include/arap_precompute.h"
#include <igl/min_quad_with_fixed.h>
// #include <igl/arap_linear_block.h>   //black listed
#include <igl/cotmatrix.h>

void arap_precompute(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const Eigen::VectorXi & b,
  igl::min_quad_with_fixed_data<double> & data,
  Eigen::SparseMatrix<double> & K)
{
  // set Eigen::Triplet
  typedef Eigen::Triplet<double> T;
  std::vector<T> triplets;
  triplets.reserve(F.rows() * 3 * 3 * 2);

  // set L, Aeq
  Eigen::SparseMatrix<double> L, Aeq;

  igl::cotmatrix(V, F, L);
  igl::min_quad_with_fixed_precompute(L, b, Aeq, false, data);

  // set C
  Eigen::MatrixXd C;
  igl::cotmatrix_entries(V, F, C);

  int i, j;
  int Vi, Vj, Vk;

  // solve K
  for (int f = 0; f < F.rows(); f++)
    for (int oppV = 0; oppV < 3; oppV++)
    {
      i = (oppV + 1) % 3;
      j = (oppV + 2) % 3;
      Vi = F(f, i);
      Vj = F(f, j);

      Eigen::RowVector3d e_ij = C(f, oppV) * (V.row(Vi) - V.row(Vj)) / 3.;
      for (int k = 0; k < 3; k++)
      {
        int V_k = F(f, k);
        for (int beta = 0; beta < 3; beta++)
        {
          triplets.push_back(T(Vi, 3 * V_k + beta, e_ij(beta)));
          triplets.push_back(T(Vj, 3 * V_k + beta, -1. * e_ij(beta)));
        }
      }
    }
  K.resize(V.rows(), 3 * V.rows());
  K.setFromTriplets(triplets.begin(), triplets.end());
}
