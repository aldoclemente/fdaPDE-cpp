// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <gtest/gtest.h> // testing framework
// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>

// #include "src/srpde_test.cpp"
// #include "src/strpde_test.cpp"
// #include "src/gsrpde_test.cpp"
// #include "src/qsrpde_test.cpp"
// #include "src/gcv_srpde_test.cpp"
// #include "src/gcv_qsrpde_test.cpp"
// #include "src/gcv_srpde_newton_test.cpp"
// #include "src/density_estimation_test.cpp"
// #include "src/kcv_srpde_test.cpp"
// #include "src/fpca_test.cpp"
// #include "src/fpls_test.cpp"
// #include "src/centering_test.cpp"

int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
