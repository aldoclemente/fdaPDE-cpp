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

#include <fdaPDE/core.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>

#include <fdaPDE/core.h>
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::PDE;

#include "../../fdaPDE/models/functional/fpca.h"
#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::FPCA;
using fdapde::models::Sampling;
#include "../../fdaPDE/calibration/symbols.h"
using fdapde::calibration::Calibration;
#include "../../fdaPDE/calibration/off.h"
using fdapde::calibration::Off;
#include "../../fdaPDE/models/functional/center.h"
using fdapde::models::center;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
using fdapde::testing::read_csv;
using fdapde::testing::read_mtx;

// test 1
//    domain:       unit square [1,1] x [1,1]
//    sampling:     locations = nodes
//    penalization: simple laplacian
//    BC:           no
//    order FE:     1
//    missing data: no
//    solver: sequential (power iteration)
// TEST(fpca_test, laplacian_samplingatnodes_sequential) {
//     // define domain
//     MeshLoader<Mesh2D> domain("unit_square");
//     // import data from files
//     DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test1/y.csv");
//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define model
//     double lambda_D = 1e-2;
//     FPCA<SpaceOnly, fdapde::sequential> model(problem, Sampling::mesh_nodes, Calibration::off);
//     model.set_lambda_D(lambda_D);
//     // set model's data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     model.set_data(df);
//     // solve FPCA problem
//     model.init();
//     model.solve();
//     // test correctness
//     EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test1/loadings_seq.mtx"));
//     EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test1/scores_seq.mtx"));
// }

// // test 2
// //    domain:       unit square [1,1] x [1,1]
// //    sampling:     locations = nodes
// //    penalization: simple laplacian
// //    BC:           no
// //    order FE:     1
// //    missing data: no
// //    solver: monolithic (rsvd)
// // TEST(fpca_test, laplacian_samplingatnodes_monolithic) {
// //     // define domain
// //     MeshLoader<Mesh2D> domain("unit_square");
// //     // import data from files
// //     DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test1/y.csv");
// //     // define regularizing PDE
// //     auto L = -laplacian<FEM>();
// //     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
// //     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
// //     // define model
// //     double lambda_D = 1e-2;
// //     FPCA<SpaceOnly, fdapde::monolithic> model(problem, Sampling::mesh_nodes);
// //     model.set_lambda_D(lambda_D);
// //     // set model's data
// //     BlockFrame<double, int> df;
// //     df.insert(OBSERVATIONS_BLK, y);
// //     model.set_data(df);
// //     // solve FPCA problem
// //     model.init();
// //     model.solve();
// //     // test correctness
// //     EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test1/loadings_mon.mtx"));
// //     EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test1/scores_mon.mtx"));
// // }

// // test 3
// //    domain:       unit square [1,1] x [1,1]
// //    sampling:     locations != nodes
// //    penalization: simple laplacian
// //    BC:           no
// //    order FE:     1
// //    missing data: no
// //    solver: sequential (power iteration) + GCV \lambda selection
// TEST(fpca_test, laplacian_samplingatlocations_sequential_gcv) {
//     // define domain
//     MeshLoader<Mesh2D> domain("unit_square");
//     // import data from files
//     DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test2/locs.csv");
//     DMatrix<double> y    = read_csv<double>("../data/models/fpca/2D_test2/y.csv");
//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define model
//     FPCA<SpaceOnly, fdapde::sequential> model(problem, Sampling::pointwise, Calibration::gcv);
//     model.set_spatial_locations(locs);
//     // grid of smoothing parameters
//     std::vector<DVector<double>> lambda_grid;
//     for (double x = -4; x <= -2; x += 0.1) { lambda_grid.push_back(SVector<1>(std::pow(10, x))); }
//     model.set_lambda(lambda_grid);
//     model.set_seed(78965);   // for reproducibility purposes in testing
// TEST(fpca_test, laplacian_samplingatnodes_monolithic) {
//     // define domain
//     MeshLoader<Triangulation<2, 2>> domain("unit_square");
//     // import data from files
//     DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test1/y.csv");
//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define model
//     double lambda_D = 1e-2;
//     FPCA<SpaceOnly> model(problem, Sampling::mesh_nodes, RegularizedSVD<fdapde::monolithic>());
//     model.set_lambda_D(lambda_D);
//     // set model's data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     model.set_data(df);
//     // solve FPCA problem
//     model.init();
//     model.solve();
//     // test correctness

//     EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test2/loadings.mtx"));
//     EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test2/scores.mtx"  ));
// }

// // test 4
// //    domain:       unit square [1,1] x [1,1]
// //    sampling:     locations != nodes
// //    penalization: simple laplacian
// //    BC:           no
// //    order FE:     1
// //    missing data: no
// //    solver: sequential (power iteration) + KCV \lambda selection
// TEST(fpca_test, laplacian_samplingatlocations_sequential_kcv) {
//     // define domain
//     MeshLoader<Mesh2D> domain("unit_square");
//     // import data from files
//     DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test3/locs.csv");
//     DMatrix<double> y    = read_csv<double>("../data/models/fpca/2D_test3/y.csv");
//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define model
//     FPCA<SpaceOnly, fdapde::sequential> model(problem, Sampling::pointwise, Calibration::kcv);
//     model.set_spatial_locations(locs);
//     // grid of smoothing parameters
//     std::vector<DVector<double>> lambdas;
//     for (double x = -4; x <= -2; x += 0.1) lambdas.push_back(SVector<1>(std::pow(10, x)));
//     model.set_lambda(lambdas);
//     model.set_seed(12654);   // for reproducibility purposes in testing
//     model.set_nfolds(10);    // perform a 10 folds cross-validation
//     // set model's data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     model.set_data(df);
//     // solve FPCA problem
//     model.init();
//     model.solve();
//     // test correctness
//     EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test3/loadings.mtx"));
//     EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test3/scores.mtx"  ));
// }
// /*
//     EXPECT_TRUE(almost_equal(model.Psi() * model.loadings(), "../data/models/fpca/2D_test1/loadings_mon.mtx"));
//     EXPECT_TRUE(almost_equal(model.scores(),                 "../data/models/fpca/2D_test1/scores_mon.mtx"  ));
// }
// */
// // // test 3
// // //    domain:       unit square [1,1] x [1,1]
// // //    sampling:     locations != nodes
// // //    penalization: simple laplacian
// // //    BC:           no
// // //    order FE:     1
// // //    missing data: no
// // //    solver: sequential (power iteration) + GCV \lambda selection
// // TEST(fpca_test, laplacian_samplingatlocations_sequential_gcv) {
// //     // define domain
// //     MeshLoader<Triangulation<2, 2>> domain("unit_square");
// //     // import data from files
// //     DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test2/locs.csv");
// //     DMatrix<double> y    = read_csv<double>("../data/models/fpca/2D_test2/y.csv");
// //     // define regularizing PDE
// //     auto L = -laplacian<FEM>();
// //     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
// //     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> pde(domain.mesh, L, u);
// //     // grid of smoothing parameters
// //     DMatrix<double> lambda_grid(20, 1);
// //     for (int i = 0; i < 20; ++i) lambda_grid(i, 0) = std::pow(10, -4 + 0.1 * i);
// //     // define model
// //     RegularizedSVD<fdapde::sequential> rsvd(Calibration::gcv);
// //     rsvd.set_lambda(lambda_grid);
// //     rsvd.set_seed(78965);   // for reproducibility purposes in testing
// //     FPCA<SpaceOnly> model(pde, Sampling::pointwise, rsvd);
// //     model.set_spatial_locations(locs);
// //     // set model's data
// //     BlockFrame<double, int> df;
// //     df.insert(OBSERVATIONS_BLK, y);
// //     model.set_data(df);
// //     // solve FPCA problem
// //     model.init();
// //     model.solve();
// //     // test correctness
// //     EXPECT_TRUE(almost_equal(model.Psi() * model.loadings(), "../data/models/fpca/2D_test2/loadings.mtx"));
// //     EXPECT_TRUE(almost_equal(model.scores(),                 "../data/models/fpca/2D_test2/scores.mtx"  ));
// // }

// // // test 4
// // //    domain:       unit square [1,1] x [1,1]
// // //    sampling:     locations != nodes
// // //    penalization: simple laplacian
// // //    BC:           no
// // //    order FE:     1
// // //    missing data: no
// // //    solver: sequential (power iteration) + KCV \lambda selection
// // TEST(fpca_test, laplacian_samplingatlocations_sequential_kcv) {
// //     // define domain
// //     MeshLoader<Triangulation<2, 2>> domain("unit_square");
// //     // import data from files
// //     DMatrix<double> locs = read_csv<double>("../data/models/fpca/2D_test3/locs.csv");
// //     DMatrix<double> y    = read_csv<double>("../data/models/fpca/2D_test3/y.csv");
// //     // define regularizing PDE
// //     auto L = -laplacian<FEM>();
// //     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
// //     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
// //     // grid of smoothing parameters
// //     DMatrix<double> lambda_grid(20, 1);
// //     for (int i = 0; i < 20; ++i) lambda_grid(i, 0) = std::pow(10, -4 + 0.1 * i);
// //     // define model
// //     RegularizedSVD<fdapde::sequential> rsvd(Calibration::kcv);
// //     rsvd.set_lambda(lambda_grid);
// //     rsvd.set_seed(12654);   // for reproducibility purposes in testing    
// //     FPCA<SpaceOnly> model(problem, Sampling::pointwise, rsvd);
// //     model.set_spatial_locations(locs);
// //     // set model's data
// //     BlockFrame<double, int> df;
// //     df.insert(OBSERVATIONS_BLK, y);
// //     model.set_data(df);
// //     // solve FPCA problem
// //     model.init();
// //     model.solve();    
// //     // test correctness
// //     EXPECT_TRUE(almost_equal(model.Psi() * model.loadings(), "../data/models/fpca/2D_test3/loadings.mtx"));
// //     EXPECT_TRUE(almost_equal(model.scores(),                 "../data/models/fpca/2D_test3/scores.mtx"  ));
// // }

// // test 5
// //    domain:       unit square [1,1] x [1,1]
// //    sampling:     locations == nodes
// //    penalization: space-time separable
// //    BC:           no
// //    order FE:     1
// //    missing data: no
// //    solver: sequential (power iteration)
// // TEST(fpca_test, laplacian_samplingatnodes_separable_sequential) {
// //   // define time domain
// //   Triangulation<1, 1> time_mesh(0, 1, 14);
// //   // define domain and regularizing PDE
// //   MeshLoader<Triangulation<2, 2>> domain("unit_square15");
// //   // import data from files
// //   DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test5/y.csv");
// //   // define regularizing PDE in space
// //   auto Ld = -laplacian<FEM>();
// //   DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3 * time_mesh.n_nodes(), 1);
// //   PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(domain.mesh, Ld, u);
// //   // define regularizing PDE in time
// //   auto Lt = -fdapde::core::bilaplacian<fdapde::core::SPLINE>();
// //   PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, fdapde::core::SPLINE, fdapde::core::spline_order<3>> time_penalty(time_mesh, Lt);
// //     // define model
// //   double lambda_D = std::pow(10, -5); // 1e-3.6
// //   double lambda_T = std::pow(10, -5); // 1e-2.2
// //   FPCA<fdapde::models::SpaceTimeSeparable> model(
// // 				 space_penalty, time_penalty, Sampling::mesh_nodes, fdapde::models::RegularizedSVD<fdapde::sequential> {Calibration::off});
// //   model.set_lambda_D(lambda_D);
// //   model.set_lambda_T(lambda_T);
// //   // set model's data
// //   BlockFrame<double, int> df;
// //   df.insert(OBSERVATIONS_BLK, y);
// //   model.set_data(df);
// //   // solve smoothing problem
// //   model.init();
// //   model.solve();

// //   std::cout << model.scores() << std::endl;
// //   std::cout << "-----" << std::endl;
// //   std::cout << model.loadings().topRows(10) << std::endl;
  
// // }

// // TEST(fpca_test, neurocose) {
// //   // define frequency domain
// //   int n_freq = 129;
// //   Triangulation<1, 1> freq_mesh(0.01, 0.5, n_freq - 1);
// //   // define domain and regularizing PDE
// //   std::cout << "import data" << std::endl;
// //   DMatrix<double> nodes = read_csv<double>("../../../../università/articoli/neurocose/mesh/node_spaziodati_onlygm.csv", false);
// //   DMatrix<int> elems = read_csv<int>("../../../../università/articoli/neurocose/mesh/elem_onlygm2.csv").array() - 1;
// //   DMatrix<int> boundary = DMatrix<int>::Zero(nodes.rows(), 1);
// //   Triangulation<3, 3> brain(nodes, elems, boundary);
// //   std::cout << "done" << std::endl;
// //   std::cout << "brain measure:" << std::accumulate(brain.cells_begin(), brain.cells_end(), 0.0, [](double result, const auto& f) {
// //       return result + f.measure();
// //   }) << std::endl;

// //   std::cout << "---" << std::endl;
  
// //   // MeshLoader<Triangulation<2, 2>> domain("unit_square15");
// //   // import data from files
// //   std::cout << "read data" << std::endl;
// //   DMatrix<double> X = read_csv<double>("../../../../università/articoli/neurocose/dati_onlyfiltrati.csv").transpose();
// //   std::cout << X.rows() << " x " << X.cols() << std::endl;
// //   std::cout << freq_mesh.n_nodes() << std::endl;
  
// //   // define regularizing PDE in space
// //   auto Ld = -laplacian<FEM>();
// //   DMatrix<double> u = DMatrix<double>::Zero(brain.n_cells() * 3 * freq_mesh.n_nodes(), 1);
// //   PDE<Triangulation<3, 3>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(brain, Ld, u);
// //   // define regularizing PDE in time
// //   auto Lt = -fdapde::core::bilaplacian<fdapde::core::SPLINE>();
// //   PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, fdapde::core::SPLINE, fdapde::core::spline_order<3>> time_penalty(freq_mesh, Lt);

// //   // std::cout << "centering" << std::endl;
// //   // auto centered_data = center(
// //   //   X,
// //   //   fdapde::models::STRPDE<fdapde::models::SpaceTimeSeparable, fdapde::iterative>(
// //   //     space_penalty, time_penalty, Sampling::mesh_nodes),
// //   //   fdapde::calibration::Off()(SVector<2>(1, 1)));

// //   // Eigen::saveMarket(centered_data.fitted, "neurodata_centered.mtx");
// //   // Eigen::saveMarket(centered_data.mean, "neurodata_mean.mtx");
  
// //   // define model
// //   double lambda_D = std::pow(10, -6); // 1e-3.6
// //   double lambda_T = std::pow(10, -6); // 1e-2.2
// //   FPCA<fdapde::models::SpaceTimeSeparable> model(
// // 				 space_penalty, time_penalty, Sampling::mesh_nodes, fdapde::models::RegularizedSVD<fdapde::sequential> {Calibration::off});
// //   model.set_lambda_D(lambda_D);
// //   model.set_lambda_T(lambda_T);
// //   model.set_npc(2);
// //   // set model's data
// //   BlockFrame<double, int> df;
// //   df.insert(OBSERVATIONS_BLK, X);
// //   model.set_data(df);

// //   std::cout << "dati settati" << std::endl;
  
// //   // solve smoothing problem
// //   model.init();
// //   std::cout << "risolvo... moriro?" << std::endl;
// //   model.solve();

// //   std::cout << model.scores() << std::endl;
// //   std::cout << "-----" << std::endl;
// //   std::cout << model.loadings().topRows(10) << std::endl;

// //   Eigen::saveMarket(model.scores(), "neuro_scores.mtx");
// //   Eigen::saveMarket(model.loadings(), "neuro_loadings.mtx");  
// // }

// TEST(fpca_test, test_disperato) {
//   // define frequency domain
//   Triangulation<1, 1> freq_mesh(0, 1, 14);
//   // define domain and regularizing PDE
//   DMatrix<double> nodes = read_csv<double>("../../../../università/articoli/presentazioni/25_sis/dati_simulazione_disperata/nodes.csv");
//   DMatrix<int> elems = read_csv<int>("../../../../università/articoli/presentazioni/25_sis/dati_simulazione_disperata/elems.csv").array() - 1;
//   DMatrix<int> boundary = read_csv<int>("../../../../università/articoli/presentazioni/25_sis/dati_simulazione_disperata/boundary.csv");
//   Triangulation<2, 2> brain(nodes, elems, boundary);
//   std::cout << "brain measure:" << std::accumulate(brain.cells_begin(), brain.cells_end(), 0.0, [](double result, const auto& f) {
//       return result + f.measure();
//   }) << std::endl;

//   std::cout << "---" << std::endl;

//   DMatrix<double> loadings(brain.n_nodes() * freq_mesh.n_nodes(), 60);
//   DMatrix<double> scores(50, 60);
  
//   for(int i = 1; i <= 20; ++i) {
  
//     std::cout << "read data " << i << std::endl;
//     DMatrix<double> X = read_csv<double>("../../../../università/articoli/presentazioni/25_sis/dati_simulazione_disperata/data_" + std::to_string(i) + ".csv");
//     std::cout << X.rows() << " x " << X.cols() << std::endl;
  
//     // define regularizing PDE in space
//     auto Ld = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(brain.n_cells() * 3 * freq_mesh.n_nodes(), 1);
//     PDE<Triangulation<2, 2>, decltype(Ld), DMatrix<double>, FEM, fem_order<1>> space_penalty(brain, Ld, u);
//     // define regularizing PDE in time
//     auto Lt = -fdapde::core::bilaplacian<fdapde::core::SPLINE>();
//     PDE<Triangulation<1, 1>, decltype(Lt), DMatrix<double>, fdapde::core::SPLINE, fdapde::core::spline_order<3>> time_penalty(freq_mesh, Lt);

//     // define model
//     double lambda_D = std::pow(10, -4); // 1e-3.6
//     double lambda_T = std::pow(10, -6); // 1e-2.2
//     FPCA<fdapde::models::SpaceTimeSeparable> model(
// 				 space_penalty, time_penalty, Sampling::mesh_nodes, fdapde::models::RegularizedSVD<fdapde::sequential> {Calibration::off});
//     model.set_lambda_D(lambda_D);
//     model.set_lambda_T(lambda_T);
//     model.set_npc(3);
//     // set model's data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, X);
//     model.set_data(df);

    
//     // solve smoothing problem
//     model.init();
//     model.solve();

//     std::cout << "FATTO" << std::endl;

//     std::cout << model.scores().rows() << "x" << model.scores().cols() << std::endl;
//     std::cout << model.loadings().rows() << "x" << model.loadings().cols() << std::endl;
    
//     loadings.middleCols((i-1)*3, 3) = model.loadings();
//     scores.middleCols((i-1)*3, 3) = model.scores();
//   }

//   Eigen::saveMarket(scores,   "test_scores.mtx");
//   Eigen::saveMarket(loadings, "test_loadings.mtx");  
// }


// /*
// // test 4
// //    domain:       unit square [1,1] x [1,1]
// //    sampling:     locations = nodes
// //    penalization: simple laplacian
// //    BC:           no
// //    order FE:     1
// //    missing data: yes
// TEST(fpca_test, laplacian_samplingatnodes_nocalibration_missingdata) {
//     // define domain
//     MeshLoader<Mesh2D> domain("unit_square_coarse");
//     // import data from files
//     DMatrix<double> y = read_csv<double>("../data/models/fpca/2D_test4/y.csv");
//     // define regularizing PDE
//     auto L = -laplacian<FEM>();
//     DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_elements() * 3, 1);
//     PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
//     // define model
//     double lambda_D = 1e-2;
//     FPCA<decltype(problem), SpaceOnly, GeoStatMeshNodes, NoCalibration> model(problem);
//     model.set_lambda_D(lambda_D);
//     // set model's data
//     BlockFrame<double, int> df;
//     df.insert(OBSERVATIONS_BLK, y);
//     model.set_data(df);
//     // solve FPCA problem
//     model.init();
//     model.solve();
//     // test correctness
//     EXPECT_TRUE(almost_equal(model.loadings(), "../data/models/fpca/2D_test4/loadings.mtx"));
//     EXPECT_TRUE(almost_equal(model.scores(),   "../data/models/fpca/2D_test4/scores.mtx"  ));
// }
// */
