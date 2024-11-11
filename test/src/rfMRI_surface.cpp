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

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>   // testing framework
#include <fstream>
#include <sstream>
#include <chrono> 
#include <filesystem>
#include <limits>


#include <fdaPDE/core.h>
using fdapde::core::advection;
using fdapde::core::diffusion;
using fdapde::core::FEM;
using fdapde::core::fem_order;
using fdapde::core::laplacian;
using fdapde::core::DiscretizedMatrixField;
using fdapde::core::PDE;
using fdapde::core::DiscretizedVectorField;

#include "../../fdaPDE/models/sampling_design.h"
using fdapde::models::Sampling;
using fdapde::models::SpaceOnly;
using fdapde::monolithic;
using fdapde::iterative;

#include "../../fdaPDE/models/regression/srpde.h"
#include "../../fdaPDE/models/regression/gcv.h"
using fdapde::models::SRPDE;
using fdapde::models::ExactEDF;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;
using fdapde::models::Sampling;
#include "../../fdaPDE/calibration/gcv.h"

#include "../../fdaPDE/calibration/kfold_cv.h"
#include "../../fdaPDE/calibration/rmse.h"
using fdapde::calibration::KCV;
using fdapde::calibration::RMSE;

#include "utils/constants.h"
#include "utils/mesh_loader.h"
#include "utils/utils.h"
#include<filesystem>
using fdapde::testing::almost_equal;
using fdapde::testing::MeshLoader;
//using fdapde::testing::read_csv;

#include <fdaPDE/geometry.h>
#include "fdaPDE/utils/IO/csv_reader.h"

using fdapde::core::Triangulation;

// I/O utils 
template <typename T> DMatrix<T> read_mtx(const std::string& file_name) {
    SpMatrix<T> buff;
    Eigen::loadMarket(buff, file_name);
    return buff;
}

template<typename T> void eigen2ext(const DMatrix<T>& M, const std::string& sep, const std::string& filename, bool append = false){
    std::ofstream file;

    if(!append) 
        file.open(filename);
    else
        file.open(filename, std::ios_base::app); 
    
    for(int i = 0; i < M.rows(); ++i) {
            for(int j=0; j < M.cols()-1; ++j) file << M(i,j) << sep;
            file << M(i, M.cols()-1) <<  "\n";  
    }
    file.close();
}

template<typename T> void eigen2txt(const DMatrix<T>& M, const std::string& filename = "mat.txt", bool append = false){
    eigen2ext<T>(M, " ", filename, append);
}

template<typename T> void eigen2csv(const DMatrix<T>& M, const std::string& filename = "mat.csv", bool append = false){
    eigen2ext<T>(M, ",", filename, append);
}

template< typename T> void vector2ext(const std::vector<T>& V, const std::string& sep, const std::string& filename, bool append = false){
    std::ofstream file;

    if(!append) 
        file.open(filename);
    else
        file.open(filename, std::ios_base::app);
    
    for(int i = 0; i < int(V.size())-1; ++i) file << V[i] << sep;
    
    file << V[V.size()-1] << "\n";  
    
    file.close();
}

template< typename T> void vector2txt(const std::vector<T>& V, const std::string& filename = "vec.txt", bool append = false){
   vector2ext<T>(V, " ", filename, append);
}

template< typename T> void vector2csv(const std::vector<T>& V, const std::string& filename = "vec.csv", bool append = false){
   vector2ext<T>(V, ",", filename, append);
}

void write_table(const DMatrix<double>& M, const std::vector<std::string>& header = {}, const std::string& filename = "data.txt"){

    std::ofstream file(filename);

    if(header.empty() || int(header.size()) != M.cols()){
        std::vector<std::string> head(M.cols());
        for(int i = 0; i < M.cols(); ++i)
                head[i] =  "V" + std::to_string(i);
        vector2txt<std::string>(head, filename);    
    }else vector2txt<std::string>(header, filename);
    
    eigen2txt<double>(M, filename, true);
}

void write_csv(const DMatrix<double>& M, const std::vector<std::string>& header = {}, const std::string& filename = "data.csv"){
    std::ofstream file(filename);

    if(header.empty() || int(header.size()) != M.cols()){
        std::vector<std::string> head(M.cols());
        for(int i = 0; i < M.cols(); ++i)
                head[i] =  "V" + std::to_string(i);
        vector2csv(head, filename);    
    }else vector2csv(header, filename);
    
    eigen2csv<double>(M, filename, true);
}

template <typename T> DMatrix<T> read_csv(const std::string& file_name) {
    fdapde::core::CSVReader<T> reader {};
    return reader.template parse_file<Eigen::Dense>(file_name);
}

template <int LocalDim, int EmbedDim> Triangulation<LocalDim, EmbedDim> read_mesh(const std::string& path) {
    DMatrix<double> nodes = read_csv<double>(path + "/points.csv");
    DMatrix<int> boundary = read_csv<int>(path + "/boundary.csv");
    DMatrix<int> cells    = read_csv<int>(path + "/elements.csv").array() - 1;
    return Triangulation<LocalDim, EmbedDim>(nodes, cells, boundary);
}


TEST(resting_state_surface_fMRI, read_mesh){
    
    std::string meshID = "brain_lh_surface_32k";
    MeshLoader<Triangulation<2, 3>> domain(meshID);

    std::cout << domain.mesh.nodes().rows() << " " << domain.mesh.nodes().cols() << std::endl;
    
    EXPECT_TRUE(1);
}

TEST(resting_state_surface_fMRI, read_slice_mesh){
    
    std::string meshID = "brain_lh_sagittal_slice";
    MeshLoader<Triangulation<2, 2>> domain(meshID);

    std::cout << domain.mesh.nodes().rows() << " " << domain.mesh.nodes().cols() << std::endl;
    
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);

    eigen2txt<double>(problem.quadrature_nodes(), "../data/mesh/brain_lh_sagittal_slice/quadrature_nodes.txt");
    EXPECT_TRUE(1);
}



/*
TEST(resting_state_surface_fMRI, read_subjectsIDs_NA_mask){
    std::string meshID = "brain_lh_surface_32k";
    MeshLoader<Triangulation<2, 3>> domain(meshID);

    std::string input_dir = "../script/data/rfMRI_surface/";
    DMatrix<int> IDs = read_csv<int>(input_dir + "subjectsIDs.csv");
    std::cout << IDs.rows() << " " << IDs.cols() << std::endl;
    
    DMatrix<int> NA_mask = read_csv<int>(input_dir + "na_mask.csv");
    std::cout << NA_mask.rows() << " " << NA_mask.cols() << std::endl;
    
    EXPECT_TRUE(1);
}
*/

/*
TEST(resting_state_surface_fMRI, one_subject){

    auto start = std::chrono::high_resolution_clock::now();

	std::string meshID = "../data/mesh/brain_lh_surface_32k";
    //std::string meshID = "../script/data/rfMRI_surface/norm_mesh";
	Triangulation<2, 3> surface = read_mesh<2, 3>(meshID);
	std::cout << surface.nodes().rows() << " " << surface.nodes().cols() << std::endl;

	std::string response_dir = "../script/data/rfMRI_surface/FCmaps/";
	std::string cov_dir = "../script/data/rfMRI_surface/thickness/";
	std::string response_tail = ".fc_map_NA.csv";
	std::string cov_tail = ".thickness.csv";
	std::size_t ID = 100307;

	DMatrix<double> y = read_csv<double>(response_dir + std::to_string(ID) + response_tail);
    DMatrix<double> X = read_csv<double>(cov_dir + std::to_string(ID) + cov_tail);
    std::cout << y.rows() << " " << y.cols() << std::endl;
    std::cout << X.rows() << " " << X.cols() << std::endl;
	// define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(surface.n_cells() * 3, 1);
    PDE<decltype(surface), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(surface, L, u);
    // define model
    SRPDE model(problem, Sampling::mesh_nodes);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
	df.insert(DESIGN_MATRIX_BLK, X);
    model.set_data(df);

    std::cout << "init model" << std::endl;
    model.init();

    double area = ( model.R0() * DMatrix<double>::Ones(surface.nodes().rows(), 1)).sum();
    std::cout << "area: " << area << std::endl; 
    
    
    // std::size_t seed = 314156;
    // std::cout << "\t --- GCV ---" << std::endl;
    // int n_lambdas = 5;
    // DMatrix<double> lambdas(n_lambdas, 1);
    // for (int i = 0; i < n_lambdas; ++i) { lambdas(i, 0) =  std::pow(10, -6.0 + 0.20 * i); }
    // std::cout <<"lambdas: " << "\n" << lambdas << "\n" << std::endl;
    
    // // define GCV function and grid of \lambda_D values
    // auto GCV = model.gcv<StochasticEDF>(100, seed);
    // // optimize GCV
    
    // fdapde::core::Grid<fdapde::Dynamic> opt;
    // opt.optimize(GCV, lambdas);
	// std::cout <<"lambda opt:" << opt.optimum() << "\n" << std::endl;

    // std::cout <<"gcvs: " << "\n" << std::endl;
    // for(std::size_t i = 0; i < GCV.gcvs().size(); ++i)
    //     std::cout << GCV.gcvs()[i] << std::endl;
    
    // std::cout <<"edfs (q+trS): \n" << std::endl;
    // for(std::size_t i = 0; i < GCV.edfs().size(); ++i)
    //     std::cout<< GCV.edfs()[i] << std::endl;
    
    std::cout << "\t --- KCV ---" << std::endl;
    
    int n_lambdas_kcv = 50;
    DMatrix<double> lambdas_kcv(n_lambdas_kcv, 1);
    for (int i = 0; i < n_lambdas_kcv; ++i) { lambdas_kcv(i, 0) =  std::pow(10, -2 + 0.125 * i); }
    std::cout <<"lambdas: " << "\n" << lambdas_kcv << "\n" << std::endl;
    
    int n_folds = 10;
    KCV kcv(n_folds);
    kcv.fit(model, lambdas_kcv, RMSE(model));
    std::cout <<"lambda opt:" << kcv.optimum() << "\n"<< std::endl;

    std::cout << "avg scores: \n" << kcv.avg_scores() << "\n" << std::endl;
    
    std::cout << "scores: \n" << kcv.scores() << std::endl;
    //auto KCV_ = fdapde::calibration::KCV {n_folds}(lambdas, RMSE());
    //KCV_.fit(model);

    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
	std::cout << "Duration: " << duration.count() <<  " s" << std::endl;
	
    EXPECT_TRUE(1);
}
*/

/*
TEST(gcv_srpde_test, laplacian_nonparametric_samplingatnodes_spaceonly_gridstochastic) {
    // define domain
    MeshLoader<Triangulation<2, 2>> domain("unit_square_coarse");
    // import data from files
    DMatrix<double> y = read_csv<double>("../data/models/gcv/2D_test2/y.csv");
    // define regularizing PDE
    auto L = -laplacian<FEM>();
    DMatrix<double> u = DMatrix<double>::Zero(domain.mesh.n_cells() * 3, 1);
    PDE<decltype(domain.mesh), decltype(L), DMatrix<double>, FEM, fem_order<1>> problem(domain.mesh, L, u);
    // define model
    SRPDE model(problem, Sampling::mesh_nodes);
    // set model's data
    BlockFrame<double, int> df;
    df.insert(OBSERVATIONS_BLK, y);
    model.set_data(df);
    model.init();

    double area = ( model.R0() * DMatrix<double>::Ones(domain.mesh.nodes().rows(), 1)).sum();
    std::cout << "area: " << area << std::endl; 
    // define GCV function and grid of \lambda_D values
    std::size_t seed = 476813;
    auto GCV = model.gcv<StochasticEDF>(100, seed);
    DMatrix<double> lambdas(13, 1);
    for (int i = 0; i < 13; ++i) { lambdas(i, 0) = std::pow(10, -6.0 + 0.25 * i); }
    //for (int i = 0; i < 13; ++i) { lambdas(i, 0) = std::pow(10, -10.0 + 0.25 * i); }
    // optimize GCV
    fdapde::core::Grid<fdapde::Dynamic> opt;
    opt.optimize(GCV, lambdas);
    // test correctness
    EXPECT_TRUE(almost_equal(GCV.edfs(), "../data/models/gcv/2D_test2/edfs.mtx"));
    EXPECT_TRUE(almost_equal(GCV.gcvs(), "../data/models/gcv/2D_test2/gcvs.mtx"));


    std::cout << "lambdas: \n" << lambdas << "\n" << std::endl;
    std::cout << "lambda opt: " << opt.optimum() << std::endl;

    std::cout <<"gcvs: " << "\n" << std::endl;
    for(std::size_t i = 0; i < GCV.gcvs().size(); ++i)
        std::cout << GCV.gcvs()[i] << std::endl;
    
    std::cout <<"edfs (q+trS): \n" << std::endl;
    for(std::size_t i = 0; i < GCV.edfs().size(); ++i)
        std::cout<< GCV.edfs()[i] << std::endl;

    // check consistency with GCV calibrator
    auto GCV_ = fdapde::calibration::GCV<SpaceOnly> {fdapde::core::Grid<fdapde::Dynamic> {}, StochasticEDF(100, seed)}(lambdas);
    
    EXPECT_TRUE(GCV_.fit(model) == opt.optimum());
}
*/