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

#include "fdaPDE/models/sampling_design.h"
using fdapde::models::Sampling;
using fdapde::models::SpaceOnly;
using fdapde::monolithic;
using fdapde::iterative;

#include "fdaPDE/models/regression/srpde.h"
#include "fdaPDE/models/regression/gcv.h"
using fdapde::models::SRPDE;
using fdapde::models::ExactEDF;
using fdapde::models::GCV;
using fdapde::models::StochasticEDF;
using fdapde::models::Sampling;
#include "fdaPDE/calibration/gcv.h"


#include <fdaPDE/geometry.h>
#include "fdaPDE/utils/IO/csv_reader.h"

using fdapde::core::Triangulation;
//using namespace fdapde;

//#include "../src/utils/constants.h"
//#include "../src/utils/mesh_loader.h"
//#include "../src/utils/utils.h"
//#include<filesystem>
//using fdapde::testing::almost_equal;
//using fdapde::testing::MeshLoader;
//using fdapde::testing::read_csv;

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
    
    for(std::size_t i = 0; i < M.rows(); ++i) {
            for(std::size_t j=0; j < M.cols()-1; ++j) file << M(i,j) << sep;
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
    
    for(std::size_t i = 0; i < V.size()-1; ++i) file << V[i] << sep;
    
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

    if(header.empty() || header.size() != M.cols()){
        std::vector<std::string> head(M.cols());
        for(std::size_t i = 0; i < M.cols(); ++i)
                head[i] =  "V" + std::to_string(i);
        vector2txt<std::string>(head, filename);    
    }else vector2txt<std::string>(header, filename);
    
    eigen2txt<double>(M, filename, true);
}

void write_csv(const DMatrix<double>& M, const std::vector<std::string>& header = {}, const std::string& filename = "data.csv"){
    std::ofstream file(filename);

    if(header.empty() || header.size() != M.cols()){
        std::vector<std::string> head(M.cols());
        for(std::size_t i = 0; i < M.cols(); ++i)
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

