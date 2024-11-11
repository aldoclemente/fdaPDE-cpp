#include "utils.h"
#include<mpi.h>

double KFC(int subjectID){

    std::string meshID = "../data/mesh/brain_lh_surface_32k";
	Triangulation<2, 3> surface = read_mesh<2, 3>(meshID);
	
	std::string response_dir = "data/rfMRI_surface/FCmaps/";
	std::string cov_dir = "data/rfMRI_surface/thickness/";
	std::string response_tail = ".fc_map.csv";
	std::string cov_tail = ".thickness.csv";
	
	DMatrix<double> y = read_csv<double>(response_dir + std::to_string(subjectID) + response_tail);
    DMatrix<double> X = read_csv<double>(cov_dir + std::to_string(subjectID) + cov_tail);
    
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
    model.init();

    // define GCV function and grid of \lambda_D values
    //std::cout << "\t --- KCV ---" << std::endl;
    
    int n_lambdas_kcv = 50;
    DMatrix<double> lambdas_kcv(n_lambdas_kcv, 1);
    for (int i = 0; i < n_lambdas_kcv; ++i) { lambdas_kcv(i, 0) =  std::pow(10, -2 + 0.125 * i); }
    
    int n_folds = 10;
    KCV kcv(n_folds);
    kcv.fit(model, lambdas_kcv, RMSE(model));
    return kcv.optimum()[0];
}

//std::pair<DMatrix<double>, DMatrix<double>> 
void solve(int subjectID, double lambda, std::string destdir){
    std::string meshID = "../data/mesh/brain_lh_surface_32k";
	Triangulation<2, 3> surface = read_mesh<2, 3>(meshID);
	
	std::string response_dir = "data/rfMRI_surface/FCmaps/";
	std::string cov_dir = "data/rfMRI_surface/thickness/";
	std::string response_tail = ".fc_map.csv";
	std::string cov_tail = ".thickness.csv";
	
	DMatrix<double> y = read_csv<double>(response_dir + std::to_string(subjectID) + response_tail);
    DMatrix<double> X = read_csv<double>(cov_dir + std::to_string(subjectID) + cov_tail);
    
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
    model.set_lambda_D(lambda);
    
    model.init();

    model.solve();
    if(!std::filesystem::exists(std::filesystem::path(destdir))) std::filesystem::create_directory(destdir);

    eigen2txt<double>(model.f(), destdir + std::to_string(subjectID) + ".f.txt");
    eigen2txt<double>(model.beta(), destdir + std::to_string(subjectID) + ".beta.txt");
    //return std::make_pair(model.f(), model.beta());
}

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    std::string datadir = "data/rfMRI_surface/";
    DMatrix<int> subjects = read_csv<int>(datadir + "subjectsIDs.csv");
    // int n_subjects;
    // if(rank == 0){
    //     std::string datadir = "data/rfMRI_surface/";
    //     subjects = read_csv<int>(datadir + "subjectsIDs.csv");
    //     n_subjects = subjects.rows();
    // }

    // // (*buffer, count, MPI_datatype, root, MPI_COMM) 
    // MPI_Bcast(&n_subjects, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // if(rank != 0) subjects = DMatrix<int>::Zero(n_subjects,1);

    // MPI_Bcast(subjects.data(), n_subjects, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank == 0) std::cout << "\t --- K-fold CV ---\n" << std::endl;

    std::vector<double> lambdas;
    lambdas.reserve(subjects.rows());

    // Cycling partition 
    for(int i=rank; i < subjects.rows(); i+=size){
        lambdas.emplace_back( KFC( subjects(i,0) ) );
    }

    if(rank == 0) std::cout << "\t ---  end  ---" << std::endl;

    double lambda_opt = std::accumulate(lambdas.begin(), lambdas.end(), 0.0);
    
    // *send_buff, *recv_buff, MPI_datatype, MPI_OP, MPI_COMM
    MPI_Allreduce(MPI_IN_PLACE, &lambda_opt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    lambda_opt /= subjects.rows();

    if(rank == 0){
        std::cout <<"lambda opt: " << lambda_opt << std::endl;

        std::ofstream out("data/rfMRI_surface/lambda_opt.txt");
        out << lambda_opt << std::endl;
        out.close();
    }

    //std::vector<DMatrix<double>, DMatrix<double>> results;
    //results.reserve(subjects.rows());
    for(int i=rank; i < subjects.rows(); i+=size){
        //results.emplace_back( solve( subjects(i,0), lambda_opt) );
        solve( subjects(i,0), lambda_opt, datadir + "results/");
    }

    
    MPI_Finalize();
    return 0;
}