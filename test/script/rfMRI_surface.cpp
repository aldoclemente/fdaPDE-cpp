#include "utils.h"

int main(){
	
	auto start = std::chrono::high_resolution_clock::now();

	std::string meshID = "../data/mesh/brain_lh_surface_32k";
	Triangulation<2, 3> surface = read_mesh<2, 3>(meshID);
	std::cout << surface.nodes().rows() << " " << surface.nodes().cols() << std::endl;
	
	std::string response_dir = "data/rfMRI_surface/FCmaps/";
	std::string cov_dir = "data/rfMRI_surface/thickness/";
	std::string response_tail = ".fc_map.csv";
	std::string cov_tail = ".thickness.csv";
	std::size_t ID = 100307;

	DMatrix<double> y = read_csv<double>(response_dir + std::to_string(ID) + response_tail);
    DMatrix<double> X = read_csv<double>(cov_dir + std::to_string(ID) + cov_tail);
    
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

	return 0;
}

