#include "utils.h"

int main(){
	
	std::string meshID = "../data/mesh/brain_lh_surface_32k";
	Triangulation<2, 3> surface = read_mesh<2, 3>(meshID);
	std::cout << surface.nodes().rows() << " " << surface.nodes().cols() << std::endl;
	
	return 0;
}

