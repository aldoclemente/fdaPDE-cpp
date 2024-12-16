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

#ifndef __FE_SAMPLING_H__
#define __FE_SAMPLING_H__

#include "../core/fdaPDE/linear_algebra/binary_matrix.h"

namespace fdapde {
namespace internals {

template <typename FeSpace_>
SpMatrix<double> fe_point_basis_eval(FeSpace_&& fe_space, const Eigen::Matrix<double, Dynamic, Dynamic>& coords) {
    using FeSpace = std::decay_t<FeSpace_>;
    static constexpr int local_dim = FeSpace::local_dim;
    static constexpr int embed_dim = FeSpace::embed_dim;
    fdapde_assert(coords.rows() > 0 && coords.cols() == embed_dim);

    int n_shape_functions = fe_space.n_shape_functions();
    int n_dofs = fe_space.n_dofs();
    int n_locs = coords.rows();
    SpMatrix<double> psi_(n_locs, n_dofs);
    std::vector<fdapde::Triplet<double>> triplet_list;
    triplet_list.reserve(n_locs * n_shape_functions);

    Eigen::Matrix<int, Dynamic, 1> cell_id = fe_space.triangulation().locate(coords);
    const DofHandler<local_dim, embed_dim>& dof_handler = fe_space.dof_handler();
    // build basis evaluation matrix
    for (int i = 0; i < n_locs; ++i) {
        SVector<embed_dim> p_i(coords.row(i));
        if (cell_id[i] != -1) {   // point falls inside domain
            auto cell = dof_handler.cell(cell_id[i]);
            // update matrix
            for (int h = 0; h < n_shape_functions; ++h) {
                triplet_list.emplace_back(
                  i, cell.dofs()[h], fe_space.eval_shape_value(h, cell.invJ() * (p_i - cell.node(0))));   // \psi_j(p_i)
            }
        }
    }
    // finalize construction
    psi_.setFromTriplets(triplet_list.begin(), triplet_list.end());
    psi_.makeCompressed();
    return psi_;
}

template <typename FeSpace_>
std::pair<SpMatrix<double>, Eigen::Matrix<double, Dynamic, 1>>
fe_areal_basis_eval(FeSpace_&& fe_space, const fdapde::BinaryMatrix<Dynamic, Dynamic>& incidence_mat) {
    using FeSpace = std::decay_t<FeSpace_>;
    fdapde_assert(incidence_mat.rows() > 0 && incidence_mat.cols() == fe_space.triangulation().n_cells());
    static constexpr int local_dim = FeSpace::local_dim;
    static constexpr int embed_dim = FeSpace::embed_dim;
    using FeType = typename FeSpace::FeType;
    using cell_dof_descriptor = typename FeSpace::cell_dof_descriptor;
    using BasisType = typename cell_dof_descriptor::BasisType;
    using Quadrature = typename FeType::template cell_quadrature_t<local_dim>;
    static constexpr int n_quadrature_nodes = Quadrature::order;
    static constexpr int n_shape_functions = fe_space.n_shape_functions();
    // compile time evaluation of \int_{\hat K} \psi_i on reference element \hat K
    static constexpr cexpr::Matrix<double, n_shape_functions, 1> int_table_ {[]() {
        std::array<double, n_shape_functions> int_table_ {};
        BasisType basis {cell_dof_descriptor().dofs_phys_coords()};
        for (int i = 0; i < n_shape_functions; ++i) {
            for (int k = 0; k < n_quadrature_nodes; ++k) {
                int_table_[i] += Quadrature::weights[k] * basis[i](Quadrature::nodes.row(k).transpose());
            }
        }
        return int_table_;
    }};

    int n_dofs = fe_space.n_dofs();
    int n_regions = incidence_mat.rows();
    SpMatrix<double> psi_(n_regions, n_dofs);
    Eigen::Matrix<double, Dynamic, 1> D(n_regions);
    std::vector<fdapde::Triplet<double>> triplet_list;
    triplet_list.reserve(n_regions * n_shape_functions);

    const DofHandler<local_dim, embed_dim>& dof_handler = fe_space.dof_handler();
    int tail = 0;
    for (int k = 0; k < n_regions; ++k) {
        int head = 0;
        double Di = 0;   // measure of region D_i
        for (int l = 0; l < n_dofs; ++l) {
            if (incidence_mat(k, l)) {   // element with ID l belongs to k-th region
                auto cell = dof_handler.cell(l);
                for (int h = 0; h < n_shape_functions; ++h) {
                    // compute \int_e \psi_h on physical element e
                    triplet_list.emplace_back(k, cell.dofs()[h], int_table_[h] * cell.measure());
                    head++;
                }
                Di += cell.measure();
            }
        }
        // divide each \int_{D_i} \psi_j by the measure of region D_i
        for (int j = 0; j < head; ++j) { triplet_list[tail + j].value() /= Di; }
        D[k] = Di;
        tail += head;
    }
    // finalize construction
    psi_.setFromTriplets(triplet_list.begin(), triplet_list.end());
    psi_.makeCompressed();
    return std::make_pair(std::move(psi_), std::move(D));
}

}   // namespace internals
}   // namespace fdapde

#endif // __FE_SAMPLING_H__
