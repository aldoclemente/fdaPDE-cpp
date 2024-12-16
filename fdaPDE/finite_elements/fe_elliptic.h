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

#ifndef __FE_ELLIPTIC_DRIVER_H__
#define __FE_ELLIPTIC_DRIVER_H__

#include "../core/fdaPDE/finite_elements/fe_mass_assembler.h"
#include "../core/fdaPDE/geoframe/geoframe.h"
#include "../core/fdaPDE/linear_algebra/binary_matrix.h"
#include "../core/fdaPDE/linear_algebra/smw.h"
#include "../core/fdaPDE/linear_algebra/sparse_block_matrix.h"
#include "../formula.h"
#include "fe_sampling.h"

namespace fdapde {
namespace internals {

// efficient left multiplication Q*x, with Q = W * (I - X * (X^\top * W * X)^{-1} * X^\top * W)
template <typename WeightMatrix, typename DesignMatrix, typename InvDesignMatrix>
Eigen::Matrix<double, Dynamic, Dynamic> lmbQ(
  const WeightMatrix& W, const DesignMatrix& X, const InvDesignMatrix& invXtWX,
  const Eigen::Matrix<double, Dynamic, Dynamic>& x) {
    using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
    if (X.cols() == 0) return W * x;
    MatrixType v = X.transpose() * W * x;   // X^\top * W * x
    MatrixType z = invXtWX.solve(v);        // (X^\top * W * X)^{-1} * X^\top * W * x
    // compute W*x - W*X*z = W*x - (W*X*(X^\top*W*X)^{-1}*X^\top*W)*x = W(I - H)*x = Q*x
    return W * (x - X * z);
}
template <typename DesignMatrix, typename InvDesignMatrix>
Eigen::Matrix<double, Dynamic, Dynamic>
lmbQ(const DesignMatrix& X, const InvDesignMatrix& invXtX, const Eigen::Matrix<double, Dynamic, Dynamic>& x) {
    using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
    if (X.cols() == 0) return x;
    MatrixType v = X.transpose() * x;   // X^\top * x
    MatrixType z = invXtX.solve(v);     // (X^\top * X)^{-1} * X^\top * x
    // compute x - X*z = x - (X*(X^\top*X)^{-1}*X^\top)*x = (I - H)*x = Q*x
    return x - X * z;
}

struct fe_elliptic_driver_base {
    using VectorType = Eigen::Matrix<double, Dynamic, 1>;
    using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
    using SparseMatrixType = Eigen::SparseMatrix<double>;
    using DiagonalMatrixType = Eigen::DiagonalMatrix<double, Dynamic, Dynamic>;

    fe_elliptic_driver_base() noexcept = default;
    template <typename BilinearForm_, typename LinearForm_, typename GeoFrame>
    fe_elliptic_driver_base(const GeoFrame& gf, BilinearForm_&& bilinear_form, LinearForm_&& linear_form) :
        R1_(bilinear_form.assemble()), u_(linear_form.assemble()) {
        using BilinearForm = std::decay_t<BilinearForm_>;
        using LinearForm = std::decay_t<LinearForm_>;
        using FeSpace = typename BilinearForm::TrialSpace;

        internals::fe_mass_assembly_loop<FeSpace> mass_assembler(bilinear_form.trial_space());
        R0_ = mass_assembler.assemble();     // mass matrix
        n_dofs_ = bilinear_form.n_dofs();    // number of basis functions over physical domain

        // evaluate basis system on physical domain
        switch (gf.layer_category(0).value()) {
        case ltype::point: {
            const auto& layer = gf.get_as(layer_t::point, 0);
            if (layer.locs_at_mesh_nodes()) {
                // locations at mesh nodes
                Psi_.resize(n_dofs_, n_dofs_);
                Psi_.setIdentity();   // \psi_i(p_j) = 1 \iff i == j, otherwise \psi_i(p_j) = 0
            } else {
                Psi_ = internals::fe_point_basis_eval(bilinear_form.trial_space(), layer.coordinates());
            }
            D_ = VectorType::Ones(Psi_.rows()).asDiagonal();
            break;
        }
        case ltype::areal: {
            const auto& layer = gf.get_as(layer_t::areal, 0);
            const auto& [psi, measure_vect] =
              internals::fe_areal_basis_eval(bilinear_form.trial_space(), layer.incidence_matrix());
            Psi_ = psi;
            D_ = measure_vect.asDiagonal();   // regions' measure
            break;
        }
        }
    }
    // observers
    int n_dofs() const { return n_dofs_; }
    const SparseMatrixType& R0() const { return R0_; }
    const SparseMatrixType& R1() const { return R1_; }
    const SparseMatrixType& Psi() const { return Psi_; }
    const VectorType& u() const { return u_; }
    // penalty matrix
    MatrixType P(double lambda) const {
        if (!invR0_) { invR0_.compute(R0_); }
        return lambda * R1_.transpose() * invR0_.solve(R1_);
    }
   protected:
    int n_dofs_ = 0;         // number of basis on physical domain
    SparseMatrixType R0_;    // mass matrix: [R0]_{ij} = \int_D \psi_i * \psi_j
    SparseMatrixType R1_;    // discretization of bilinear form a: [R1]_{ij} = \int_D a(\psi_i, \psi_j)
    SparseMatrixType Psi_;   // evaluation of basis system \psi_1, ..., \psi_{n_dofs_} on physical domain
    VectorType u_;           // discretized forcing term u_i = \int_D u * \psi_i
    DiagonalMatrixType D_;   // vector of regions' measures (areal sampling)
    mutable fdapde::SparseLU<SpMatrix<double>> invR0_;
};

// solves \min_{f, \beta} \| y_i - x_i^\top * \beta - f(p_i) \|_2^2 + \int_D (Lf - u)^2, L elliptic operator
struct fe_elliptic_driver_impl : fe_elliptic_driver_base {
    fe_elliptic_driver_impl() noexcept = default;
    template <typename BilinearForm, typename LinearForm, typename GeoFrame>
    fe_elliptic_driver_impl(
      const std::string& formula, const GeoFrame& gf, BilinearForm&& bilinear_form, LinearForm&& linear_form) :
        fe_elliptic_driver_base(gf, bilinear_form, linear_form) {
        // parse formula
        Formula formula_(formula);
        std::vector<std::string> covs;
        for (const std::string& token : formula_.rhs()) {
            if (gf.contains(token)) { covs.push_back(token); }
        }
        q_ = covs.size();
	// extract data from geoframe
        y_.resize(gf[0].rows());
        gf[0].template col<double>(formula_.lhs()).data().assign_to(y_);
        if (q_ != 0) {
            // assemble design matrix
            X_.resize(gf[0].rows(), q_);
            for (int i = 0; i < q_; ++i) { gf[0].template col<double>(covs[i]).data().assign_to(X_.col(i)); }
            XtX_ = X_.transpose() * X_;
            invXtX_ = XtX_.partialPivLu();
            invXtXXt_ = invXtX_.solve(X_.transpose());   // (X^\top * X)^{-1} * X^\top
            // woodbury decomposition matrices
            U_ = MatrixType::Zero(2 * n_dofs_, q_);
            U_.block(0, 0, n_dofs_, q_) = Psi_.transpose() * D_ * X_;
            V_ = MatrixType::Zero(q_, 2 * n_dofs_);
            V_.block(0, 0, q_, n_dofs_) = X_.transpose() * Psi_;
        }
    }

    void operator()(double lambda) {
        // assemble system matrix for nonparameteric part
        SparseBlockMatrix<double, 2, 2> A_(
          -Psi_.transpose() * D_ * Psi_, lambda * R1_.transpose(), lambda * R1_, lambda * R0_);
        invA_.compute(A_);
        // linear system rhs
        VectorType b_(2 * n_dofs_);
        b_.block(n_dofs_, 0, n_dofs_, 1) = lambda * u_;

        VectorType x;
        if (q_ == 0) {   // nonparametric case
            b_.block(0, 0, n_dofs_, 1) = -Psi_.transpose() * D_ * y_;
            x = invA_.solve(b_);   // solve A * x = b
            f_ = x.head(n_dofs_);
        } else {   // parametric case
            b_.block(0, 0, n_dofs_, 1) = -Psi_.transpose() * D_ * internals::lmbQ(X_, invXtX_, y_);
            x = SMW().solve(invA_, U_, XtX_, V_, b_);   // solve A * x = (A_ + U_ * (X^\top*X) * V_) * x = b
            f_ = x.head(n_dofs_);
            beta_ = invXtXXt_ * (y_ - Psi_ * f_);
        } 
        g_ = x.tail(n_dofs_);   // PDE misfit
        return;
    }
    // observers
    const VectorType& f() const { return f_; }   // return as FeFunction
    const VectorType& beta() const { return beta_; }
    const VectorType& g() const { return g_; }
   private:
    int q_;                       // number of covariates
    MatrixType X_;                // n_obs x q design matrix
    VectorType y_;                // n_obs x 1 observation vector
    MatrixType XtX_, invXtXXt_;   // q x q matrix X^\top * X and q x n_obs matrix (X^\top * X)^{-1} * X^\top
    MatrixType U_, V_;            // (2 * n_dofs) x q matrices [\Psi^\top * D * y, 0] and [X^\top * \Psi, 0]
    Eigen::PartialPivLU<MatrixType> invXtX_;    // factorization of q x q matrix X^\top * X
    fdapde::SparseLU<SpMatrix<double>> invA_;   // factorization of (2 * n_dofs) x (2 * n_dofs) nonparametric matrix
    VectorType f_, beta_, g_;
};

// solves \min_{f, \beta} \| W^{1/2} * (y_i - x_i^\top * \beta - f(p_i)) \|_2^2 + \int_D (Lf - u)^2, L elliptic operator
struct fe_weighted_elliptic_driver_impl : public fe_elliptic_driver_base {
    fe_weighted_elliptic_driver_impl() noexcept = default;
    template <typename BilinearForm, typename LinearForm, typename GeoFrame>
    fe_weighted_elliptic_driver_impl(
      const std::string& formula, const GeoFrame& gf, BilinearForm&& bilinear_form, LinearForm&& linear_form) :
        fe_elliptic_driver_base(gf, bilinear_form, linear_form) {
        // parse formula
        Formula formula_(formula);
        std::vector<std::string> covs;
        for (const std::string& token : formula_.rhs()) {
            if (gf.contains(token)) { covs.push_back(token); }
        }
        q_ = covs.size();
        // extract data from geoframe
        y_.resize(gf[0].rows());
        gf[0].template col<double>(formula_.lhs()).data().assign_to(y_);   // observation vector
        if (q_ != 0) {
            // assemble design matrix
            X_.resize(gf[0].rows(), q_);
            for (int i = 0; i < q_; ++i) { gf[0].template col<double>(covs[i]).data().assign_to(X_.col(i)); }
            // woodbury decomposition matrices
            U_ = MatrixType::Zero(2 * n_dofs_, q_);
            V_ = MatrixType::Zero(q_, 2 * n_dofs_);
        }
    }

    template <typename WeightMatrix> void operator()(double lambda, WeightMatrix&& W) {
        // assemble system matrix for nonparameteric part
        SparseBlockMatrix<double, 2, 2> A_(
          -Psi_.transpose() * D_ * W * Psi_, lambda * R1_.transpose(), lambda * R1_, lambda * R0_);
        invA_.compute(A_);
        // linear system rhs
        VectorType b_(2 * n_dofs_);
        b_.block(n_dofs_, 0, n_dofs_, 1) = lambda * u_;

        VectorType x;
        if (q_ == 0) {   // nonparametric case
            b_.block(0, 0, n_dofs_, 1) = -Psi_.transpose() * D_ * W * y_;
            x = invA_.solve(b_);   // solve A * x = b
            f_ = x.head(n_dofs_);
        } else {   // parametric case
            MatrixType XtWX_ = X_.transpose() * W * X_;
            Eigen::PartialPivLU<MatrixType> invXtWX_ = XtWX_.partialPivLu();
            b_.block(0, 0, n_dofs_, 1) = -Psi_.transpose() * D_ * internals::lmbQ(W, X_, invXtWX_, y_);
            // woodbury matrices
            U_.block(0, 0, n_dofs_, q_) = Psi_.transpose() * D_ * W * X_;
            V_.block(0, 0, q_, n_dofs_) = X_.transpose() * W * Psi_;
            // solve A * x = (A_ + U_ * (X^\top*W*X) * V_) * x = b
            x = SMW().solve(invA_, U_, XtWX_, V_, b_);
            f_ = x.head(n_dofs_);
            beta_ = invXtWX_.solve(X_.transpose() * W) * (y_ - Psi_ * f_);
        }
        g_ = x.tail(n_dofs_);   // PDE misfit
        return;
    }
   private:
    int q_;              // number of covariates
    MatrixType X_;       // n_obs x q design matrix
    VectorType y_;       // n_obs x 1 observation vector
    MatrixType U_, V_;   // (2 * n_dofs) x q matrices [\Psi^\top * D * y, 0] and [X^\top * \Psi, 0]
    fdapde::SparseLU<SpMatrix<double>> invA_;   // factorization of (2 * n_dofs) x (2 * n_dofs) nonparametric matrix
    VectorType f_, beta_, g_;
};

}   // namespace internals
}   // namespace fdapde

#endif // __FE_ELLIPTIC_DRIVER_H__
