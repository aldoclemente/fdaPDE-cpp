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

#ifndef __STRPDE_H__
#define __STRPDE_H__

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/pde.h>
#include <fdaPDE/utils.h>

#include <memory>
#include <type_traits>

#include "../model_base.h"
#include "../model_macros.h"
#include "../model_traits.h"
#include "../sampling_design.h"
#include "regression_base.h"
using fdapde::core::BlockVector;
using fdapde::core::Kronecker;
using fdapde::core::SMW;
using fdapde::core::SparseBlockMatrix;
using fdapde::core::KroneckerTensorProduct;
using fdapde::core::SplineBasis;

namespace fdapde {
namespace models {

// STRPDE model signature
template <typename RegularizationType, typename SolutionPolicy> class STRPDE;

// implementation of STRPDE for separable space-time regularization
template <>
class STRPDE<SpaceTimeSeparable, monolithic> :
    public RegressionBase<STRPDE<SpaceTimeSeparable, monolithic>, SpaceTimeSeparable> {
   private:
    SparseBlockMatrix<double, 2, 2> A_ {};      // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
    DVector<double> b_ {};                      // right hand side of problem's linear system (1 x 2N vector)
    DMatrix<double> T_;                         // T = \Psi^T*Q*\Psi + \lambda*R
    SpMatrix<double> K_;                        // P1 \kron R0
   public:
    using RegularizationType = SpaceTimeSeparable;
    using Base = RegressionBase<STRPDE<RegularizationType, monolithic>, RegularizationType>;
    // import commonly defined symbols from base
    IMPORT_REGRESSION_SYMBOLS;
    using Base::lambda_D;   // smoothing parameter in space
    using Base::lambda_T;   // smoothing parameter in time
    using Base::P;          // discretized penalty: P = \lambda_D*((R1^T*R0^{-1}*R1) \kron Rt) + \lambda_T*(R0 \kron Pt)
    using Base::P0;         // time mass matrix: [P0_]_{ij} = \int_{[0,T]} \phi_i*\phi_j
    using Base::P1;         // time penalization matrix: [P1_]_{ij} = \int_{[0,T]} (\phi_i)_tt*(\phi_j)_tt
    // constructor
    STRPDE() = default;
    STRPDE(const pde_ptr& space_penalty, const pde_ptr& time_penalty, Sampling s) :
        Base(space_penalty, time_penalty, s) {};

    void init_model() {
        // a change in the smoothing parameter must reset the whole linear system
        if (runtime().query(runtime_status::is_lambda_changed)) {
            // assemble system matrix for the nonparameteric part
            if (is_empty(K_)) K_ = Kronecker(P1(), pde().mass());
            A_ = SparseBlockMatrix<double, 2, 2>(
              -PsiTD() * W() * Psi() - lambda_T() * K_, lambda_D() * R1().transpose(),
	      lambda_D() * R1(),                        lambda_D() * R0()            );
            invA_.compute(A_);
            // prepare rhs of linear system
            b_.resize(A_.rows());
            b_.block(A_.rows() / 2, 0, A_.rows() / 2, 1) = lambda_D() * u();
            return;
        }
        if (runtime().query(runtime_status::require_W_update)) {
            // adjust north-west block of matrix A_ and factorize
            A_.block(0, 0) = -PsiTD() * W() * Psi() - lambda_T() * K_;
            invA_.compute(A_);
            return;
        }
    }
    void solve() {   // finds a solution to the smoothing problem
        BLOCK_FRAME_SANITY_CHECKS;
        DVector<double> sol;             // room for problem' solution
        if (!Base::has_covariates()) {   // nonparametric case
            // update rhs of STR-PDE linear system
            b_.block(0, 0, A_.rows() / 2, 1) = -PsiTD() * W() * y();
            // solve linear system A_*x = b_
            sol = invA_.solve(b_);
            f_ = sol.head(A_.rows() / 2);
        } else {   // parametric case
            // update rhs of STR-PDE linear system
            b_.block(0, 0, A_.rows() / 2, 1) = -PsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z

            // definition of matrices U and V  for application of woodbury formula
            U_ = DMatrix<double>::Zero(A_.rows(), q());
            U_.block(0, 0, A_.rows() / 2, q()) = PsiTD() * W() * X();
            V_ = DMatrix<double>::Zero(q(), A_.rows());
            V_.block(0, 0, q(), A_.rows() / 2) = X().transpose() * W() * Psi();
            // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
            sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
            // store result of smoothing
            f_ = sol.head(A_.rows() / 2);
            beta_ = invXtWX().solve(X().transpose() * W()) * (y() - Psi() * f_);
        }
        // store PDE misfit
        g_ = sol.tail(A_.rows() / 2);
        return;
    }
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const {   // euclidian norm of op1 - op2
        return (op1 - op2).squaredNorm();
    }

    // getters
    const SparseBlockMatrix<double, 2, 2>& A() const { return A_; }
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }

    virtual ~STRPDE() = default;
};

template <>
class STRPDE<SpaceTimeSeparable, iterative> :
    public RegressionBase<STRPDE<SpaceTimeSeparable, iterative>, SpaceTimeSeparable> {
   private:
  // SparseBlockMatrix<double, 3, 3> A_ {};
    fdapde::SparseLU<SpMatrix<double>> invA_;
    DVector<double> b_ {};   // right hand side of problem's linear system
  // SparseBlockMatrix<double, 2, 2> As_ {};
    fdapde::SparseLU<SpMatrix<double>> invAs_;
    DVector<double> bs_ {};
    double DeltaT_;                             // \DeltaT = t_1 - t_0 (assume equidistant points in time)
  DVector<double> x_old_;
  DVector<double> x_new_;
  double alpha_;
  
    // the functional minimized by the iterative scheme
    // J(f,g) = \sum_{k=1}^m (z^k - \Psi*f^k)^T*(z^k - \Psi*f^k) + \lambda_S*(g^k)^T*(g^k) + \lambda_T*(l^k)^T*(l^k)
    double J_(const DVector<double>& x) const {
        double sse = 0;   // \sum_{k=1}^m (y^k - \Psi*f^k)^T*(y^k - \Psi*f^k)
        for (int t = 0; t < n_temporal_locs(); ++t) {
            sse += (y(t) - Psi() * x.middleRows(t * n_spatial_basis(), n_spatial_basis())).squaredNorm();
        }
        sse += lambda_D() * x.middleRows(n_dofs(), n_dofs()).squaredNorm() +
               lambda_T() * x.middleRows(2 * n_dofs(), n_dofs()).squaredNorm();
	return sse;
    }
    // accessors
    auto f_old(int k) { return x_old_.block(n_spatial_basis() * k, 0, n_spatial_basis(), 1); }
    auto g_old(int k) { return x_old_.block(n_dofs() + n_spatial_basis() * k, 0, n_spatial_basis(), 1); }
    auto l_old(int k) { return x_old_.block(2 * n_dofs() +  n_spatial_basis() * k, 0, n_spatial_basis(), 1); }
    auto f_new(int k) { return x_new_.block(n_spatial_basis() * k, 0, n_spatial_basis(), 1); }
    auto g_new(int k) { return x_new_.block(n_dofs() + n_spatial_basis() * k, 0, n_spatial_basis(), 1); }
    auto l_new(int k) { return x_new_.block(2 * n_dofs() +  n_spatial_basis() * k, 0, n_spatial_basis(), 1); }
  
    void solve_(int t) {
        DVector<double> x = invA_.solve(b_);
        f_new(t) = x.middleRows(0, n_spatial_basis());
        g_new(t) = x.middleRows(n_spatial_basis(), n_spatial_basis());
        // impose \frac{\partial^2 f}{\partial t^2} = l = 0 for t \in {0, m-1}
        if (t != 0 && t != n_temporal_locs() - 1) l_new(t) = x.middleRows(2 * n_spatial_basis(), n_spatial_basis());
    }
    // internal utilities
    DMatrix<double> y(int k) const { return y().block(n_spatial_locs() * k, 0, n_spatial_locs(), 1); }
    DMatrix<double> u(int k) const { return u_.block(n_spatial_basis() * k, 0, n_spatial_basis(), 1); }
    int n_dofs() const { return n_spatial_basis() * n_temporal_locs(); }

    // quantities related to iterative scheme
    double tol_ = 1e-4;   // tolerance used as stopping criterion
    int max_iter_ = 20;   // maximum number of iterations
   public:
    using RegularizationType = SpaceTimeSeparable;
    using Base = RegressionBase<STRPDE<RegularizationType, iterative>, RegularizationType>;
    // import commonly defined symbols from base
    IMPORT_REGRESSION_SYMBOLS
    using Base::lambda_D;          // smoothing parameter in space
    using Base::lambda_T;          // smoothing parameter in time
    using Base::n_temporal_locs;   // number of time instants m defined over [0,T]
    using Base::space_pde_;        // space-only differential operator Lf - u
    // constructor
    STRPDE() = default;
    STRPDE(const Base::PDE& space_penalty, const Base::PDE& time_penalty, Sampling s) :
        Base(space_penalty, time_penalty, s) {};
  
    void tensorize_psi() { return; } // avoid tensorization of \Psi matrix
    void init_regularization() {
      	std::cout << "init regularization" << std::endl;
        space_pde_.init();
        // compute time step (assuming equidistant points)
        DeltaT_ = is_empty(time_locs_) ? time_[1] - time_[0] : time_locs_[1] - time_locs_[0];
        // stack forcing term
        u_.resize(n_spatial_basis() * n_temporal_locs());
        for (int i = 0; i < n_temporal_locs(); ++i) {
            u_.segment(i * n_spatial_basis(), n_spatial_basis()) = space_pde_.force();
        }
    }
    // getters
    const SpMatrix<double>& R0() const { return space_pde_.mass(); }
    const SpMatrix<double>& R1() const { return space_pde_.stiff(); }
  
  int n_basis() const { return n_spatial_basis() * (n_temporal_locs() - 2); } // this is not true, it should be n_dofs(), TODO correct!!
  
  // const SparseBlockMatrix<double, 3, 3>& A() const { return A_; }
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    const DVector<double>& b() const { return b_; }


  // can we do better??
    double ftPf(const SVector<Base::n_lambda>& lambda, const DVector<double>& f, const DVector<double>& g) const {
        fdapde_assert(f.rows() == n_spatial_basis() * n_temporal_locs() && g.rows() == f.rows());
        double ftPf_ = 0;
        int N = n_spatial_basis(), m = n_temporal_locs();
        // \lambda_D * f^\top * (R_1^\top * R0^{-1} * R1) * f = g^\top * (I \kron R0) * g = \sum g_i^\top * R0 * g_i
        for (int i = 0; i < m; ++i) { ftPf_ += lambda[0] * g.middleRows(i * N, N).dot(R0() * g.middleRows(i * N, N)); }
        // \lambda_T * f^\top * (D * R0^{-1} * D) * f = \lambda_T * f^\top * (D^2 \kron R0) * f
        ftPf_ += lambda[1] * (4 * f.middleRows(0, N).dot(R0() * f.middleRows(0, N)) +
                              f.middleRows(N, N).dot(R0() * f.middleRows(N, N)));
        for (int i = 1; i < m - 2; ++i) {
            ftPf_ += lambda[1] * (f.middleRows((i - 1) * N, N).dot(R0() * f.middleRows((i - 1) * N, N)) +
                                  4 * f.middleRows(i * N, N).dot(R0() * f.middleRows(i * N, N)) +
                                  f.middleRows((i + 1) * N, N).dot(R0() * f.middleRows((i + 1) * N, N)));
        }
        ftPf_ += lambda[1] * (4 * f.middleRows((m - 2) * N, N).dot(R0() * f.middleRows((m - 2) * N, N)) +
                              f.middleRows((m - 1) * N, N).dot(R0() * f.middleRows((m - 1) * N, N)));
        return ftPf_;
    }

  // to be removed, this is here just because of type erasure
  double ftPf(const SVector<Base::n_lambda>& lambda) const { 
        if (is_empty(g_)) return f().dot(Base::P(lambda) * f());   // fallback to explicit f^\top*P*f
        return ftPf(lambda, f(), g());
    }
  double ftR0f(const DVector<double>& f) const {
    double ftR0f_ = 0;
    int N = n_spatial_basis(), m = n_temporal_locs();
    for(int i = 0; i < m; ++i) { ftR0f_ += f.middleRows(i * N, N).dot(R0() * f.middleRows(i * N, N)); }
    return ftR0f_;
  }
  double ftR0f() const { return ftR0f(f()); }

  void init_model() {
    if (runtime().query(runtime_status::is_lambda_changed)) {
      std::cout << "init model" << std::endl;
      std::cout << "n_temporal_locs: " << n_temporal_locs() << std::endl;
      std::cout << "n_dofs: " << n_dofs() << std::endl;
            x_old_.resize(3 * n_dofs());
            x_new_.resize(3 * n_dofs());
	    std::cout << "allocated x_old_, x_new_" << std::endl;
            // assemble system matrix for the space-only part
            SparseBlockMatrix<double, 2, 2> As_(
              PsiTD() * Psi(),    lambda_D() * R1().transpose(),
	      lambda_D() * R1(), -lambda_D() * R0()            );
	    std::cout << "built matrix As" << std::endl;
	    invAs_.compute(As_);
	    std::cout << "factorized" << std::endl;
	    bs_.resize(As_.rows());
	    std::cout << "allocated vector bs" << std::endl;

	    const SpMatrix<double>& Zero = SpMatrix<double>(n_spatial_basis(), n_spatial_basis());
	    alpha_ = lambda_T() / std::pow(DeltaT_, 2);

            SparseBlockMatrix<double, 3, 3> A_(
              PsiTD() * Psi(),    lambda_D() * R1().transpose(), -2 * alpha_ * R0(),
	      lambda_D() * R1(), -lambda_D() * R0(),              Zero,
              -2 * alpha_ * R0(),     Zero,                      -lambda_T() * R0());
	    std::cout << "built matrix A" << std::endl;
            invA_.compute(A_);
	    std::cout << "factorized" << std::endl;
            b_.resize(A_.rows());
	    std::cout << "allocated vector b" << std::endl;
            return;
        }
        return;
    }
    void solve() {
        fdapde_assert(y().rows() != 0);
	std::cout << "solve strpde" << std::endl;
        int N = n_spatial_basis(), m = n_temporal_locs();
        for (int k = 0; k < m; k++) {   // f^(k,0), g^(k,0) k = 1 ... m as solution of Ax = b(k)
            bs_ << PsiTD() * y(k), lambda_D() * u(k);
            DVector<double> sol = invAs_.solve(bs_);
            f_old(k) = sol.head(N);
            g_old(k) = sol.tail(N);
        }
        // l^(k,0) k = 1 ... m as l^(k,0) = \frac{1}{\DeltaT^2} (f^(k+1, 0) - 2*f^(k,0) + f^(k-1, 0))
        l_old(0)     = DVector<double>::Zero(N);
        l_old(m - 1) = DVector<double>::Zero(N);
        for (int k = 1; k < m - 1; ++k) {
            l_old(k) = (f_old(k + 1) - 2 * f_old(k) + f_old(k - 1)) / std::pow(DeltaT_, 2);
        }

	// std::cout << y(0).topRows(10) << std::endl;
	// std::cout << "----" << std::endl;
	// std::cout << f_old(0).topRows(10) << std::endl;
	// std::cout << "----" << std::endl;
	// std::cout << g_old(0).topRows(10) << std::endl;
	// std::cout << "----" << std::endl;
	
        // iterative scheme initialization
        double Jold = std::numeric_limits<double>::max();
        double Jnew = J_(x_old_);
	int i = 1;
        // main loop
        while (i < max_iter_ && std::abs((Jnew - Jold) / Jnew) > tol_) {
	  // std::cout << "iter: " << i << std::endl;
            // at step 0 f^(-1,i-1) = l^(-1,i-1) = 0
            b_ << PsiTD() * y(0) - alpha_ * R0() * l_old(1), lambda_D() * u(0), - alpha_ * R0() * f_old(1);
	    solve_(0);
            // general step
            for (int k = 1; k < m - 1; ++k) {
                b_ << PsiTD() * y(k) - alpha_ * R0() * (l_old(k + 1) + l_old(k - 1)), lambda_D() * u(k),
                  - alpha_ * R0() * (f_old(k + 1) + f_old(k - 1));
		solve_(k);
            }
            // at last step f^(m+1,i-1) = l^(m+1, i-1) = 0
            b_ << PsiTD() * y(m - 1) - alpha_ * R0() * l_old(m - 1), lambda_D() * u(m - 1),
              -alpha_ * R0() * f_old(m - 1);
	    solve_(m - 1);

	    // std::cout << f_new(0).topRows(10) << std::endl;
	    // std::cout << "----" << std::endl;
	    // std::cout << g_new(0).topRows(10) << std::endl;
	    // std::cout << "----" << std::endl;
	    
            // prepare for next iteration
            Jold = Jnew;
            Jnew = J_(x_new_);
	    std::cout << Jnew << std::endl;
            // if (Jnew > Jold) {
	    //   std::cout << "break" << std::endl;
	    //   break;
	    // }
	    x_old_ = x_new_;   // f_old = f_new; g_old = g_new; l_old = l_new;
        // if (Jnew >= Jold) {
	    //   std::cout << Jold << std::endl;
	    //   std::cout << Jnew << std::endl;
	    //   std::cout << "increasing objective functional" << std::endl;
	    //   return;
            // }
            i++;
        }
        f_ = x_old_.middleRows(0, n_dofs());
        g_ = x_old_.middleRows(n_dofs(), n_dofs());
	std::cout << "~~~~" << std::endl;
        return;
    }

  // setters
    void set_tolerance(double tol) { tol_ = tol; }
    void set_max_iter(int max_iter) { max_iter_ = max_iter; }
  
  
    int r_ = 100;          // number of monte carlo realizations
    DMatrix<double> Us_;   // sample from Rademacher distribution
    DMatrix<double> Y_;    // Us_^T*\Psi
    int seed_ = fdapde::random_seed;
    bool init_ = false;
  void set_seed(int seed) { seed_ = seed; }
  std::map<DVector<double>, double, fdapde::d_vector_compare<double>> cache_;
  
    // TODO: make more elegant
    double gcv() {
        if (!init_) {
            // compute sample from Rademacher distribution
            std::mt19937 rng(seed_);
            std::bernoulli_distribution Be(0.5);   // bernulli distribution with parameter p = 0.5
            Us_.resize(n_locs(), r_);              // preallocate memory for matrix Us
	    Y_.resize(r_, n_locs());
            // fill matrix
            for (int i = 0; i < n_locs(); ++i) {
                for (int j = 0; j < r_; ++j) {
                    if (Be(rng))
                        Us_(i, j) = 1.0;
                    else
                        Us_(i, j) = -1.0;
                }
            }
            // prepare matrix Y
	    for(int i = 0; i < n_temporal_locs(); ++i) { // need this because Psi() is only spatial, but Us_ is spatio-temporal
	      Y_.middleCols(i * Base::n_spatial_basis(), Base::n_spatial_basis()) =
		Us_.transpose().middleCols(i * Base::n_spatial_basis(), Base::n_spatial_basis()) * Psi();
	    }
            init_ = true;   // never reinitialize again
        }

        if (cache_.find(lambda()) == cache_.end()) {
            DMatrix<double> y_ = Base::y();   // store before overwrite
            double trS = 0;
            for (int i = 0; i < r_; ++i) {
   	        Base::data().insert(OBSERVATIONS_BLK, DMatrix<double>(Us_.col(i)));
                solve();
                trS += Y_.row(i).dot(fitted());
            }
            Base::data().insert(OBSERVATIONS_BLK, y_);
	    cache_[lambda()] = trS / r_;
	}
        solve();                // solve again with right observations!!!!!!
        int n = n_obs();        // number of observations
        double dor = n - cache_.at(lambda()); // residual degrees of freedom
        // return gcv at point
        double gcv_value = (n / std::pow(dor, 2)) * ((fitted() - y()).squaredNorm());
        return gcv_value;
    }

    DVector<double> fitted() const {
        DVector<double> result(Base::n_locs());
        int n = Base::n_spatial_basis();
        for (int i = 0; i < Base::n_temporal_locs(); ++i) {
            result.middleRows(i * n, n) = Psi() * f_.middleRows(i * n, n);
        }
	return result;
    }
};

    //   void solve() {
    //     fdapde_assert(y().rows() != 0);
    //     int N = n_spatial_basis(), m = n_temporal_locs();
    //     BlockVector<double> f_old(m, N), g_old(m, N), l_old(m, N);

    //     // compute starting point (f^(k,0), g^(k,0), l^(k,0)) k = 1 ... m <--------------- also this can be factored once per lambda
	
    //     SparseBlockMatrix<double, 2, 2> A(
    //       PsiTD() * Psi(),    lambda_D() * R1().transpose(),
    // 	  lambda_D() * R1(), -lambda_D() * R0()            );
    //     fdapde::SparseLU<SpMatrix<double>> invA;
    //     invA.compute(A);
    //     DVector<double> b(2 * N);
    //     for (int k = 0; k < m; k++) {   // f^(k,0), g^(k,0) k = 1 ... m as solution of Ax = b(k)
    //         b << PsiTD() * y(k), lambda_D() * u(k);
    //         DVector<double> sol = invA.solve(b);
    //         f_old(k) = sol.head(N);
    //         g_old(k) = sol.tail(N);
    //     }
    //     // l^(k,0) k = 1 ... m as l^(k,0) = \frac{1}{\DeltaT^2} (f^(k+1, 0) - 2*f^(k,0) + f^(k-1, 0))
    //     l_old(0)     = DVector<double>::Zero(N);
    //     l_old(m - 1) = DVector<double>::Zero(N);
    //     for (int k = 1; k < m - 1; ++k) {
    //         l_old(k) = (f_old(k + 1) - 2 * f_old(k) + f_old(k - 1)) / std::pow(DeltaT_, 2);
    //     }
    //     // iterative scheme initialization
    //     BlockVector<double> f_new(m, N), g_new(m, N), l_new(m, N);
    //     double Jold = std::numeric_limits<double>::max();
    //     double Jnew = J_(f_old.get(), g_old.get(), l_old.get());
    //     const SpMatrix<double>& Zero = SpMatrix<double>(N, N);
    //     double alpha = 2 * lambda_T() / std::pow(DeltaT_, 2);
    // 	int i = 1;
    //     A_ = SparseBlockMatrix<double, 3, 3>(
    //       PsiTD() * Psi(),    lambda_D() * R1().transpose(), -alpha * R0(),
    // 	  lambda_D() * R1(), -lambda_D() * R0(),              Zero,
    //       -alpha * R0(),      Zero,                          -lambda_T() * R0());
    //     invA_.compute(A_);
    //     b_.resize(3 * N);
    //     // main loop
    //     while (i < max_iter_ && std::abs((Jnew - Jold) / Jnew) > tol_) {
    //         // at step 0 f^(-1,i-1) = l^(-1,i-1) = 0
    //         b_ << PsiTD() * y(0) - 2 * alpha * R0() * l_old(1), lambda_D() * u(0), -2 * alpha * R0() * f_old(1);
    //         solve_(0, f_new, g_new, l_new);
    //         // general step
    //         for (int k = 1; k < m - 1; ++k) {
    //             b_ << PsiTD() * y(k) + 2 * alpha * R0() * (l_old(k + 1) + l_old(k - 1)), lambda_D() * u(k),
    //               -2 * alpha * R0() * (f_old(k + 1) + f_old(k - 1));
    //             solve_(k, f_new, g_new, l_new);
    //         }
    //         // at last step f^(m+1,i-1) = l^(m+1, i-1) = 0
    //         b_ << PsiTD() * y(m - 1) - 2 * alpha * R0() * l_old(m - 1), lambda_D() * u(m - 1),
    //           -2 * alpha * R0() * f_old(m - 1);
    //         solve_(m - 1, f_new, g_new, l_new);
    //         // prepare for next iteration
    //         Jold = Jnew;
    //         f_old = f_new; g_old = g_new; l_old = l_new;
    //         Jnew = J_(f_old.get(), g_old.get(), l_old.get());
    //         i++;
    //     }
    //     f_ = f_old.get();
    //     g_ = g_old.get();
    //     return;
    // }
  
// implementation of STRPDE for parabolic space-time regularization, monolithic approach
template <>
class STRPDE<SpaceTimeParabolic, monolithic> :
    public RegressionBase<STRPDE<SpaceTimeParabolic, monolithic>, SpaceTimeParabolic> {
   private:
    SparseBlockMatrix<double, 2, 2> A_ {};      // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
    DVector<double> b_ {};                      // right hand side of problem's linear system (1 x 2N vector)
    SpMatrix<double> L_;                        // L \kron R0
   public:
    using RegularizationType = SpaceTimeParabolic;
    using Base = RegressionBase<STRPDE<RegularizationType, monolithic>, RegularizationType>;
    // import commonly defined symbols from base
    IMPORT_REGRESSION_SYMBOLS;
    using Base::L;          // [L]_{ii} = 1/DeltaT for i \in {1 ... m} and [L]_{i,i-1} = -1/DeltaT for i \in {1 ... m-1}
    using Base::lambda_D;   // smoothing parameter in space
    using Base::lambda_T;   // smoothing parameter in time
    using Base::n_temporal_locs;   // number of time instants m defined over [0,T]
    using Base::s;                 // initial condition
    // constructor
    STRPDE() = default;
    STRPDE(const pde_ptr& pde, Sampling s, const DMatrix<double>& time) : Base(pde, s, time) {};

    void init_model() {   // update model object in case of **structural** changes in its definition
        if (runtime().query(runtime_status::is_lambda_changed)) {
            // assemble system matrix for the nonparameteric part of the model
            if (is_empty(L_)) L_ = Kronecker(L(), pde().mass());
            A_ = SparseBlockMatrix<double, 2, 2>(
              -PsiTD() * W() * Psi(),                lambda_D() * (R1() + lambda_T() * L_).transpose(),
              lambda_D() * (R1() + lambda_T() * L_), lambda_D() * R0()                                );
            // cache system matrix for reuse
            invA_.compute(A_);
            // prepare rhs of linear system
            b_.resize(A_.rows());
            b_.block(A_.rows() / 2, 0, A_.rows() / 2, 1) = lambda_D() * u();
            return;
        }
        if (runtime().query(runtime_status::require_W_update)) {
            // adjust north-west block of matrix A_ and factorize
            A_.block(0, 0) = -PsiTD() * W() * Psi();
            invA_.compute(A_);
            return;
        }
    }
    void solve() {   // finds a solution to the smoothing problem
        BLOCK_FRAME_SANITY_CHECKS;
        DVector<double> sol;   // room for problem' solution

        if (!Base::has_covariates()) {   // nonparametric case
            // update rhs of STR-PDE linear system
            b_.block(0, 0, A_.rows() / 2, 1) = -PsiTD() * W() * y();
            // solve linear system A_*x = b_
            sol = invA_.solve(b_);
            f_ = sol.head(A_.rows() / 2);
        } else {   // parametric case
            // rhs of STR-PDE linear system
            b_.block(0, 0, A_.rows() / 2, 1) = -PsiTD() * lmbQ(y());   // -\Psi^T*D*Q*z

            // definition of matrices U and V  for application of woodbury formula
            U_ = DMatrix<double>::Zero(A_.rows(), q());
            U_.block(0, 0, A_.rows() / 2, q()) = PsiTD() * W() * X();
            V_ = DMatrix<double>::Zero(q(), A_.rows());
            V_.block(0, 0, q(), A_.rows() / 2) = X().transpose() * W() * Psi();
            // solve system (A_ + U_*(X^T*W_*X)*V_)x = b using woodbury formula from NLA module
            sol = SMW<>().solve(invA_, U_, XtWX(), V_, b_);
            // store result of smoothing
            f_ = sol.head(A_.rows() / 2);
            beta_ = invXtWX().solve(X().transpose() * W()) * (y() - Psi() * f_);
        }
        // store PDE misfit
        g_ = sol.tail(A_.rows() / 2);
        return;
    }

    // getters
    const SparseBlockMatrix<double, 2, 2>& A() const { return A_; }
    const fdapde::SparseLU<SpMatrix<double>>& invA() const { return invA_; }
    double norm(const DMatrix<double>& op1, const DMatrix<double>& op2) const {   // euclidian norm of op1 - op2
        return (op1 - op2).squaredNorm(); // NB: to check, defined just for compiler
    }
  
    virtual ~STRPDE() = default;
};

// implementation of STRPDE for parabolic space-time regularization, iterative approach
template <>
class STRPDE<SpaceTimeParabolic, iterative> :
    public RegressionBase<STRPDE<SpaceTimeParabolic, iterative>, SpaceTimeParabolic> {
   private:
    SparseBlockMatrix<double, 2, 2> A_ {};      // system matrix of non-parametric problem (2N x 2N matrix)
    fdapde::SparseLU<SpMatrix<double>> invA_;   // factorization of matrix A
    DVector<double> b_ {};                      // right hand side of problem's linear system (1 x 2N vector)

    // the functional minimized by the iterative scheme
    // J(f,g) = \sum_{k=1}^m (z^k - \Psi*f^k)^T*(z^k - \Psi*f^k) + \lambda_S*(g^k)^T*(g^k)
<<<<<<< Updated upstream
    double J(const DMatrix<double>& f, const DMatrix<double>& g) const {
        double SSE = 0;
        // SSE = \sum_{k=1}^m (z^k - \Psi*f^k)^T*(z^k - \Psi*f^k)
        for (std::size_t t = 0; t < n_temporal_locs(); ++t) {
            SSE += (y(t) - Psi() * f.block(n_spatial_basis() * t, 0, n_spatial_basis(), 1)).squaredNorm();
=======
    double J_(const DMatrix<double>& f, const DMatrix<double>& g) const {
        double sse = 0;   // \sum_{k=1}^m (z^k - \Psi*f^k)^T*(z^k - \Psi*f^k)
        for (int t = 0; t < n_temporal_locs(); ++t) {
            sse += (y(t) - Psi() * f.block(n_spatial_basis() * t, 0, n_spatial_basis(), 1)).squaredNorm();
>>>>>>> Stashed changes
        }
        return sse + lambda_D() * g.squaredNorm();
    }
<<<<<<< Updated upstream
    // internal solve routine used by the iterative method
    void solve(std::size_t t, BlockVector<double>& f_new, BlockVector<double>& g_new) const {
=======
    void solve_(int t, BlockVector<double>& f_new, BlockVector<double>& g_new) const {
>>>>>>> Stashed changes
        DVector<double> x = invA_.solve(b_);
        f_new(t) = x.topRows(n_spatial_basis());
        g_new(t) = x.bottomRows(n_spatial_basis());
    }
    // internal utilities
    DMatrix<double> y(std::size_t k) const { return y().block(n_spatial_locs() * k, 0, n_spatial_locs(), 1); }
    DMatrix<double> u(std::size_t k) const { return u_.block(n_basis() * k, 0, n_basis(), 1); }

    // quantities related to iterative scheme
<<<<<<< Updated upstream
    double tol_ = 1e-4;           // tolerance used as stopping criterion
    std::size_t max_iter_ = 50;   // maximum number of allowed iterations
=======
    double tol_ = 1e-4;   // tolerance used as stopping criterion
    int max_iter_ = 50;   // maximum number of iterations
>>>>>>> Stashed changes
   public:
    using RegularizationType = SpaceTimeParabolic;
    using Base = RegressionBase<STRPDE<RegularizationType, iterative>, RegularizationType>;
    // import commonly defined symbols from base
    IMPORT_REGRESSION_SYMBOLS;
    using Base::DeltaT;            // distance between two time instants
    using Base::lambda_D;          // smoothing parameter in space
    using Base::lambda_T;          // smoothing parameter in time
    using Base::n_temporal_locs;   // number of time instants m defined over [0,T]
    using Base::pde_;              // parabolic differential operator df/dt + Lf - u
    // constructor
    STRPDE() = default;
    STRPDE(const pde_ptr& pde, Sampling s, const DMatrix<double>& time) : Base(pde, s, time) { pde_.init(); };

    void tensorize_psi() { return; } // avoid tensorization of \Psi matrix
    void init_regularization() {
        pde_.init();
        // compute time step (assuming equidistant points)
        DeltaT_ = time_[1] - time_[0];
        u_ = pde_.force();   // compute forcing term
        // correct first n rows of discretized force as (u_1 + R0*s/DeltaT)
        u_.block(0, 0, n_basis(), 1) += (1.0 / DeltaT_) * (pde_.mass() * s_);
    }
    // getters
<<<<<<< Updated upstream
    const SpMatrix<double>& R0() const { return pde_.mass(); }    // mass matrix in space
    const SpMatrix<double>& R1() const { return pde_.stiff(); }   // discretization of differential operator L
    std::size_t n_basis() const { return pde_.n_dofs(); }         // number of basis functions

    void init_model() { return; };
    void solve() {
        // compute starting point (f^(k,0), g^(k,0)) k = 1 ... m for iterative minimization of functional J(f,g)
=======
    const SpMatrix<double>& R0() const { return pde_.mass(); }
    const SpMatrix<double>& R1() const { return pde_.stiff(); }
    int n_basis() const { return pde_.n_dofs(); }

    void init_model() { return; };
    void solve() {
        fdapde_assert(y().rows() != 0);
        int N = n_spatial_basis(), m = n_temporal_locs();
        // compute starting point (f^(k,0), g^(k,0)) k = 1 ... m for iterative minimization of J(f,g)
>>>>>>> Stashed changes
        A_ = SparseBlockMatrix<double, 2, 2>(
          PsiTD() * Psi(),   lambda_D() * R1().transpose(),
	  lambda_D() * R1(), -lambda_D() * R0()           );
        invA_.compute(A_);
        b_.resize(A_.rows());
        // compute f^(k,0), k = 1 ... m as solution of Ax = b_(k)
<<<<<<< Updated upstream
        BlockVector<double> f_old(n_temporal_locs(), n_spatial_basis());
        // solve n_temporal_locs() space only linear systems
        for (std::size_t t = 0; t < n_temporal_locs(); ++t) {
            // right hand side at time step t
            b_ << PsiTD() * y(t),   // should put W()
              lambda_D() * lambda_T() * u(t);
            // solve linear system Ax = b_(t) and store estimate of spatial field
            f_old(t) = invA_.solve(b_).head(A_.rows() / 2);
=======
        BlockVector<double> f_old(m, N);
        for (int k = 0; k < m; k++) {   // solve m space only linear systems
            b_ << PsiTD() * y(k), lambda_D() * lambda_T() * u(k);
            f_old(k) = invA_.solve(b_).head(A_.rows() / 2);
>>>>>>> Stashed changes
        }

	// factorization of G0 = [(\lambda_S*\lambda_T)/DeltaT * R_0 + \lambda_S*R_1^T]
        SpMatrix<double> G0 =
          (lambda_D() * lambda_T() / DeltaT()) * R0() + SpMatrix<double>((lambda_D() * R1()).transpose());
        Eigen::SparseLU<SpMatrix<double>, Eigen::COLAMDOrdering<int>> invG0;
        invG0.compute(G0);

        // compute g^(k,0), k = 1 ... m as solution (in backward order) of the system
        // G0*g^(k,0) = \Psi^T*y^k + (\lambda_S*\lambda_T/DeltaT*R_0)*g^(k+1,0) - \Psi^T*\Psi*f^(k,0)
        BlockVector<double> g_old(m, N);
        b_ = PsiTD() * (y(m - 1) - Psi() * f_old(m - 1));   // g^(k+1,0) = 0
        g_old(m - 1) = invG0.solve(b_);
        for (int k = m - 2; k >= 0; k--) {
            b_ = PsiTD() * (y(k) - Psi() * f_old(k)) + (lambda_D() * lambda_T() / DeltaT()) * R0() * g_old(k + 1);
            g_old(k) = invG0.solve(b_);
        }

        // iterative scheme initialization
        BlockVector<double> f_new(m, N), g_new(m, N);
        double Jold = std::numeric_limits<double>::max();
<<<<<<< Updated upstream
        double Jnew = J(f_old.get(), g_old.get());
        std::size_t i = 1;   // iteration number

        // build system matrix for the iterative scheme
=======
        double Jnew = J_(f_old.get(), g_old.get());
        int i = 1;
>>>>>>> Stashed changes
        A_.block(0, 1) += lambda_D() * lambda_T() / DeltaT() * R0();
        A_.block(1, 0) += lambda_D() * lambda_T() / DeltaT() * R0();
        invA_.compute(A_);
        b_.resize(A_.rows());
	// main loop
        while (i < max_iter_ && std::abs((Jnew - Jold) / Jnew) > tol_) {
            // at step 0 f^(k-1,i-1) = 0
            b_ << PsiTD() * y(0) + (lambda_D() * lambda_T() / DeltaT()) * R0() * g_old(1), lambda_D() * u(0);
<<<<<<< Updated upstream
            // solve linear system
            solve(0, f_new, g_new);

            // general step
            for (std::size_t t = 1; t < n_temporal_locs() - 1; ++t) {
                // \Psi^T*y^k   + (\lambda_D*\lambda_T/DeltaT)*R_0*g^(k+1,i-1),
                // \lambda_D*u^k + (\lambda_D*\lambda_T/DeltaT)*R_0*f^(k-1,i-1)
                b_ << PsiTD() * y(t) + (lambda_D() * lambda_T() / DeltaT()) * R0() * g_old(t + 1),
                  lambda_D() * (lambda_T() / DeltaT() * R0() * f_old(t - 1) + u(t));
                // solve linear system
                solve(t, f_new, g_new);
            }

            // at last step g^(k+1,i-1) is zero
            b_ << PsiTD() * y(n_temporal_locs() - 1),
              lambda_D() * (lambda_T() / DeltaT() * R0() * f_old(n_temporal_locs() - 2) + u(n_temporal_locs() - 1));
            // solve linear system
            solve(n_temporal_locs() - 1, f_new, g_new);

=======
            solve_(0, f_new, g_new);
            // general step
            for (int k = 1; k < m - 1; ++k) {
                b_ << PsiTD() * y(k) + (lambda_D() * lambda_T() / DeltaT()) * R0() * g_old(k + 1),
                  lambda_D() * (lambda_T() / DeltaT() * R0() * f_old(k - 1) + u(k));
                solve_(k, f_new, g_new);
            }
            // at last step g^(k+1,i-1) = 0
            b_ << PsiTD() * y(m - 1),
              lambda_D() * (lambda_T() / DeltaT() * R0() * f_old(m - 2) + u(m - 1));
            solve_(m - 1, f_new, g_new);
>>>>>>> Stashed changes
            // prepare for next iteration
            Jold = Jnew;
            f_old = f_new;
            g_old = g_new;
            Jnew = J_(f_old.get(), g_old.get());
            i++;
        }
<<<<<<< Updated upstream

        // store solution
=======
>>>>>>> Stashed changes
        f_ = f_old.get();
        g_ = g_old.get();
        return;
    }

    // setters
    void set_tolerance(double tol) { tol_ = tol; }
    void set_max_iter(std::size_t max_iter) { max_iter_ = max_iter; }

    virtual ~STRPDE() = default;
};

}   // namespace models
}   // namespace fdapde

#endif   // __STRPDE_H__
