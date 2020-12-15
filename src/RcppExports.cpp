// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// refit_LA_ADMM_export
Rcpp::List refit_LA_ADMM_export(const int& it_out, const int& it_in, const arma::mat& S, const arma::mat& A, const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D, const double& rho, const arma::mat& omP, const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u2, const arma::mat& init_u3, const arma::mat& init_gam, const arma::mat& init_u4, const arma::mat& init_u5);
RcppExport SEXP _taglasso_refit_LA_ADMM_export(SEXP it_outSEXP, SEXP it_inSEXP, SEXP SSEXP, SEXP ASEXP, SEXP AtildeSEXP, SEXP A_for_gammaSEXP, SEXP A_for_BSEXP, SEXP CSEXP, SEXP C_for_DSEXP, SEXP rhoSEXP, SEXP omPSEXP, SEXP init_omSEXP, SEXP init_u1SEXP, SEXP init_u2SEXP, SEXP init_u3SEXP, SEXP init_gamSEXP, SEXP init_u4SEXP, SEXP init_u5SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const int& >::type it_out(it_outSEXP);
    Rcpp::traits::input_parameter< const int& >::type it_in(it_inSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Atilde(AtildeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A_for_gamma(A_for_gammaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A_for_B(A_for_BSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type C(CSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type C_for_D(C_for_DSEXP);
    Rcpp::traits::input_parameter< const double& >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type omP(omPSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_om(init_omSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u1(init_u1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u2(init_u2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u3(init_u3SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_gam(init_gamSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u4(init_u4SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u5(init_u5SEXP);
    rcpp_result_gen = Rcpp::wrap(refit_LA_ADMM_export(it_out, it_in, S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, rho, omP, init_om, init_u1, init_u2, init_u3, init_gam, init_u4, init_u5));
    return rcpp_result_gen;
END_RCPP
}
// LA_ADMM_taglasso_export
Rcpp::List LA_ADMM_taglasso_export(const int& it_out, const int& it_in, const arma::mat& S, const arma::mat& A, const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D, const double& lambda1, const double& lambda2, const double& rho, const bool& pendiag, const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u2, const arma::mat& init_u3, const arma::mat& init_gam, const arma::mat& init_u4, const arma::mat& init_u5);
RcppExport SEXP _taglasso_LA_ADMM_taglasso_export(SEXP it_outSEXP, SEXP it_inSEXP, SEXP SSEXP, SEXP ASEXP, SEXP AtildeSEXP, SEXP A_for_gammaSEXP, SEXP A_for_BSEXP, SEXP CSEXP, SEXP C_for_DSEXP, SEXP lambda1SEXP, SEXP lambda2SEXP, SEXP rhoSEXP, SEXP pendiagSEXP, SEXP init_omSEXP, SEXP init_u1SEXP, SEXP init_u2SEXP, SEXP init_u3SEXP, SEXP init_gamSEXP, SEXP init_u4SEXP, SEXP init_u5SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const int& >::type it_out(it_outSEXP);
    Rcpp::traits::input_parameter< const int& >::type it_in(it_inSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type S(SSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Atilde(AtildeSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A_for_gamma(A_for_gammaSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A_for_B(A_for_BSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type C(CSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type C_for_D(C_for_DSEXP);
    Rcpp::traits::input_parameter< const double& >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< const double& >::type lambda2(lambda2SEXP);
    Rcpp::traits::input_parameter< const double& >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< const bool& >::type pendiag(pendiagSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_om(init_omSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u1(init_u1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u2(init_u2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u3(init_u3SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_gam(init_gamSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u4(init_u4SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type init_u5(init_u5SEXP);
    rcpp_result_gen = Rcpp::wrap(LA_ADMM_taglasso_export(it_out, it_in, S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, lambda1, lambda2, rho, pendiag, init_om, init_u1, init_u2, init_u3, init_gam, init_u4, init_u5));
    return rcpp_result_gen;
END_RCPP
}
// determine_dfs
int determine_dfs(arma::mat A, const double& tol);
RcppExport SEXP _taglasso_determine_dfs(SEXP ASEXP, SEXP tolSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type A(ASEXP);
    Rcpp::traits::input_parameter< const double& >::type tol(tolSEXP);
    rcpp_result_gen = Rcpp::wrap(determine_dfs(A, tol));
    return rcpp_result_gen;
END_RCPP
}
// matrix_unique_rows
double matrix_unique_rows(const arma::mat& A);
RcppExport SEXP _taglasso_matrix_unique_rows(SEXP ASEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    rcpp_result_gen = Rcpp::wrap(matrix_unique_rows(A));
    return rcpp_result_gen;
END_RCPP
}
// ADMM_taglasso_cv_parallel
Rcpp::List ADMM_taglasso_cv_parallel(const arma::vec& fold_vector, const arma::vec& lambda1_vector, const arma::vec& lambda1_index_vector, const arma::vec& lambda2_vector, const arma::vec& lambda2_index_vector, const arma::cube& Sin, const arma::cube& Sout, const int& fold, const arma::mat& A, const arma::vec& lambda1, const arma::vec& lambda2, const double& rho, const bool& pendiag, const int& it_out, const int& it_in, const int& it_out_refit, const int& it_in_refit, const bool& do_parallel, const int& nc);
RcppExport SEXP _taglasso_ADMM_taglasso_cv_parallel(SEXP fold_vectorSEXP, SEXP lambda1_vectorSEXP, SEXP lambda1_index_vectorSEXP, SEXP lambda2_vectorSEXP, SEXP lambda2_index_vectorSEXP, SEXP SinSEXP, SEXP SoutSEXP, SEXP foldSEXP, SEXP ASEXP, SEXP lambda1SEXP, SEXP lambda2SEXP, SEXP rhoSEXP, SEXP pendiagSEXP, SEXP it_outSEXP, SEXP it_inSEXP, SEXP it_out_refitSEXP, SEXP it_in_refitSEXP, SEXP do_parallelSEXP, SEXP ncSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type fold_vector(fold_vectorSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda1_vector(lambda1_vectorSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda1_index_vector(lambda1_index_vectorSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda2_vector(lambda2_vectorSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda2_index_vector(lambda2_index_vectorSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type Sin(SinSEXP);
    Rcpp::traits::input_parameter< const arma::cube& >::type Sout(SoutSEXP);
    Rcpp::traits::input_parameter< const int& >::type fold(foldSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type A(ASEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda1(lambda1SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type lambda2(lambda2SEXP);
    Rcpp::traits::input_parameter< const double& >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< const bool& >::type pendiag(pendiagSEXP);
    Rcpp::traits::input_parameter< const int& >::type it_out(it_outSEXP);
    Rcpp::traits::input_parameter< const int& >::type it_in(it_inSEXP);
    Rcpp::traits::input_parameter< const int& >::type it_out_refit(it_out_refitSEXP);
    Rcpp::traits::input_parameter< const int& >::type it_in_refit(it_in_refitSEXP);
    Rcpp::traits::input_parameter< const bool& >::type do_parallel(do_parallelSEXP);
    Rcpp::traits::input_parameter< const int& >::type nc(ncSEXP);
    rcpp_result_gen = Rcpp::wrap(ADMM_taglasso_cv_parallel(fold_vector, lambda1_vector, lambda1_index_vector, lambda2_vector, lambda2_index_vector, Sin, Sout, fold, A, lambda1, lambda2, rho, pendiag, it_out, it_in, it_out_refit, it_in_refit, do_parallel, nc));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_taglasso_refit_LA_ADMM_export", (DL_FUNC) &_taglasso_refit_LA_ADMM_export, 18},
    {"_taglasso_LA_ADMM_taglasso_export", (DL_FUNC) &_taglasso_LA_ADMM_taglasso_export, 20},
    {"_taglasso_determine_dfs", (DL_FUNC) &_taglasso_determine_dfs, 2},
    {"_taglasso_matrix_unique_rows", (DL_FUNC) &_taglasso_matrix_unique_rows, 1},
    {"_taglasso_ADMM_taglasso_cv_parallel", (DL_FUNC) &_taglasso_ADMM_taglasso_cv_parallel, 19},
    {NULL, NULL, 0}
};

RcppExport void R_init_taglasso(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
