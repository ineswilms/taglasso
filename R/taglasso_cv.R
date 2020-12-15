#' K-fold cross-validation for tag-lasso estimation of the precision matrix
#' @export
#' @description This function performs K-fold cross-validation to select the tuning parameters lambda1 and lambda2
#' @param X An (\eqn{n}x\eqn{p})-matrix of \eqn{p} variables and \eqn{n} observations
#' @param A An (\eqn{p}x\eqn{|T|})- binary matrix incorporating the tree-based aggregation structure
#' @param pendiag Logical indicator whether or not to penalize the diagonal in Omega. The default is \code{TRUE} (penalization of the diagonal)
#' @param seed Set the seed to ensure reproducible results for the K-fold cross-validation exercise
#' @param fold Set the fold for the cross-validation exercise. Default is fold = 5 for 5-fold cross-validation
#' @param lambda1 Numeric vector for the aggregation tuning parameter. Default is \code{NULL}, then the program determines this internally.
#' @param l1length Number of aggregation tuning parameters to be considered. Default is 10
#' @param l1gran Ratio of largest to smallest aggregation tuning parameter. Default is 10^2
#' @param l1max Largest value of the aggregation tuning parameter to be considered. Default: the program determines this internally based on starting estimate of 5.
#' @param lambda2 Numeric vector for the sparsity tuning parameter. Default is \code{NULL}, then the program determines this internally.
#' @param l2length Number of sparsity tuning parameters to be considered. Default is 10
#' @param l2gran Ratio of largest to smallest sparsity tuning parameter. Default is 10^2
#' @param l2max Largest value of the sparsity tuning parameter to be considered. Default: the program determines this internally.
#' @param rho Starting value for the LA-ADMM tuning parameter. Default is 10^2; will be locally adjusted via LA-ADMM
#' @param it_in Number of inner stages of the LA-ADMM algorithm. Default is 100
#' @param it_out Number of outer stages of the LA-ADMM algorithm. Default is 10
#' @param it_in_refit Number of inner stages of the LA-ADMM algorithm for re-fitting. Default is 100
#' @param it_out_refit Number of outer stages of the LA-ADMM algorithm for re-fitting. Default is 10
#' @param do_parallel Logical indicator whether K-fold cross-validation should be executed in parallel with OpenMP or not. Default is \code{FALSE}
#' @param nc Number of cores to be used in the parallel loop.
#' @return A list with the following components
#' \item{\code{l1vec}}{Numeric vector of aggregation tuning parameters}
#' \item{\code{l2vec}}{Numeric vector of sparsity tuning parameters}
#' \item{\code{l1opt}}{Optimal aggregation tuning parameter, as selected by K-fold cross-validation}
#' \item{\code{l2opt}}{Optimal sparsity tuning parameter, as selected by K-fold cross-validation}
#' \item{\code{cvobj}}{Cross-validation scores along the two-dimensional grid}
taglasso_cv <- function(X, A, pendiag = F,  seed, fold = 5,
                                lambda1 = NULL, l1length = 10, l1gran = 10^2, l1max = 5,
                                lambda2 = NULL, l2length = 10, l2gran = 10^2, l2max = max(max(stats::cor(X) - diag(ncol(X))), -min(stats::cor(X) - diag(ncol(X)))),
                                rho = 10^-2, it_in = 100, it_out = 10,  it_in_refit = 100, it_out_refit = 10,
                                do_parallel = TRUE, nc = 1){

  #### Data for K-fold cross validation ####
  # Dimensions
  p <- ncol(X)
  n <- nrow(X)
  # Training and Test observations
  set.seed(seed)
  resample <- sample(1:n, n)
  cvcut <- floor(n/fold)
  cutlast <- n - (fold - 1)*cvcut
  # Strain <- Stest  <- vector("list", fold)
  Sin <- Sout <- array(NA, c(p, p, fold))
  # for(icv in 1:fold){
  #   Sin[,,icv] = Strain[[icv]]
  #   Sout[,,icv] = Stest[[icv]]
  # }

  for(icv in 1:fold){
    if(icv < fold){
      train_obs <- resample[-c((1 + (icv - 1)*cvcut):(icv*cvcut))]
      # Strain[[icv]] <- cov(X[train_obs,])
      Sin[,,icv] <- stats::cov(X[train_obs,])
      test_obs <- resample[(1 + (icv - 1)*cvcut):(icv*cvcut)]
      # Stest[[icv]] <- cov(X[test_obs,])
      Sout[,,icv] <- stats::cov(X[test_obs,])
    }
    if(icv == fold){ # Last cut
      train_obs <- resample[-c((1 + (icv - 1)*cvcut):n)]
      # Strain[[icv]] <- cov(X[train_obs,])
      Sin[,,icv] <- stats::cov(X[train_obs,])
      test_obs <- resample[(1 + (icv - 1)*cvcut):n]
      # Stest[[icv]] <- cov(X[test_obs,])
      Sout[,,icv] <- stats::cov(X[test_obs,])
    }
  }


  #### Preliminaries for the A matrix ####
  A_precompute <- preliminaries_for_refit_in_R(A = A)


  #### Set grids for tuning parameters lambda1 and lambda2 ####
  if(is.null(lambda1) | is.null(lambda2)){
    grid <- get_grid_bounds(data = X, A = A, prelim_A = A_precompute, pendiag = pendiag,
                              get_lambda1 = lambda1, l1_length = l1length, l1_ratio = l1gran, l1_max = l1max,
                              get_lambda2 = lambda2, l2_length = l2length, l2_ratio = l2gran, l2_max = l2max,
                              it_out = it_out, it_in = it_in, rho = rho)
  }

  if(is.null(lambda1)){
    lambda1 <- grid$get_lambda1
  }
  if(is.null(lambda2)){
    lambda2 <- grid$get_lambda2
  }

  ##### K-fold cross-validation ####
  # Sin <- Sout <- array(NA, c(p, p, fold))
  # for(icv in 1:fold){
  #   Sin[,,icv] = Strain[[icv]]
  #   Sout[,,icv] = Stest[[icv]]
  # }
  # l1length = length(lambda1)
  # l2length = length(lambda2)
  folds_vector <- rep(1:fold, each = length(lambda1)*length(lambda2))
  l1_vector <- rep(rep(lambda1, each = length(lambda2)), fold)
  l1_vector_index <- rep(rep(1:length(lambda1), each = length(lambda2)), fold)
  l2_vector <- rep(rep(lambda2, length(lambda1)), fold)
  l2_vector_index <- rep(rep(1:length(lambda2), length(lambda1)), fold)

  ADMM_aglasso_cv_in_cpp <- ADMM_taglasso_cv_parallel(fold_vector = folds_vector, lambda1_vector = l1_vector, lambda2_vector = l2_vector,
                                                        lambda1_index_vector = l1_vector_index, lambda2_index_vector = l2_vector_index,
                                                        fold = fold, Sin = Sin, Sout = Sout, A = A,
                                                        lambda1 = lambda1, lambda2 = lambda2,
                                                        rho = rho, pendiag = pendiag, it_out = it_out, it_in = it_in,
                                                        it_out_refit = it_out_refit, it_in_refit = it_in_refit,
                                                        do_parallel = do_parallel, nc = nc)

  ##### Select regularization parameters ####
  cvobj <- apply(ADMM_aglasso_cv_in_cpp$cvobj, c(2, 3), mean)
  rownames(cvobj) <- paste0("lambda1 value", 1:l1length)
  colnames(cvobj) <- paste0("lambda2 value", 1:l2length)
  cvobj[which(cvobj== -Inf)] = NA
  cvobj[which(cvobj== Inf)] = NA
  l1matrix <- apply(ADMM_aglasso_cv_in_cpp$l1check, c(2, 3), unique)
  l2matrix <- apply(ADMM_aglasso_cv_in_cpp$l2check, c(2, 3), unique)
  l1opt <- l1matrix[which.min(cvobj)]
  l2opt <- l2matrix[which.min(cvobj)]


  out <- list("l1vec" = lambda1, "l2vec" = lambda2, "l1opt" = l1opt, "l2opt" = l2opt, "cvobj" = cvobj)
}

