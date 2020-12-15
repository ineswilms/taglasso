#### Some auxiliary functions ####

preliminaries_for_refit_in_R <- function(A){
  # Preliminaries on A to be used when solving for D, Omega^{(2)} and Gamma^{(2)}
  z <- ncol(A)
  p <- nrow(A)
  Atilde <- rbind(A, diag(1, z))
  A_for_gamma <- solve(t(A)%*%A + diag(1, z))%*%cbind(t(A), diag(1, z))
  A_for_B <- diag(1, p+z) - Atilde%*% solve(t(Atilde)%*%Atilde) %*%t(Atilde)
  C <- t(cbind(diag(1, p), matrix(0, p, z))) - Atilde%*%solve(t(Atilde)%*%Atilde)%*%t(A)
  C_for_D <- solve(diag(diag(t(C)%*%C)))
  out <- list("A_for_gamma" = A_for_gamma, "Atilde" = Atilde, "A_for_B" = A_for_B, "C" = C, "C_for_D" = C_for_D)
}

check_aggregation_and_sparsity <- function (fit, A) {
  # Determine aggregation and sparsity structure from fit

  p <- dim(fit$gam1)[2]
  Z <- which(apply(fit$gam1, 1, function(U) {
    all(U == 0)
  }) == F)
  if (length(Z) == 0) {
    Z <- nrow(fit$gam1)
  }
  AZ <- A[, Z]
  if (length(Z) == 1) {
    AZ = matrix(AZ, p, 1)
  }
  agg <- AZ
  unique_rows <- unique(agg)
  cluster <- rep(NA, nrow(agg))
  for (i in 1:nrow(unique_rows)) {
    check <- apply(agg, 1, unique_rows = unique_rows, function(unique_rows, X) {
      all(X == unique_rows[i, ])
    })
    cluster[which(check == T)] <- i
  }
  my_clusters <- length(unique(cluster))
  om_P <- (fit$om3 != 0) * 1
  my_sparsity <- length(which(om_P == 0))
  prelim_AZ <- preliminaries_for_refit_in_R(AZ)

  out <- list("my_clusters" = my_clusters, "my_sparsity" = my_sparsity,
              "cluster" = cluster, "omP" = om_P, "AZ" = AZ)
}

get_grid_bounds <- function(data, S = stats::cov(data), A, prelim_A, pendiag = F, max_iter = 10,
                                get_lambda1 = NULL, l1_length = 10, l1_ratio = 10^2,  l1_max = 5,
                                get_lambda2 = NULL, l2_length = 10, l2_ratio = 10^2, l2_max = max(max(stats::cor(data) - diag(ncol(data))), -min(stats::cor(data) - diag(ncol(data)))),
                                it_out = 10, it_in = 100,
                                rho = 0.01){
  # Preliminaries
  p <- ncol(data)
  ominit <- matrix(0, p, p)
  gaminit <- matrix(0, ncol(A), nrow(A))
  if (is.null(get_lambda1)) {
    get_lambda1 <- c(exp(seq(log(l1_max), log(l1_max/l1_ratio), length = l1_length)))
  }
  if (is.null(get_lambda2)) {
    get_lambda2 <- c(exp(seq(log(l2_max), log(l2_max/l2_ratio), length = l2_length)))
  }

  it <- 1
  l1_max <- get_lambda1[1]
  l2_max <- get_lambda2[1]
  nbr_clusters <- p
  nbr_sparse <- 0
  max_sparsity <- p * p - p
  while ((nbr_clusters != 1 | nbr_sparse != max_sparsity) & it <= max_iter) {
    my_fit <- LA_ADMM_taglasso_export(it_out = it_out,
                                     it_in = it_in, S = S, A = A,  Atilde = prelim_A$Atilde,
                                     A_for_gamma = prelim_A$A_for_gamma, A_for_B = prelim_A$A_for_B,
                                     C = prelim_A$C, C_for_D = prelim_A$C_for_D, lambda1 = l1_max,
                                     lambda2 = l2_max, rho = rho, pendiag = pendiag, init_om = ominit,
                                     init_u1 = ominit, init_u2 = ominit, init_u3 = ominit,
                                     init_gam = gaminit, init_u4 = gaminit, init_u5 = gaminit)
    first_check <- check_aggregation_and_sparsity(fit = my_fit, A = A)
    nbr_clusters <- first_check$my_clusters
    nbr_sparse <- first_check$my_sparsity
    if (nbr_clusters != 1) {
      l1_max <- l1_max * 1.5
    }
    if (nbr_sparse != max_sparsity) {
      l2_max <- l2_max * 1.5
    }
    it <- it + 1
  }

  max_iter <- 10
  nbr_clusters <- p
  it <- 1
  l1_min <- get_lambda1[length(get_lambda1)]
  while (nbr_clusters == p & it <= max_iter) {
    my_fit <- LA_ADMM_taglasso_export(it_out = it_out,
                                     it_in = it_in, S = S, A = A, Atilde = prelim_A$Atilde,
                                     A_for_gamma = prelim_A$A_for_gamma, A_for_B = prelim_A$A_for_B,
                                     C = prelim_A$C, C_for_D = prelim_A$C_for_D, lambda1 = l1_min,
                                     lambda2 = 0, rho = rho, pendiag = pendiag, init_om = ominit,
                                     init_u1 = ominit, init_u2 = ominit, init_u3 = ominit,
                                     init_gam = gaminit, init_u4 = gaminit, init_u5 = gaminit)
    first_check <- check_aggregation_and_sparsity(my_fit,
                                                  A)
    nbr_clusters <- first_check$my_clusters
    if (nbr_clusters == p) {
      l1_min <- l1_min * 1.5
    }
    if (nbr_clusters != p) {
      l1_min <- l1_min/1.5
      break
    }
    it <- it + 1
  }
  max_iter <- 10
  nbr_clusters <- p
  it <- 1
  l2_min <- get_lambda2[length(get_lambda2)]
  nbr_sparsity <- max_sparsity
  while (nbr_clusters == p & it <= max_iter) {
    my_fit <- LA_ADMM_taglasso_export(it_out = it_out,
                                     it_in = it_in, S = S, A = A,  Atilde = prelim_A$Atilde,
                                     A_for_gamma = prelim_A$A_for_gamma, A_for_B = prelim_A$A_for_B,
                                     C = prelim_A$C, C_for_D = prelim_A$C_for_D, lambda1 = l1_min,
                                     lambda2 = l2_min, rho = rho, pendiag = pendiag, init_om = ominit,
                                     init_u1 = ominit, init_u2 = ominit, init_u3 = ominit,
                                     init_gam = gaminit, init_u4 = gaminit, init_u5 = gaminit)
    first_check <- check_aggregation_and_sparsity(my_fit,
                                                  A)
    nbr_sparsity <- first_check$my_sparsity
    if (nbr_sparsity == 0) {
      l2_min <- l2_min * 1.5
    }
    if (nbr_sparsity != 0) {
      l2_min <- l2_min/1.5
      break
    }
    it <- it + 1
  }
  get_lambda1 <- c(exp(seq(log(l1_max), log(l1_min), length = length(get_lambda1))))
  get_lambda2 <- c(exp(seq(log(l2_max), log(l2_min), length = length(get_lambda2))))
  out <- list("get_lambda1" = get_lambda1, "get_lambda2" = get_lambda2)
}
