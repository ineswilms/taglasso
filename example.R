rm(list=ls())

#### Install package from GitHub ####
install.packages("devtools")
devtools::install_github("ineswilms/taglasso")
library(taglasso)

#### Pre-process the data ####
data('rv')
rv_data <- rv$data
A <- rv$A
# Estimate HAR models
estimate.HAR <- function(y){
  # Function : HAR model with daily, weekly and monthly computed for the realized variances

  # INPUT
  # XHAR (matrix) : Tx22 : RV_{t-k} for k=1, ..., 22
  # YHAR (vector) : Tx1 : RV_t
  # Xtest
  # Ytest

  # OUTPUT
  # beta : estimated parameter (matrix : 3x1)
  # BIC : BIC value

  HARdata <- embed(y, 22+1)
  XHAR <- HARdata[, -1]

  # Start Function
  YHAR <- HARdata[,1]
  X.D <- as.matrix(XHAR[,1])
  X.W <- as.matrix(apply(XHAR[,1:5]/5,1,sum))
  X.M <- as.matrix(apply(XHAR[,]/22,1,sum))
  X.HAR <- cbind(1, X.D,X.W,X.M)
  beta.HAR <- solve(t(X.HAR)%*%X.HAR)%*%t(X.HAR)%*%YHAR
  resid.HAR <- YHAR - X.HAR%*%beta.HAR
  return(resid.HAR)

}
resid_HAR <- apply(rv_data, 2, estimate.HAR)
data <- resid_HAR

#### 5-fold cross-validation to select the regularization parameters ####
library(parallel)
ptm <- proc.time()
rv_taglasso_cv <- taglasso_cv(X = data, A = A, seed = floor(abs(data[1]*1000)), fold = 5,
                              l1gran = 5, l2gran = 5, nc = detectCores()-1, do_parallel = FALSE)
proc.time() - ptm

#### tag-lasso fit ####
rv_taglasso <- taglasso(X = data, A = rv_A, lambda1 = rv_taglasso_cv$l1opt, lambda2 = rv_taglasso_cv$l2opt, hc = TRUE, plot = TRUE)

#### networks ####
library(corrplot)
corrplot(rv_taglasso$omega_aggregated!=0, cl.pos = "n", tl.cex = 1.5,
         method = "color", main = "" ,
         mar = c(0,0,0.5,0), addgrid.col = "black", cex.main = 1.5, font.main = 1,
         is.corr = F, col=c("#F0F0F0", "White", "Black"),
         tl.col = "black")
corrplot(rv_taglasso$omega_full!=0, cl.pos = "n", tl.cex = 1.5,
         method = "color", main = "" ,
         mar = c(0,0,0.5,0), addgrid.col = "black", cex.main = 1.5, font.main = 1,
         is.corr = F, col=c("#F0F0F0", "White", "Black"),
         tl.col = "black")
