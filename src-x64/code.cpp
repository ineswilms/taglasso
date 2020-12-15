// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace arma;

// Define output structures
struct sub_dog_out{
  arma::mat D;
  arma::mat Gamma;
  arma::mat Omega;
};

struct ADMM_block_out{
  arma::mat om1;
  arma::mat om2;
  arma::mat om3;
  arma::mat gam1;
  arma::mat gam2;
  arma::mat D;
  arma::mat omega;
  arma::mat gamma;
  arma::mat Atilde;
  arma::mat C;
  arma::mat u1;
  arma::mat u2;
  arma::mat u3;
  arma::mat u4;
  arma::mat u5;
  arma::mat omP;
};

struct ADMM_glasso_block_out{
  arma::mat om1;
  arma::mat om2;
  arma::mat omega;
  arma::mat u1;
  arma::mat u2;
};

struct LA_ADMM_out{
  arma::mat omega;
  arma::mat gamma;
  arma::mat om1;
  arma::mat om2;
  arma::mat om3;
  arma::mat gam1;
  arma::mat gam2;
  arma::mat D;
  arma::mat u1;
  arma::mat u2;
  arma::mat u3;
  arma::mat u4;
  arma::mat u5;
  arma::mat omP;
  double rho;
};

struct LA_ADMM_glasso_out{
  arma::mat omega;
  arma::mat om1;
  arma::mat om2;
  arma::mat u1;
  arma::mat u2;
  double rho;
};

struct prelim_A_out{
  arma::mat Atilde;
  arma::mat A_for_gamma;
  arma::mat A_for_B;
  arma::mat C;
  arma::mat C_for_D;
};

struct Areduced_out{
  arma::mat AZ;
  int znodes;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////// ORACLE FUNCTIONS /////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

arma::mat refit_omega_ed_sym(const arma::mat& S, const arma::mat& Omega, const arma::mat& U, const double& rho){

  arma::mat rhs = rho * Omega - U - S;
  rhs = (rhs + rhs.t())/2;

  vec eigval;
  mat eigvec;

  arma::eig_sym(eigval, eigvec, rhs);

  arma::mat Omegabar =  arma::diagmat((eigval + sqrt(square(eigval) + 4 * rho) ) / (2*rho));
  arma::mat Omeganew = eigvec * Omegabar *eigvec.t();

  return(Omeganew);

}

arma::mat refit_omega_soft(const arma::mat& Omega, const arma::mat& U, const double& rho, const arma::mat& omP){
  // Input
  // Omega : matrix of dimension p times p
  // U : dual variable, matrix of dimension p times p
  // rho : parameter ADMM
  // omP : matrix of dimension p times p with 0 or 1 as entries: 0 (zero-elements) and 1 (non-zero elements)

  // Function : Obtain estimate of Omega^(3) (5.2)

  // Output
  // Omeganew : matrix of dimension p times p

  int p = Omega.n_cols;
  arma::mat Omeganew = zeros(p, p);
  arma::mat soft_input = Omega - U/rho;

  arma::vec diago = zeros(p);

  for(int ir=0; ir < p; ++ir ){
    for(int ic=0; ic < p; ++ ic){
      if(omP(ir, ic)==0){
        Omeganew(ir, ic) = 0;
      }else{
        Omeganew(ir, ic) = soft_input(ir, ic);
      }
    }
  }

  return(Omeganew);
}

arma::mat refit_gamma_soft(const arma::mat& Gamma, const arma::mat& U, const double& rho){
  // Input
  // Gamma : matrix of dimension |Z| times p
  // U : dual variable, matrix of dimension |Z| times p
  // rho : parameter ADMM

  // Function : Obtain estimate of Gamma_Z^(1) (5.4)

  // Output
  // Gammanew : matrix of dimension |Z| times p

  int z = Gamma.n_rows;
  int p = Gamma.n_cols;
  arma::mat Gammanew = zeros(z, p);

  Gammanew = Gamma - U/rho;
  // double avgtl0 = mean(Gammanew.row(z-1)); // last row corresponds to the root
  // Rcpp::NumericVector avgrowl0(p, avgtl0);
  // arma::rowvec avgrow2l0 = avgrowl0;
  // Gammanew.row(z-1) = avgrow2l0;
  Gammanew.row(z-1).fill(mean(Gammanew.row(z-1)));

  return(Gammanew);
}

sub_dog_out refit_DOG(const arma::mat& A, const arma::mat& Omega, const arma::mat& Uom, const arma::mat& Gamma,
                      const arma::mat& Ugam, const double& rho, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                      const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D){
  // Input
  // A : matrix of dimension p times |Z|
  // Omega : matrix of dimension p times p
  // Uom: dual variable, matrix of dimension p times p
  // Gamma : matrix of dimension |Z| times p
  // Ugam : dual variable, matrix of dimension |Z| times p
  // rho : parameter ADMM
  // Atilde : rbind(A, I_|Z|x|Z|)
  // A_for_gamma : matrix of dimension |Z|x|Z|
  // A_for_B : matrix of dimension dimension (p+|Z|)x(p+|Z|)
  // C : matrix of dimension (p+ |Z|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p

  // Function : Obtain estimates of Omega^(2), Gamma_Z^(2) and D (5.3)

  // Output
  // D : matrix of dimension p times p
  // Gamma: matrix of dimension |Z| times p
  // Omega : matrix of dimension p times p

  int p = Omega.n_cols; // number of variables
  int z = A.n_cols; // size of Z

  // solve for D
  arma::mat Mtilde = arma::join_cols( Omega - Uom / rho, Gamma - Ugam/rho );
  arma::mat B = A_for_B * Mtilde;
  arma::vec BCd = arma::diagvec(B.t() * C);
  arma::mat BCdm = max(BCd, zeros(p));
  arma::mat BCdm2 = arma::diagmat(BCdm);
  arma::mat Dnew  = C_for_D * BCdm2;

  // solve for Gamma^(2)
  arma::mat Dtilde = arma::join_cols( Dnew, zeros(z, p) );
  arma::mat Gammanew = A_for_gamma * (Mtilde - Dtilde);

  // solve for Omega^(2)
  arma::mat Omeganew = A * Gammanew + Dnew;

  // Rcpp::List results=Rcpp::List::create(
  //                   Rcpp::Named("D") = Dnew,
  //                   Rcpp::Named("Gamma") = Gammanew,
  //                   Rcpp::Named("Omega") = Omeganew);
  //
  // return(results);

  sub_dog_out dogout;
  dogout.D = Dnew;
  dogout.Gamma = Gammanew;
  dogout.Omega = Omeganew;
  return dogout;

}

ADMM_block_out refit_ADMM_block_new(const arma::mat& S, const arma::mat& A, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                                    const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D, const double& rho,
                                    const arma::mat& omP, double maxite, const arma::mat& init_om, const arma::mat& init_u1,
                                    const arma::mat& init_u2, const arma::mat& init_u3, const arma::mat& init_gam,
                                    const arma::mat& init_u4, const arma::mat& init_u5){
  // Input
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |Z|
  // Atilde : rbind(A, I_|Z|x|Z|)
  // A_for_gamma : matrix of dimension |Z|x|Z|
  // A_for_B : matrix of dimension dimension (p+|Z|)x(p+|Z|)
  // C : matrix of dimension (p+ |Z|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // rho : scalar, parameter ADMM
  // omP : matrix of dimension p times p with 0 or 1 as entries: 0 (zero-elements) and 1 (non-zero elements)
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_gam : matrix of dimension |Z| times p, initialization of Gamma
  // init_u4 : matrix of dimension |Z| times p, initialization of dual variable U4 of Gamma^(1)
  // init_u5 : matrix of dimension |Z| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : ADMM update

  // Output : List

  int p = S.n_cols; // number of variables
  int nnodes = A.n_cols; // size of |Z|

  arma::mat omegaold = init_om;
  arma::mat gammaold = init_gam;

  arma::mat u1 = init_u1;
  arma::mat u2 = init_u2;
  arma::mat u3 = init_u3;
  arma::mat u4 = init_u4;
  arma::mat u5 = init_u5;

  arma::mat om1 = zeros(p, p); // eigenvalue decomposition
  arma::mat om2 = zeros(p, p); // AG+D
  arma::mat om3 = zeros(p, p); // soft- thresholding
  arma::mat gam1 = zeros(nnodes, p); // groupwise soft-thresholding
  arma::mat gam2 = zeros(nnodes, p); // AG+D
  arma::mat d = zeros(p, p); // AG+D

  sub_dog_out dogout_fit;

  // Rcpp::List dog;

  for(int iin=0; iin < maxite; ++iin){

    //Rcpp::Rcout << "iin = " << iin << std::endl;

    // Solve for Omega^(1) : Eigenvalue decomposition
    om1 = refit_omega_ed_sym(S, omegaold, u1, rho); // output is a matrix of dimension p times p

    // Solve for Omega^(3) : Soft-thresholding
    om3 = refit_omega_soft(omegaold, u3, rho, omP); // output is a matrix of dimension p times p

    // Solve for Gamma_Z^(1) : Groupwise soft-thresholding
    gam1 = refit_gamma_soft(gammaold, u4, rho); // output is a matrix of dimension |Z| times p

    // Solve for D, Omega^(2) and Gamma_Z^(2)
    // dog = refit_DOG(A, omegaold, u2, gammaold, u5, rho, Atilde, A_for_gamma, A_for_B, C, C_for_D); // output is a List
    // om2 = Rcpp::as<arma::mat>(dog["Omega"]);
    // gam2 = Rcpp::as<arma::mat>(dog["Gamma"]);
    // d = Rcpp::as<arma::mat>(dog["D"]);
    dogout_fit = refit_DOG(A, omegaold, u2, gammaold, u5, rho, Atilde, A_for_gamma, A_for_B, C, C_for_D); // output is a List
    om2 = dogout_fit.Omega;
    gam2 = dogout_fit.Gamma;
    d = dogout_fit.D;

    // Updating Omega and Gamma_Z
    omegaold = (om1 + om2 + om3) / 3;
    gammaold = (gam1 + gam2) / 2;

    // Update Dual variables
    u1 = u1 + rho * ( om1 - omegaold);
    u2 = u2 + rho * ( om2 - omegaold);
    u3 = u3 + rho * ( om3 - omegaold);
    u4 = u4 + rho * ( gam1 - gammaold);
    u5 = u5 + rho * ( gam2 - gammaold);

  }

  ADMM_block_out ADMMblockout;
  ADMMblockout.om1 = om1;
  ADMMblockout.om2 = om2;
  ADMMblockout.om3 = om3;
  ADMMblockout.gam1 = gam1;
  ADMMblockout.gam2 = gam2;
  ADMMblockout.D = d;
  ADMMblockout.omega = omegaold;
  ADMMblockout.gamma = gammaold;
  ADMMblockout.Atilde = Atilde;
  ADMMblockout.C = C;
  ADMMblockout.u1 = u1;
  ADMMblockout.u2 = u2;
  ADMMblockout.u3 = u3;
  ADMMblockout.u4 = u4;
  ADMMblockout.u5 = u5;
  ADMMblockout.omP = omP;
  return(ADMMblockout);
}
// [[Rcpp::export]]
Rcpp::List refit_LA_ADMM_export(const int& it_out, const int& it_in , const arma::mat& S, const arma::mat& A,
                                const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B,
                                const arma::mat& C, const arma::mat& C_for_D, const double& rho, const arma::mat& omP,
                                const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u2,
                                const arma::mat& init_u3, const arma::mat& init_gam, const arma::mat& init_u4,
                                const arma::mat& init_u5){
  // Input
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of ADMM algorithm
  // The remainder are the same inputs as the ones used in the taglasso_block function:
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |Z|
  // Atilde : rbind(A, I_|Z|x|Z|)
  // A_for_gamma : matrix of dimension |Z|x|Z|
  // A_for_B : matrix of dimension dimension (p+|Z|)x(p+|Z|)
  // C : matrix of dimension (p+ |Z|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // omP : matrix of dimension p times p with 0 or 1 as entries: 0 (zero-elements) and 1 (non-zero elements)
  // rho : scalar, parameter ADMM
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_gam : matrix of dimension |Z| times p, initialization of Gamma
  // init_u4 : matrix of dimension |Z| times p, initialization of dual variable U4 of Gamma^(1)
  // init_u5 : matrix of dimension |Z| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : LA-ADMM updates

  // Output : List

  // Preliminaries
  // Rcpp::List fit;

  arma::mat in_om = init_om;
  arma::mat in_gam = init_gam;

  double rhoold = rho;
  double rhonew = rho;
  ADMM_block_out ADMMblockout_fit;

  for(int iout=0; iout < it_out; ++iout){

    // fit = refit_ADMM_block_new(S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, rhonew , omP, it_in, in_om, init_u1, init_u2, init_u3,
    //                            in_gam, init_u4, init_u5);

    ADMMblockout_fit = refit_ADMM_block_new(S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, rhonew , omP, it_in, in_om, init_u1, init_u2, init_u3,
                                            in_gam, init_u4, init_u5);

    in_om  = ADMMblockout_fit.omega;
    in_gam = ADMMblockout_fit.gamma;

    // in_om  = Rcpp::as<arma::mat>(fit["omega"]);
    // in_gam = Rcpp::as<arma::mat>(fit["gamma"]);
    rhonew = 2*rhoold;
    rhoold = rhonew;
  }

  Rcpp::List results=Rcpp::List::create(
    Rcpp::Named("omega") = ADMMblockout_fit.omega,
    Rcpp::Named("gamma") = ADMMblockout_fit.gamma,
    Rcpp::Named("om1") = ADMMblockout_fit.om1,
    Rcpp::Named("om2") = ADMMblockout_fit.om2,
    Rcpp::Named("om3") = ADMMblockout_fit.om3,
    Rcpp::Named("gam1") = ADMMblockout_fit.gam1,
    Rcpp::Named("gam2") = ADMMblockout_fit.gam2,
    Rcpp::Named("D") = ADMMblockout_fit.D,
    Rcpp::Named("u1") = ADMMblockout_fit.u1,
    Rcpp::Named("u2") = ADMMblockout_fit.u2,
    Rcpp::Named("u3") = ADMMblockout_fit.u3,
    Rcpp::Named("u4") = ADMMblockout_fit.u4,
    Rcpp::Named("u5") = ADMMblockout_fit.u5,
    Rcpp::Named("omP") = omP,
    Rcpp::Named("rho") = rhonew);

  return(results);

  // LA_ADMM_out LAADMMout;
  // LAADMMout.omega = ADMMblockout_fit.omega;
  // LAADMMout.gamma = ADMMblockout_fit.gamma;
  // LAADMMout.om1 = ADMMblockout_fit.om1;
  // LAADMMout.om2 = ADMMblockout_fit.om2;
  // LAADMMout.gam1 = ADMMblockout_fit.gam1;
  // LAADMMout.gam2 = ADMMblockout_fit.gam2;
  // LAADMMout.D = ADMMblockout_fit.D;
  // LAADMMout.u1 = ADMMblockout_fit.u1;
  // LAADMMout.u2 = ADMMblockout_fit.u2;
  // LAADMMout.u3 = ADMMblockout_fit.u3;
  // LAADMMout.u4 = ADMMblockout_fit.u4;
  // LAADMMout.u5 = ADMMblockout_fit.u5;
  // LAADMMout.omP = omP;
  // LAADMMout.rho = rhonew;
  // return(LAADMMout);
}

LA_ADMM_out refit_LA_ADMM_new(const int& it_out, const int& it_in , const arma::mat& S, const arma::mat& A,
                              const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B,
                              const arma::mat& C, const arma::mat& C_for_D, const double& rho, const arma::mat& omP,
                              const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u2,
                              const arma::mat& init_u3, const arma::mat& init_gam, const arma::mat& init_u4,
                              const arma::mat& init_u5){
  // Input
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of ADMM algorithm
  // The remainder are the same inputs as the ones used in the taglasso_block function:
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |Z|
  // Atilde : rbind(A, I_|Z|x|Z|)
  // A_for_gamma : matrix of dimension |Z|x|Z|
  // A_for_B : matrix of dimension dimension (p+|Z|)x(p+|Z|)
  // C : matrix of dimension (p+ |Z|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // omP : matrix of dimension p times p with 0 or 1 as entries: 0 (zero-elements) and 1 (non-zero elements)
  // rho : scalar, parameter ADMM
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_gam : matrix of dimension |Z| times p, initialization of Gamma
  // init_u4 : matrix of dimension |Z| times p, initialization of dual variable U4 of Gamma^(1)
  // init_u5 : matrix of dimension |Z| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : LA-ADMM updates

  // Output : List

  // Preliminaries
  // Rcpp::List fit;

  arma::mat in_om = init_om;
  arma::mat in_gam = init_gam;

  double rhoold = rho;
  double rhonew = rho;
  ADMM_block_out ADMMblockout_fit;

  for(int iout=0; iout < it_out; ++iout){

    // fit = refit_ADMM_block_new(S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, rhonew , omP, it_in, in_om, init_u1, init_u2, init_u3,
    //                            in_gam, init_u4, init_u5);

    ADMMblockout_fit = refit_ADMM_block_new(S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, rhonew , omP, it_in, in_om, init_u1, init_u2, init_u3,
                                            in_gam, init_u4, init_u5);

    in_om  = ADMMblockout_fit.omega;
    in_gam = ADMMblockout_fit.gamma;

    // in_om  = Rcpp::as<arma::mat>(fit["omega"]);
    // in_gam = Rcpp::as<arma::mat>(fit["gamma"]);
    rhonew = 2*rhoold;
    rhoold = rhonew;
  }


  LA_ADMM_out LAADMMout;
  LAADMMout.omega = ADMMblockout_fit.omega;
  LAADMMout.gamma = ADMMblockout_fit.gamma;
  LAADMMout.om1 = ADMMblockout_fit.om1;
  LAADMMout.om2 = ADMMblockout_fit.om2;
  LAADMMout.gam1 = ADMMblockout_fit.gam1;
  LAADMMout.gam2 = ADMMblockout_fit.gam2;
  LAADMMout.D = ADMMblockout_fit.D;
  LAADMMout.u1 = ADMMblockout_fit.u1;
  LAADMMout.u2 = ADMMblockout_fit.u2;
  LAADMMout.u3 = ADMMblockout_fit.u3;
  LAADMMout.u4 = ADMMblockout_fit.u4;
  LAADMMout.u5 = ADMMblockout_fit.u5;
  LAADMMout.omP = omP;
  LAADMMout.rho = rhonew;
  return(LAADMMout);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////// taglasso functions ///////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

prelim_A_out preliminaries_for_taglasso_A(const arma::mat& A){
  // Input
  // A : matrix of dimension p times |T|

  int p = A.n_rows; // number of variables
  int t = A.n_cols; // size of the tree

  // Preliminaries for DOG functions
  arma::mat Atilde = arma::join_cols( A, arma::eye(t, t) );
  arma::mat A_for_gamma = inv( A.t() * A + arma::eye(t, t) ) * arma::join_rows( A.t(), arma::eye(t, t) );
  arma::mat A_for_B = ( arma::eye(p + t, p + t) - Atilde * inv(Atilde.t() * Atilde) * Atilde.t() );
  arma::mat C = (arma::join_rows(arma::eye(p, p), zeros(p, t))).t() - Atilde * inv( Atilde.t() * Atilde ) * A.t();
  arma::mat C_for_D = inv( arma::diagmat( arma::diagvec(C.t() * C) ) ) ;

  // // Preliminaries for Omega
  // Rcpp::List results=Rcpp::List::create(
  //   Rcpp::Named("Atilde") = Atilde,
  //   Rcpp::Named("A_for_gamma") = A_for_gamma,
  //   Rcpp::Named("A_for_B") = A_for_B,
  //   Rcpp::Named("C") = C,
  //   Rcpp::Named("C_for_D") = C_for_D);
  // return(results);

  prelim_A_out prelimAout;
  prelimAout.Atilde = Atilde;
  prelimAout.A_for_gamma = A_for_gamma;
  prelimAout.A_for_B = A_for_B;
  prelimAout.C = C;
  prelimAout.C_for_D = C_for_D;

  return(prelimAout);
}

double softelem(const double& a, const double& lambda){
  // Input
  // a : scalar
  // lambda : scalar, tuning parameter

  // Function : Elementwise soft-thresholding (needed in function solve_omega_soft)

  // Output
  // sg : scalar after elementwise soft-thresholding

  double out = ((a > 0) - (a < 0)) * std::max(0.0, std::abs(a) - lambda);
  return(out);
}

arma::mat solve_omega_soft(const arma::mat& Omega, const arma::mat& U, const double& rho, const double& lambda,
                           const bool& pendiag){
  // Input
  // Omega : matrix of dimension p times p
  // U : dual variable, matrix of dimension p times p
  // rho : parameter ADMM
  // lambda : regularization parameter for elementwise soft thresholding
  // pendiag : logical, penalize diagonal or not

  // Function : Obtain estimate of Omega^(1) (4.1.1)

  // Output
  // Omeganew : matrix of dimension p times p

  int p = Omega.n_cols;
  arma::mat Omeganew = zeros(p, p);
  arma::mat soft_input = Omega - U/rho;

  arma::vec diago = zeros(p);

  for(int ir=0; ir < p; ++ir ){
    for(int ic=0; ic < p; ++ ic){
      Omeganew(ir, ic) = softelem(soft_input(ir, ic), lambda/rho);
    }
  }

  if(!pendiag){
    Omeganew.diag() = arma::diagvec(Omega - U/rho);
  }

  return(Omeganew);
}

arma::vec softgroup(const arma::vec& u, const double& lambda){
  // Input
  // u : vector (of dimension p)
  // lambda : scalar, tuning parameter

  // Function : Groupwise soft-thresholding (needed in function solve_gamma_soft)

  // Output
  // sg : vector after groupwise soft-thresholding

  arma::vec sg = std::max(0.0, 1 - lambda/arma::norm(u,2))*u;
  return(sg);
}

arma::mat solve_gamma_soft(const arma::mat& Gamma, const arma::mat& U, const double& rho, const double& lambda){
  // Input
  // Gamma : matrix of dimension |T| times p
  // U : dual variable, matrix of dimension |T| times p
  // rho : parameter ADMM
  // lambda : regularization parameter for groupwise soft-thresholding

  // Function : Obtain estimate of Gamma^(1) (4.1.2)

  // Output
  // Gammanew : matrix of dimension |T| times p

  int t = Gamma.n_rows;
  int p = Gamma.n_cols;
  arma::mat Gammanew = zeros(t, p);
  arma::mat soft_input = Gamma - U/rho;
  arma::rowvec outt = soft_input.row(0);

  if(lambda==0){
    Gammanew = Gamma - U/rho;  // no shrinkage
    // double avgtl0 = mean(Gammanew.row(t-1)); // last row corresponds to the root
    // Rcpp::NumericVector avgrowl0(p, avgtl0);
    // arma::rowvec avgrow2l0 = avgrowl0;
    // Gammanew.row(t-1) = avgrow2l0;
    Gammanew.row(t-1).fill(mean(Gammanew.row(t-1)));
  }else{

    for(int ir=0; ir < t; ++ir){
      outt = soft_input.row(ir);
      if(ir == (t-1)){
        // Gammanew.row(ir) = outt;
        // double avgt = mean(outt);
        // Rcpp::NumericVector avgrow(p, avgt);
        // arma::rowvec avgrow2 = avgrow;
        // Gammanew.row(ir) = avgrow2;
        Gammanew.row(ir).fill(mean(outt));
        //Rcpp::Rcout << "I'm here = " << outt << std::endl;
      }else{
        Gammanew.row(ir) = softgroup(outt.t(), lambda/rho).t();
      }
    }

  }

  // Rcpp::Rcout << "Gamma1 = " << Gammanew << std::endl;

  return(Gammanew);
}

sub_dog_out solve_DOG(const arma::mat& A, const arma::mat& Omega, const arma::mat& Uom, const arma::mat& Gamma,
                      const arma::mat& Ugam, const double& rho, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                      const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D){
  // Input
  // A : matrix of dimension p times |T|
  // Omega : matrix of dimension p times p
  // Uom: dual variable, matrix of dimension p times p
  // Gamma : matrix of dimension |T| times p
  // Ugam : dual variable, matrix of dimension |T| times p
  // rho : parameter ADMM
  // Atilde : rbind(A, I_|T|x|T|)
  // A_for_gamma : matrix of dimension |T|x|T|
  // A_for_B : matrix of dimension dimension (p+|T|)x(p+|T|)
  // C : matrix of dimension (p+ |T|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p

  // Function : Obtain estimates of Omega^(2), Gamma^(2) and D (4.1.3)

  // Output
  // D : matrix of dimension p times p
  // Gamma: matrix of dimension |T| times p
  // Omega : matrix of dimension p times p

  int p = Omega.n_cols; // number of variables
  int t = A.n_cols; // size of the tree

  // solve for D
  arma::mat Mtilde = arma::join_cols( Omega - Uom / rho, Gamma - Ugam/rho );
  arma::mat B = A_for_B * Mtilde;
  // Rcpp::NumericVector BCd = Rcpp::wrap(arma::diagvec(B.t() * C));
  // arma::mat BCdm = Rcpp::pmax(BCd, 0);
  arma::vec BCd = arma::diagvec(B.t() * C);
  arma::mat BCdm = max(BCd, zeros(p));
  arma::mat BCdm2 = arma::diagmat(BCdm);
  arma::mat Dnew  = C_for_D * BCdm2;

  // solve for Gamma^(2)
  arma::mat Dtilde = arma::join_cols( Dnew, zeros(t, p) );
  arma::mat Gammanew = A_for_gamma * (Mtilde - Dtilde);

  // solve for Omega^(2)
  arma::mat Omeganew = A * Gammanew + Dnew;

  sub_dog_out dogout;
  dogout.D = Dnew;
  dogout.Gamma = Gammanew;
  dogout.Omega = Omeganew;
  return(dogout);

  // Rcpp::List results=Rcpp::List::create(
  //   Rcpp::Named("D") = Dnew,
  //   Rcpp::Named("Gamma") = Gammanew,
  //   Rcpp::Named("Omega") = Omeganew);
  //
  // return(results);
}

ADMM_block_out ADMM_taglasso_block(const arma::mat& S, const arma::mat& A, const arma::mat& Atilde, const arma::mat& A_for_gamma,
                                  const arma::mat& A_for_B, const arma::mat& C, const arma::mat& C_for_D, const double& lambda1,
                                  const double& lambda2, const double& rho, const bool& pendiag, const double& maxite,
                                  const arma::mat& init_om, const arma::mat& init_u1, const arma::mat& init_u3,
                                  const arma::mat& init_u4, const arma::mat& init_gam, const arma::mat& init_u2,
                                  const arma::mat& init_u5){
  // Input
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |T|
  // Atilde : rbind(A, I_|T|x|T|)
  // A_for_gamma : matrix of dimension |T|x|T|
  // A_for_B : matrix of dimension dimension (p+|T|)x(p+|T|)
  // C : matrix of dimension (p+ |T|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // lambda1 : scalar, regularization parameter lambda1||Gamma^(1)_r||_{2,1}
  // lambda2: scalar, regularization parameter lambda2||Omega||_1
  // rho : scalar, parameter ADMM
  // pendiag : logical, penalize diagonal or not when solving for Omega^(1)
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_u4 : matrix of dimension |T| times p, initialization of dual variable U4 of Gamma^(1)
  // init_gam : matrix of dimension |T| times p, initialization of Gamma
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u5 : matrix of dimension |T| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : ADMM update

  // Output : List


  int p = S.n_cols; // number of variables
  int nnodes = A.n_cols; // size of the tree |T|

  arma::mat omegaold = init_om;
  arma::mat gammaold = init_gam;

  arma::mat u1 = init_u1;
  arma::mat u2 = init_u2;
  arma::mat u3 = init_u3;
  arma::mat u4 = init_u4;
  arma::mat u5 = init_u5;

  arma::mat om1 = zeros(p, p); // eigenvalue decomposition
  arma::mat om2 = zeros(p, p); // AG+D
  arma::mat om3 = zeros(p, p); // soft thresholding
  arma::mat gam1 = zeros(nnodes, p); // groupwise soft thresholding
  arma::mat gam2 = zeros(nnodes, p); // AG+D
  arma::mat d = zeros(p, p); // AG+D

  // Rcpp::List dog;
  sub_dog_out dogout_fit;
  for(int iin=0; iin < maxite; ++iin){

    // Solve for Omega^(1) : Eigenvalue decomposition
    om1 = refit_omega_ed_sym(S, omegaold, u1, rho); // output is a matrix of dimension p times p

    // Solve for Omega^(3): Soft-thresholding
    om3 = solve_omega_soft(omegaold, u3, rho, lambda2, pendiag); // output is a matrix of dimension p times p

    // Solve for Gamma^(1) : Groupwise soft-thresholding
    gam1 = solve_gamma_soft(gammaold, u4, rho, lambda1); // output is a matrix of dimension |T| times p

    // Solve for D, Omega^(2) and Gamma^(2)
    // dog = solve_DOG(A, omegaold, u2, gammaold, u5, rho, Atilde, A_for_gamma, A_for_B, C, C_for_D); // output is a List
    // om2 = Rcpp::as<arma::mat>(dog["Omega"]);
    // gam2 = Rcpp::as<arma::mat>(dog["Gamma"]);
    // d = Rcpp::as<arma::mat>(dog["D"]);
    dogout_fit = solve_DOG(A, omegaold, u2, gammaold, u5, rho, Atilde, A_for_gamma, A_for_B, C, C_for_D); // output is a List
    om2 = dogout_fit.Omega;
    gam2 = dogout_fit.Gamma;
    d = dogout_fit.D;

    // Updating Omega, Gamma and Theta
    omegaold = (om1 + om2 + om3) / 3;
    gammaold = (gam1 + gam2) / 2;

    // Update Dual variables
    u1 = u1 + rho * ( om1 - omegaold);
    u2 = u2 + rho * ( om2 - omegaold);
    u3 = u3 + rho * ( om3 - omegaold);
    u4 = u4 + rho * ( gam1 - gammaold);
    u5 = u5 + rho * ( gam2 - gammaold);

  }

  ADMM_block_out ADMMblockout;
  ADMMblockout.om1 = om1;
  ADMMblockout.om2 = om2;
  ADMMblockout.om3 = om3;
  ADMMblockout.gam1 = gam1;
  ADMMblockout.gam2 = gam2;
  ADMMblockout.D = d;
  ADMMblockout.omega = omegaold;
  ADMMblockout.gamma = gammaold;
  ADMMblockout.Atilde = Atilde;
  ADMMblockout.C = C;
  ADMMblockout.u1 = u1;
  ADMMblockout.u2 = u2;
  ADMMblockout.u3 = u3;
  ADMMblockout.u4 = u4;
  ADMMblockout.u5 = u5;
  return(ADMMblockout);

  // Rcpp::List results = Rcpp::List::create(
  //   Rcpp::Named("om1") = om1,
  //   Rcpp::Named("om2") = om2,
  //   Rcpp::Named("om3") = om3,
  //   Rcpp::Named("gam1") = gam1,
  //   Rcpp::Named("gam2") = gam2,
  //   Rcpp::Named("D") = d,
  //   Rcpp::Named("omega") = omegaold,
  //   Rcpp::Named("gamma") = gammaold,
  //   Rcpp::Named("Atilde") = Atilde,
  //   Rcpp::Named("C") = C,
  //   Rcpp::Named("u1") = u1,
  //   Rcpp::Named("u2") = u2,
  //   Rcpp::Named("u3") = u3,
  //   Rcpp::Named("u4") = u4,
  //   Rcpp::Named("u5")=u5);
  //
  // return(results);
}

LA_ADMM_out LA_ADMM_taglasso(const int& it_out, const int& it_in , const arma::mat& S, const arma::mat& A,
                            const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B,
                            const arma::mat& C, const arma::mat& C_for_D, const double& lambda1,
                            const double& lambda2, const double& rho, const bool& pendiag, const arma::mat& init_om,
                            const arma::mat& init_u1, const arma::mat& init_u2, const arma::mat& init_u3,
                            const arma::mat& init_gam, const arma::mat& init_u4, const arma::mat& init_u5){
  // Input
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of ADMM algorithm
  // The remainder are the same inputs as the ones used in the ADMM_taglasso_block function:
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |T|
  // Atilde : rbind(A, I_|T|x|T|)
  // A_for_gamma : matrix of dimension |T|x|T|
  // A_for_B : matrix of dimension dimension (p+|T|)x(p+|T|)
  // C : matrix of dimension (p+ |T|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // lambda1 : scalar, regularization parameter  + lambda1||Gamma^(1)_r||_{2,1}
  // lambda2 : scalar, regularization parameter lambda2||Omega^(1)||_1
  // rho : scalar, parameter ADMM
  // pendiag : logical, penalize diagonal or not when solving for Omega^(1)
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_u4 : matrix of dimension |T| times p, initialization of dual variable U4 of Gamma^(1)
  // init_gam : matrix of dimension |T| times p, initialization of Gamma
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u5 : matrix of dimension |T| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : LA-ADMM updates

  // Output : List

  // Preliminaries
  // Rcpp::List fit;

  arma::mat in_om = init_om;
  arma::mat in_gam = init_gam;

  double rhoold = rho;
  double rhonew = rho;

  ADMM_block_out ADMMout;

  for(int iout=0; iout < it_out; ++iout){

    ADMMout = ADMM_taglasso_block(S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, lambda1, lambda2, rhonew, pendiag, it_in,
                                 in_om, init_u1, init_u3, init_u4, in_gam, init_u2, init_u5);
    in_om  = ADMMout.omega;
    in_gam = ADMMout.gamma;
    // in_om  = Rcpp::as<arma::mat>(fit["omega"]);
    // in_gam = Rcpp::as<arma::mat>(fit["gamma"]);
    rhonew = 2*rhoold;
    rhoold = rhonew;
  }

  LA_ADMM_out LAADMMout;
  LAADMMout.omega = ADMMout.omega;
  LAADMMout.gamma = ADMMout.gamma;
  LAADMMout.om1 = ADMMout.om1;
  LAADMMout.om2 = ADMMout.om2;
  LAADMMout.om3 = ADMMout.om3;
  LAADMMout.gam1 = ADMMout.gam1;
  LAADMMout.gam2 = ADMMout.gam2;
  LAADMMout.D = ADMMout.D;
  LAADMMout.u1 = ADMMout.u1;
  LAADMMout.u2 = ADMMout.u2;
  LAADMMout.u3 = ADMMout.u3;
  LAADMMout.u4 = ADMMout.u4;
  LAADMMout.u5 = ADMMout.u5;
  LAADMMout.rho = rhonew;
  return(LAADMMout);
  // Rcpp::List results=Rcpp::List::create(
  //   Rcpp::Named("omega") = fit["omega"],
  //   Rcpp::Named("gamma") = fit["gamma"],
  //   Rcpp::Named("om1") = fit["om1"],
  //   Rcpp::Named("om2") = fit["om2"],
  //   Rcpp::Named("om3") = fit["om3"],
  //   Rcpp::Named("gam1") = fit["gam1"],
  //   Rcpp::Named("gam2") = fit["gam2"],
  //   Rcpp::Named("D") = fit["D"],
  //   Rcpp::Named("u1") = fit["u1"],
  //   Rcpp::Named("u2") = fit["u2"],
  //   Rcpp::Named("u3") = fit["u3"],
  //   Rcpp::Named("u4") = fit["u4"],
  //   Rcpp::Named("u5") = fit["u5"],
  //   Rcpp::Named("rho") = rhonew);
  // return(results);
}

// [[Rcpp::export]]
Rcpp::List LA_ADMM_taglasso_export(const int& it_out, const int& it_in , const arma::mat& S, const arma::mat& A,
                                  const arma::mat& Atilde, const arma::mat& A_for_gamma, const arma::mat& A_for_B,
                                  const arma::mat& C, const arma::mat& C_for_D, const double& lambda1,
                                  const double& lambda2, const double& rho, const bool& pendiag, const arma::mat& init_om,
                                  const arma::mat& init_u1, const arma::mat& init_u2, const arma::mat& init_u3,
                                  const arma::mat& init_gam, const arma::mat& init_u4, const arma::mat& init_u5){
  // Input
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of ADMM algorithm
  // The remainder are the same inputs as the ones used in the ADMM_taglasso_block function:
  // S : sample covariance matrix of dimension p times p
  // A : matrix of dimension p times |T|
  // Atilde : rbind(A, I_|T|x|T|)
  // A_for_gamma : matrix of dimension |T|x|T|
  // A_for_B : matrix of dimension dimension (p+|T|)x(p+|T|)
  // C : matrix of dimension (p+ |T|) times p (requires Atilde)
  // C_for_D : matrix of dimension p times p
  // lambda1 : scalar, regularization parameter  + lambda1||Gamma^(1)_r||_{2,1}
  // lambda2 : scalar, regularization parameter lambda2||Omega^(1)||_1
  // rho : scalar, parameter ADMM
  // pendiag : logical, penalize diagonal or not when solving for Omega^(1)
  // maxite : scalar, maximum number of iterations
  // init_om : matrix of dimension p times p, initialization of Omega
  // init_u1 : matrix of dimension p times p, initialization of dual variable U1 of Omega^(1)
  // init_u3 : matrix of dimension p times p, initialization of dual variable U3 of Omega^(3)
  // init_u4 : matrix of dimension |T| times p, initialization of dual variable U4 of Gamma^(1)
  // init_gam : matrix of dimension |T| times p, initialization of Gamma
  // init_u2 : matrix of dimension p times p, initialization of dual variable U2 of Omega^(2)
  // init_u5 : matrix of dimension |T| times p, initialization of dual variable U5 of Gamma^(2)

  // Function : LA-ADMM updates

  // Output : List

  // Preliminaries
  // Rcpp::List fit;

  arma::mat in_om = init_om;
  arma::mat in_gam = init_gam;

  double rhoold = rho;
  double rhonew = rho;

  ADMM_block_out ADMMout;

  for(int iout=0; iout < it_out; ++iout){

    ADMMout = ADMM_taglasso_block(S, A, Atilde, A_for_gamma, A_for_B, C, C_for_D, lambda1, lambda2, rhonew, pendiag, it_in,
                                 in_om, init_u1, init_u3, init_u4, in_gam, init_u2, init_u5);
    in_om  = ADMMout.omega;
    in_gam = ADMMout.gamma;
    // in_om  = Rcpp::as<arma::mat>(fit["omega"]);
    // in_gam = Rcpp::as<arma::mat>(fit["gamma"]);
    rhonew = 2*rhoold;
    rhoold = rhonew;
  }

  Rcpp::List results=Rcpp::List::create(
    Rcpp::Named("omega") = ADMMout.omega,
    Rcpp::Named("gamma") = ADMMout.gamma,
    Rcpp::Named("om1") = ADMMout.om1,
    Rcpp::Named("om2") = ADMMout.om2,
    Rcpp::Named("om3") = ADMMout.om3,
    Rcpp::Named("gam1") = ADMMout.gam1,
    Rcpp::Named("gam2") = ADMMout.gam2,
    Rcpp::Named("D") = ADMMout.D,
    Rcpp::Named("u1") = ADMMout.u1,
    Rcpp::Named("u2") = ADMMout.u2,
    Rcpp::Named("u3") = ADMMout.u3,
    Rcpp::Named("u4") = ADMMout.u4,
    Rcpp::Named("u5") = ADMMout.u5,
    Rcpp::Named("rho") = rhonew);
  return(results);
}

arma::mat determine_sparsity_pattern(const arma::mat& Omega){
  // Input
  // Gamma : matrix of dimension pxp
  // A : matrix of dimension px|T|

  // Function to determine sparsity pattern in Omega (1 for non-zero, 0 for zero)

  // Output
  // omP : matrix of dimension pxp
  int p = Omega.n_cols;
  int p2 = p*p;
  arma::mat omP = zeros(p, p);
  for(int it=0; it < p2 ; ++ it){ // loop over the rows of Gamma
    if(Omega(it)!=0){
      omP(it) = 1;
    }else{
      omP(it) = 0;
    }
  }
  return(omP);

}

Areduced_out determine_A_reduced(const arma::mat& Gamma, const arma::mat& A){
  // Input
  // Gamma : matrix of dimension |T|xp
  // A : matrix of dimension px|T|

  // Function to determine A_Z after the ADMM CLIME Algorithm

  // Output
  // AZ : matrix of dimension px|Z|

  int nnodes = A.n_cols;
  arma::mat AZ = A.col(nnodes-1); // always take the last column which corresponds to the root
  arma::rowvec newrow = Gamma.row(0);
  for(int it=(nnodes-2); it > -1 ; -- it){ // loop over the rows of Gamma

    newrow = Gamma.row(it);
    double rowsum = sum(abs(newrow));

    if(rowsum!=0){
      AZ = join_rows(A.col(it), AZ);
    }
  }
  int znodes = AZ.n_cols;

  Areduced_out Aout;
  Aout.AZ = AZ;
  Aout.znodes = znodes;
  return(Aout);

  // Rcpp::List results=Rcpp::List::create(
  //   Rcpp::Named("AZ") = AZ,
  //   Rcpp::Named("znodes") = znodes);
  //
  // return(results);
}

double objvalue(const arma::mat& omega, const arma::mat& Sout){
  // Input
  // omega :  matrix of dimension p times p
  // Sout : matrix of dimension p times p
  double out = -log(det(omega)) + accu(arma::diagvec(Sout * omega));

  return(out);

}

prelim_A_out preliminaries_for_refit(const arma::mat& A){
  // Input
  // A : matrix of dimension p times |Z|

  int p = A.n_rows; // number of variables
  int z = A.n_cols; // size of |Z|

  // Preliminaries for DOG functions
  arma::mat Atilde = arma::join_cols( A, eye(z, z) );
  arma::mat A_for_gamma = inv( A.t() * A + arma::eye(z, z) ) * arma::join_rows( A.t(), arma::eye(z, z) );
  arma::mat A_for_B = ( arma::eye(p + z, p + z) - Atilde * inv(Atilde.t() * Atilde) * Atilde.t() );
  arma::mat C = (arma::join_rows(arma::eye(p, p), zeros(p, z))).t() - Atilde * inv( Atilde.t() * Atilde ) * A.t();
  arma::mat C_for_D = inv( arma::diagmat( arma::diagvec(C.t() * C) ) ) ;

  prelim_A_out prelimAout;
  prelimAout.Atilde = Atilde;
  prelimAout.A_for_gamma = A_for_gamma;
  prelimAout.A_for_B = A_for_B;
  prelimAout.C = C;
  prelimAout.C_for_D = C_for_D;
  return(prelimAout);

  // Rcpp::List results=Rcpp::List::create(
  //   Rcpp::Named("Atilde") = Atilde,
  //   Rcpp::Named("A_for_gamma") = A_for_gamma,
  //   Rcpp::Named("A_for_B") = A_for_B,
  //   Rcpp::Named("C") = C,
  //   Rcpp::Named("C_for_D") = C_for_D);
  //
  // return(results);
}

// [[Rcpp::export]]
int determine_dfs(arma::mat A, const double& tol = 1e-06){
  arma::mat Aabs = abs(A);
  A.elem( find(Aabs < tol) ).zeros(); // make small entries zero
  arma::mat A_unique = unique(A); // get unique elements (includes zero still)
  arma::vec A_unique_nz = nonzeros(A_unique); // get unique non-zeros
  int df = A_unique_nz.size();
  return(df);

}

// [[Rcpp::export]]
double matrix_unique_rows(const arma::mat& A) {

  arma::uvec ulmt = arma::zeros<arma::uvec>(A.n_rows);

  for (arma::uword i = 0; i < A.n_rows; i++) {
    for (arma::uword j = i + 1; j < A.n_rows; j++) {
      if (arma::approx_equal(A.row(i), A.row(j), "absdiff", 0.00000001)) { ulmt(j) = 1; break; }
    }
  }

  arma::mat AZ = A.rows(find(ulmt == 0));
  double K = AZ.n_rows;
  return(K);

}


// [[Rcpp::export]]
Rcpp::List ADMM_taglasso_cv_parallel(const arma::vec& fold_vector, const arma::vec& lambda1_vector, const arma::vec& lambda1_index_vector,
                                    const arma::vec& lambda2_vector, const arma::vec& lambda2_index_vector,
                                    const arma::cube& Sin, const arma::cube& Sout,
                                    const int& fold, const arma::mat& A,
                                    const arma::vec& lambda1, const arma::vec& lambda2, const double& rho, const bool& pendiag,
                                    const int& it_out, const int& it_in, const int& it_out_refit, const int& it_in_refit,
                                    const bool& do_parallel, const int& nc){
  // Input
  // fold : fold- cross-validation
  // Sin : list (of length fold) of sample covariance matrices on training data
  // Sout : list (of length fold) of sample covariance matrices on test data
  // A : matrix of dimension p times |T|
  // lambda1 : vector of regularization parameters lambda1||Gamma^(1)_r||_{2,1}
  // lambda2: vector of regularization parameters lambda2||Omega||_1
  // rho : scalar, parameter ADMM
  // pendiag : logical, penalize diagonal or not when solving for Omega^(1)
  // it_out : scalar, T_stages of LA-ADMM algorithm
  // it_in : scalar, maximum number of iterations of ADMM algorithm

  // Function : Cross-validation for selecting the regularization parameters alpha and lambda
  int p = Sin.n_cols;
  int nnodes = A.n_cols;
  int l1length = lambda1.size();
  int l2length = lambda2.size();
  int size_fold_lambdas = fold*l1length*l2length;

  // Preliminaries for A and S when A is of dimension px|T|
  prelim_A_out prelimoutA = preliminaries_for_taglasso_A(A); // preliminaries for A when A is of dimension px|T|

  arma::cube obj = zeros(fold, l1length, l2length);
  arma::cube l1check = zeros(fold, l1length, l2length);
  arma::cube l2check = zeros(fold, l1length, l2length);
  arma::cube dfs = zeros(fold, l1length, l2length);
  arma::cube sparsity = zeros(fold, l1length, l2length);
  arma::cube clusters = zeros(fold, l1length, l2length);
  int iall;

  LA_ADMM_out LAADMMout_fit;
  Areduced_out fitAZ;
  prelim_A_out AZprelims;
  LA_ADMM_out refit;
  arma::mat omP;

#ifdef _OPENMP
  omp_set_num_threads(nc);
#endif

  if(do_parallel){
#pragma omp parallel for schedule(static) default(none) private(iall, LAADMMout_fit, omP, fitAZ, AZprelims, refit) shared(size_fold_lambdas, it_out, it_in, Sin, fold_vector, A, prelimoutA, lambda1_vector, lambda2_vector, rho, pendiag, p, nnodes, it_out_refit, it_in_refit, l1check, l2check, obj, dfs, sparsity, clusters, lambda1_index_vector, lambda2_index_vector, Sout)

    for(iall=0; iall < size_fold_lambdas; ++ iall){ // cross-validation loo
      LAADMMout_fit = LA_ADMM_taglasso(it_out, it_in, Sin.slice(fold_vector[iall]-1), A,
                                      prelimoutA.Atilde, prelimoutA.A_for_gamma, prelimoutA.A_for_B, prelimoutA.C, prelimoutA.C_for_D,
                                      lambda1_vector[iall], lambda2_vector[iall], rho, pendiag,
                                      zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p), zeros(nnodes, p), zeros(nnodes, p), zeros(nnodes, p));
      omP = determine_sparsity_pattern(LAADMMout_fit.om3);
      fitAZ = determine_A_reduced(LAADMMout_fit.gam1, A);
      AZprelims = preliminaries_for_refit(fitAZ.AZ); // preliminaries for AZ when AZ is of dimension px|Z| AZ[counter]
      refit = refit_LA_ADMM_new(it_out_refit, it_in_refit, Sin.slice(fold_vector[iall]-1),
                                fitAZ.AZ,  AZprelims.Atilde, AZprelims.A_for_gamma, AZprelims.A_for_B, AZprelims.C, AZprelims.C_for_D,
                                rho, omP,
                                zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p), zeros(fitAZ.znodes, p), zeros(fitAZ.znodes, p), zeros(fitAZ.znodes, p));

      l1check.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = lambda1_vector[iall];
      l2check.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = lambda2_vector[iall];
      obj.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = objvalue(refit.om1, Sout.slice(fold_vector[iall]-1));
      dfs.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = determine_dfs(A*LAADMMout_fit.gam1);
      sparsity.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = p*p - sum(sum(omP));
      clusters.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = matrix_unique_rows(fitAZ.AZ);
    }
  }else{
    for(iall=0; iall < size_fold_lambdas; ++ iall){ // cross-validation loo
      LAADMMout_fit = LA_ADMM_taglasso(it_out, it_in, Sin.slice(fold_vector[iall]-1), A,
                                      prelimoutA.Atilde, prelimoutA.A_for_gamma, prelimoutA.A_for_B, prelimoutA.C, prelimoutA.C_for_D,
                                      lambda1_vector[iall], lambda2_vector[iall], rho, pendiag,
                                      zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p), zeros(nnodes, p), zeros(nnodes, p), zeros(nnodes, p));
      omP = determine_sparsity_pattern(LAADMMout_fit.om3);
      fitAZ = determine_A_reduced(LAADMMout_fit.gam1, A);
      AZprelims = preliminaries_for_refit(fitAZ.AZ); // preliminaries for AZ when AZ is of dimension px|Z| AZ[counter]
      refit = refit_LA_ADMM_new(it_out_refit, it_in_refit, Sin.slice(fold_vector[iall]-1),
                                fitAZ.AZ,  AZprelims.Atilde, AZprelims.A_for_gamma, AZprelims.A_for_B, AZprelims.C, AZprelims.C_for_D,
                                rho, omP,
                                zeros(p, p), zeros(p, p), zeros(p, p), zeros(p, p), zeros(fitAZ.znodes, p), zeros(fitAZ.znodes, p), zeros(fitAZ.znodes, p));

      l1check.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = lambda1_vector[iall];
      l2check.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = lambda2_vector[iall];
      obj.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = objvalue(refit.om1, Sout.slice(fold_vector[iall]-1));
      dfs.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = determine_dfs(A*LAADMMout_fit.gam1);
      sparsity.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = p*p - sum(sum(omP));
      clusters.slice(lambda2_index_vector[iall]-1)(fold_vector[iall]-1, lambda1_index_vector[iall]-1) = matrix_unique_rows(fitAZ.AZ);
    }
  }


  Rcpp::List results=Rcpp::List::create(
    Rcpp::Named("l1check") = l1check,
    Rcpp::Named("l2check") = l2check,
    Rcpp::Named("cvobj") = obj,
    Rcpp::Named("dfs") = dfs,
    Rcpp::Named("sparsity") = sparsity,
    Rcpp::Named("clusters") = clusters,
    Rcpp::Named("Atilde") = prelimoutA.Atilde,
    Rcpp::Named("A_for_gamma") = prelimoutA.A_for_gamma,
    Rcpp::Named("A_for_B") = prelimoutA.A_for_B,
    Rcpp::Named("C") = prelimoutA.C,
    Rcpp::Named("C_for_D") = prelimoutA.C_for_D);
  return(results);
}

