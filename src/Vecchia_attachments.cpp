/*
 // MIT License
 // 
 //   Copyright (c) 2018 Joseph Guinness
 // 
 //   Permission is hereby granted, free of charge, to any person obtaining a copy
 //   of this software and associated documentation files (the "Software"), to deal
 //   in the Software without restriction, including without limitation the rights
 //   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 //   copies of the Software, and to permit persons to whom the Software is
 //   furnished to do so, subject to the following conditions:
 // 
 //     The above copyright notice and this permission notice shall be included in all
 //     copies or substantial portions of the Software.
 // 
 //   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 //     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 //     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 //     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 //     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 //     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 //     SOFTWARE.
 */

#include <RcppArmadillo.h>
#include <armadillo>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


arma::mat backward_solve_mat( arma::mat cholmat, arma::mat b ){
  
  int n = cholmat.n_rows;
  int p = b.n_cols;
  arma::mat x(n,p);
  for(int k=0; k<p; k++){ x(n-1,k) = b(n-1,k)/cholmat(n-1,n-1); }
  
  for(int i=n-2; i>=0; i--){
    for(int k=0; k<p; k++){
      double dd = 0.0;
      for(int j=n-1; j>i; j--){
        dd += cholmat(j,i)*x(j,k);
      }
      x(i,k) = (b(i,k)-dd)/cholmat(i,i);
    }
  }    
  return x;
} 
arma::mat forward_solve_mat( arma::mat cholmat, arma::mat b ){
  
  int n = cholmat.n_rows;
  int p = b.n_cols;
  arma::mat x(n,p);
  for(int k=0; k<p; k++){ x(0,k) = b(0,k)/cholmat(0,0); }
  
  for(int i=1; i<n; i++){
    for(int k=0; k<p; k++){
      double dd = 0.0;
      for(int j=0; j<i; j++){
        dd += cholmat(i,j)*x(j,k);
      }
      x(i,k) = (b(i,k)-dd)/cholmat(i,i);
    }
  }    
  return x;
} 
arma::vec forward_solve( arma::mat cholmat, arma::vec b ){
  
  int n = cholmat.n_rows;
  arma::vec x(n);
  x(0) = b(0)/cholmat(0,0);
  
  for(int i=1; i<n; i++){
    double dd = 0.0;
    for(int j=0; j<i; j++){
      dd += cholmat(i,j)*x(j);
    }
    x(i) = (b(i)-dd)/cholmat(i,i);
  }    
  return x;
  
} 
arma::mat mychol( arma::mat A ){
  
  arma::uword n = A.n_rows;
  arma::mat L(n,n);
  //bool pd = true;
  
  // upper-left entry
  if( A(0,0) < 0 ){
    //pd = false;
    L(0,0) = 1.0;
  } else {
    L(0,0) = std::sqrt(A(0,0));
  }
  if( n > 1 ){
    // second row
    L(1,0) = A(1,0)/L(0,0);
    double f = A(1,1) - L(1,0)*L(1,0);
    if( f < 0 ){
      //pd = false;
      L(1,1) = 1.0;
    } else {
      L(1,1) = std::sqrt( f );
    }
    // rest of the rows
    if( n > 2 ){
      for(uword i=2; i<n; i++){
        // leftmost entry in row i
        L(i,0) = A(i,0)/L(0,0);
        // middle entries in row i 
        for(uword j=1; j<i; j++){
          double d = A(i,j);
          for(uword k=0; k<j; k++){
            d -= L(i,k)*L(j,k);
          }
          L(i,j) = d/L(j,j);
        }
        // diagonal entry in row i
        double e = A(i,i);
        for(uword k=0; k<i; k++){
          e -= L(i,k)*L(i,k);
        }
        if( e < 0 ){
          //pd = false;
          L(i,i) = 1.0;
        } else {
          L(i,i) = std::sqrt(e);
        }
      }
    }
  }
  return L;	
}


// [[Rcpp::export]]
arma::mat matern15_scaledim(arma::vec covparms, arma::mat locs ){

    int dim = locs.n_cols;

    if (static_cast<int>(covparms.n_elem) - 2 != dim) {
        stop("length of covparms does not match dim of locs");
    }
            
    int n = locs.n_rows;
    double nugget = covparms( 0 )*covparms( dim + 1 );

    // create scaled locations
    arma::mat locs_scaled(n,dim);
    for(int j=0; j<dim; j++){ 
        for(int i=0; i<n; i++){
            locs_scaled(i,j) = locs(i,j)/covparms(1+j);
        }
    }

    // calculate covariances
    arma::mat covmat(n,n);
    for(int i1 = 0; i1 < n; i1++){ for(int i2 = 0; i2 <= i1; i2++){
            
        // calculate distance
        double d = 0.0;
        for(int j=0; j<dim; j++){
            d += pow( locs_scaled(i1,j) - locs_scaled(i2,j), 2.0 );
        }
        d = pow( d, 0.5 );
        
        if( d == 0.0 ){
            covmat(i2,i1) = covparms(0);
        } else {
            // calculate covariance            
            covmat(i2,i1) = covparms(0)*(1 + d)*exp(-d);
        }
        // add nugget
        if( i1 == i2 ){ covmat(i2,i2) += nugget; } 
        // fill in opposite entry
        else { covmat(i1,i2) = covmat(i2,i1); }
    }}
    return covmat;
}

// [[Rcpp::export]]
arma::cube d_matern15_scaledim(arma::vec covparms, arma::mat locs ){

    int dim = locs.n_cols;
    if( static_cast<int>(covparms.n_elem) - 2 != dim ){
        stop("length of covparms does not match dim of locs");
    }
            
    int n = locs.n_rows;
   // double nugget = covparms( 0 )*covparms( dim + 1 );

    // create scaled locations
    arma::mat locs_scaled(n,dim);
    for(int j=0; j<dim; j++){ 
        for(int i=0; i<n; i++){
            locs_scaled(i,j) = locs(i,j)/covparms(1+j);
        }
    }

    // calculate derivatives
    arma::cube dcovmat = arma::cube(n,n,covparms.n_elem, arma::fill::zeros);
    for(int i2=0; i2<n; i2++){ for(int i1=0; i1<=i2; i1++){
        
        double d = 0.0;
        for(int j=0; j<dim; j++){
            d += pow( locs_scaled(i1,j) - locs_scaled(i2,j), 2.0 );
        }
        d = pow( d, 0.5 );
        
        double cov;        
        if( d == 0.0 ){
            cov = covparms(0);
            dcovmat(i1,i2,0) += 1.0;
        } else {
            cov = covparms(0)*(1 + d)*exp(-d);
            // variance parameter
            dcovmat(i1,i2,0) += cov/covparms(0);
            // range parameters
            for(int j=0; j<dim; j++){
                double dj2 = pow( locs_scaled(i1,j) - locs_scaled(i2,j), 2.0 );
                dcovmat(i1,i2,j+1) += covparms(0)*exp(-d)*dj2/covparms(j+1);
            }
        }
        if( i1 == i2 ){ // update diagonal entry
            dcovmat(i1,i2,0) += covparms(dim+1);
            dcovmat(i1,i2,dim+1) += covparms(0); 
        } else { // fill in opposite entry
            for(int j=0; j<static_cast<int>(covparms.n_elem); j++){
                dcovmat(i2,i1,j) = dcovmat(i1,i2,j);
            }
        }
    }}
    return dcovmat;
}
void get_covfun(std::string covfun_name_string,  arma::mat (*p_covfun[1])(arma::vec, arma::mat), arma::cube (*p_d_covfun[1])(arma::vec, arma::mat)  )
{
  p_covfun[0] = matern15_scaledim;
  p_d_covfun[0] = d_matern15_scaledim;
}

void compute_pieces_grouped(
    arma::vec covparms, 
    StringVector covfun_name,
    arma::mat locs, 
    List NNlist,
    arma::vec y, 
    arma::mat X,
    arma::mat* XSX,
    arma::vec* ySX,
    double* ySy,
    double* logdet,
    arma::cube* dXSX,
    arma::mat* dySX,
    arma::vec* dySy,
    arma::vec* dlogdet,
    arma::mat* ainfo,
    bool profbeta,
    bool grad_info
){
  
  // data dimensions
  //int n = y.length();
  //int m = NNarray.ncol();
  int p = X.n_cols;
  int nparms = covparms.n_elem;
  int dim = locs.n_cols;
  
  
  // convert StringVector to std::string to use .compare() below
  std::string covfun_name_string;
  covfun_name_string = covfun_name[0];
  
  // assign covariance fun and derivative based on covfun_name_string
  arma::mat (*p_covfun[1])(arma::vec, arma::mat);
  arma::cube (*p_d_covfun[1])(arma::vec, arma::mat);
  get_covfun(covfun_name_string, p_covfun, p_d_covfun);
  
  // vector of all indices
  arma::vec all_inds = NNlist["all_inds"];
  // vector of local response indices
  arma::vec local_resp_inds = as<arma::vec>(NNlist["local_resp_inds"]);
  // vector of global response indices
  arma::vec global_resp_inds = as<arma::vec>(NNlist["global_resp_inds"]);
  // last index of each block in all_inds
  arma::vec last_ind_of_block = as<arma::vec>(NNlist["last_ind_of_block"]);
  // last response index of each block in local_resp_inds and global_resp_inds
  arma::vec last_resp_of_block = as<arma::vec>(NNlist["last_resp_of_block"]);
  
  int nb = last_ind_of_block.n_elem;  // number of blocks
  
#pragma omp parallel 
{   
  arma::mat l_XSX = arma::mat(p, p, arma::fill::zeros);
  arma::vec l_ySX = arma::vec(p, arma::fill::zeros);
  double l_ySy = 0.0;
  double l_logdet = 0.0;
  arma::cube l_dXSX = arma::cube(p,p, nparms, arma::fill::zeros);
  arma::mat l_dySX = arma::mat(p, nparms, arma::fill::zeros);
  arma::vec l_dySy = arma::vec(nparms, arma::fill::zeros);
  arma::vec l_dlogdet = arma::vec(nparms, arma::fill::zeros);
  arma::mat l_ainfo = arma::mat(nparms, nparms, arma::fill::zeros);
  
#pragma omp for	    
  for(int i=0; i<nb; i++){
    
    // first ind and last ind are the positions in all_inds
    // of the observations for block i.
    // these come in 1-indexing and are converted to 0-indexing here
    int first_ind;
    int last_ind;
    if(i==0){ first_ind = 0; } else {first_ind = last_ind_of_block[i-1]+1-1; }
    last_ind = last_ind_of_block[i]-1;
    int bsize = last_ind - first_ind + 1;
    
    int first_resp;
    int last_resp;
    if(i==0){ first_resp = 0; } else {first_resp = last_resp_of_block[i-1]+1-1; }
    last_resp = last_resp_of_block[i]-1;
    int rsize = last_resp - first_resp + 1;
    
    arma::uvec whichresp(rsize);
    for(int j=0; j<rsize; j++){
      whichresp(j) = local_resp_inds(first_resp+j) - 1;
    }
    
    // fill in ysub, locsub, and X0 in forward order
    arma::mat locsub(bsize, dim);
    arma::vec ysub(bsize);
    arma::mat X0(bsize,p);
    for(int j=0; j<bsize; j++){
      int jglobal = all_inds[first_ind + j] - 1;
      ysub(j) = y( jglobal );
      for(int k=0;k<dim;k++){ locsub(j,k) = locs( jglobal, k ); }
      if(profbeta){
        for(int k=0;k<p;k++){ X0(j,k) = X( jglobal, k ); }
      }
    }
    
    // compute covariance matrix and derivatives and take cholesky
    arma::mat covmat = p_covfun[0]( covparms, locsub );
    
    arma::cube dcovmat(bsize,bsize,nparms);
    if(grad_info){
      dcovmat = p_d_covfun[0]( covparms, locsub );
    }
    
    //arma::mat cholmat = eye( size(covmat) );
    //chol( cholmat, covmat, "lower" );
    arma::mat cholmat = mychol(covmat);
    
    // get response rows of inverse cholmat, put in column vectors
    arma::mat onemat = arma::zeros(bsize,rsize);
    for(int j=0; j<rsize; j++){ 
      onemat(whichresp(j),j) = 1.0;
    }
    
    arma::mat choli2(bsize,bsize);
    if(grad_info){
      //choli2 = solve( trimatu(cholmat.t()), onemat );
      choli2 = backward_solve_mat( cholmat, onemat );
    }
    
    bool cond = bsize > rsize;
    //double fac = 1.0;
    
    // do solves with X and y
    arma::mat LiX0( bsize, p );
    if(profbeta){
      //LiX0 = solve( trimatl(cholmat), X0 );
      LiX0 = forward_solve_mat( cholmat, X0 );
    }
    
    //arma::vec Liy0 = solve( trimatl(cholmat), ysub );
    arma::vec Liy0 = forward_solve( cholmat, ysub );
    
    // loglik objects
    for(int j=0; j<rsize; j++){
      int ii = whichresp(j);
      l_logdet += 2.0*std::log( as_scalar(cholmat(ii,ii)) ); 
      l_ySy +=    pow( as_scalar(Liy0(ii)), 2 );
      if(profbeta){
        l_XSX +=    LiX0.row(ii).t() * LiX0.row(ii);
        l_ySX += ( Liy0(ii) * LiX0.row(ii) ).t();
      }
    }    
    
    if(grad_info){
      // gradient objects
      // LidSLi3 is last column of Li * (dS_j) * Lit for 1 parameter i
      // LidSLi2 stores these columns in a matrix for all parameters
      if(cond){ // if we condition on anything
        
        arma::cube LidSLi2 = arma::cube(bsize,rsize,nparms,fill::zeros);
        for(int j=0; j<nparms; j++){
          // compute last column of Li * (dS_j) * Lit
          //arma::mat LidSLi4 = solve( trimatl(cholmat), dcovmat.slice(j) * choli2 );
          arma::mat LidSLi4 = forward_solve_mat( cholmat, dcovmat.slice(j) * choli2 );
          
          for(int k=0; k<rsize; k++){
            int i2 = whichresp(k);
            arma::span i1 = span(0,i2);
            arma::vec LidSLi3 = LidSLi4(i1,k); 
            // store LiX0.t() * LidSLi3 and Liy0.t() * LidSLi3
            arma::vec v1 = LiX0.rows(i1).t() * LidSLi3;
            double s1 = dot( Liy0(i1), LidSLi3 ); 
            // update all quantities
            // bottom-right corner gets double counted, so need to subtract it off
            (l_dXSX).slice(j) += v1 * LiX0.row(i2) + ( v1 * LiX0.row(i2) ).t() - 
              LidSLi3(i2) * LiX0.row(i2).t() * LiX0.row(i2);
            (l_dySy)(j) += 2.0 * s1 * Liy0(i2)  - 
              LidSLi3(i2) * Liy0(i2) * Liy0(i2);
            (l_dySX).col(j) += (  s1 * LiX0.row(i2) + ( v1 * Liy0(i2) ).t() -  
              LidSLi3(i2) * LiX0.row(i2) * Liy0(i2) ).t();
            (l_dlogdet)(j) += LidSLi3(i2);
            // store last column of Li * (dS_j) * Lit
            LidSLi2.subcube(i1, span(k,k), span(j,j)) = LidSLi3;
          }
        }
        
        // fisher information object
        // bottom right corner gets double counted, so subtract it off
        for(int i=0; i<nparms; i++){ for(int j=0; j<i+1; j++){
          (l_ainfo)(i,j) += accu( LidSLi2.slice(i) % LidSLi2.slice(j) );
          for(int k=0; k<rsize; k++){
            int i2 = whichresp(k);
            (l_ainfo)(i,j) -= 0.5*LidSLi2(i2,k,j) * LidSLi2(i2,k,i);
          }
        }}
      } else { // similar calculations, but for when there is no conditioning set
        arma::cube LidSLi2(bsize,bsize,nparms);
        for(int j=0; j<nparms; j++){
          //arma::mat LidSLi = solve( trimatl(cholmat), dcovmat.slice(j) );
          arma::mat LidSLi = forward_solve_mat( cholmat, dcovmat.slice(j) );
          //LidSLi = solve( trimatl(cholmat), LidSLi.t() );
          LidSLi = forward_solve_mat( cholmat, LidSLi.t() );
          (l_dXSX).slice(j) += LiX0.t() *  LidSLi * LiX0; 
          (l_dySy)(j) += as_scalar( Liy0.t() * LidSLi * Liy0 );
          (l_dySX).col(j) += ( ( Liy0.t() * LidSLi ) * LiX0 ).t();
          (l_dlogdet)(j) += trace( LidSLi );
          LidSLi2.slice(j) = LidSLi;
        }
        
        // fisher information object
        for(int i=0; i<nparms; i++){ for(int j=0; j<i+1; j++){
          (l_ainfo)(i,j) += 0.5*accu( LidSLi2.slice(i) % LidSLi2.slice(j) ); 
        }}
      }
    }
    
  }
  
#pragma omp critical
{
  *XSX += l_XSX;
  *ySX += l_ySX;
  *ySy += l_ySy;
  *logdet += l_logdet;
  *dXSX += l_dXSX;
  *dySX += l_dySX;
  *dySy += l_dySy;
  *dlogdet += l_dlogdet;
  *ainfo += l_ainfo;
}
}
}    

void synthesize_grouped(
    NumericVector covparms, 
    StringVector covfun_name,
    const NumericMatrix locs, 
    List NNlist,
    NumericVector& y, 
    NumericMatrix X,
    NumericVector* ll, 
    NumericVector* betahat,
    NumericVector* grad,
    NumericMatrix* info,
    NumericMatrix* betainfo,
    bool profbeta,
    bool grad_info){
  
  // data dimensions
  int n = y.length();
  //int m = NNarray.ncol();
  int p = X.ncol();
  int nparms = covparms.length();
  //int dim = locs.ncol();
  
  // likelihood objects
  arma::mat XSX = arma::mat(p, p, arma::fill::zeros);
  arma::vec ySX = arma::vec(p, arma::fill::zeros);
  double ySy = 0.0;
  double logdet = 0.0;
  
  // gradient objects    
  arma::cube dXSX = arma::cube(p,p,nparms,arma::fill::zeros);
  arma::mat dySX = arma::mat(p, nparms, arma::fill::zeros);
  arma::vec dySy = arma::vec(nparms, arma::fill::zeros);
  arma::vec dlogdet = arma::vec(nparms, arma::fill::zeros);
  // fisher information
  arma::mat ainfo = arma::mat(nparms, nparms, arma::fill::zeros);
  
  // this is where the big computation happens
  
  // convert Numeric- to arma
  arma::vec covparms_c = arma::vec(covparms.begin(),covparms.length());
  arma::mat locs_c = arma::mat(locs.begin(),locs.nrow(),locs.ncol());
  arma::vec y_c = arma::vec(y.begin(),y.length());
  arma::mat X_c = arma::mat(X.begin(),X.nrow(),X.ncol());
  
  
  compute_pieces_grouped(
    covparms_c, covfun_name, locs_c, NNlist, y_c, X_c,
    &XSX, &ySX, &ySy, &logdet, &dXSX, &dySX, &dySy, &dlogdet, &ainfo,
    profbeta, grad_info
  );
  
  // synthesize everything and update loglik, grad, beta, betainfo, info
  
  // betahat and dbeta
  arma::vec abeta = arma::vec(p,arma::fill::zeros);
  if(profbeta){ abeta = solve( XSX, ySX ); }
  for(int j=0; j<p; j++){ (*betahat)(j) = abeta(j); };
  
  arma::mat dbeta(p,nparms);
  if(profbeta && grad_info){
    for(int j=0; j<nparms; j++){
      dbeta.col(j) = solve( XSX, dySX.col(j) - dXSX.slice(j) * abeta );
    }
  }
  
  // get sigmahatsq
  double sig2 = ( ySy - 2.0*as_scalar( ySX.t() * abeta ) + 
                  as_scalar( abeta.t() * XSX * abeta ) )/n;
  // loglikelihood
  (*ll)(0) = -0.5*( n*std::log(2.0*M_PI) + logdet + n*sig2 ); 
  
  if(profbeta){    
    // betainfo
    for(int i=0; i<p; i++){ for(int j=0; j<i+1; j++){
      (*betainfo)(i,j) = XSX(i,j);
      (*betainfo)(j,i) = XSX(j,i);
    }}
  }
  
  if(grad_info){
    // gradient
    for(int j=0; j<nparms; j++){
      (*grad)(j) = 0.0;
      (*grad)(j) -= 0.5*dlogdet(j);
      (*grad)(j) += 0.5*dySy(j);
      (*grad)(j) -= 1.0*as_scalar( abeta.t() * dySX.col(j) );
      (*grad)(j) += 1.0*as_scalar( ySX.t() * dbeta.col(j) );
      (*grad)(j) += 0.5*as_scalar( abeta.t() * dXSX.slice(j) * abeta );
      (*grad)(j) -= 1.0*as_scalar( abeta.t() * XSX * dbeta.col(j) );
    }
    // fisher information
    for(int i=0; i<nparms; i++){ for(int j=0; j<i+1; j++){
      (*info)(i,j) = ainfo(i,j);
      (*info)(j,i) = (*info)(i,j);
    }}
  }
}


// [[Rcpp::export]]
List vecchia_grouped_profbeta_loglik_grad_info(
    NumericVector covparms, 
    StringVector covfun_name,
    NumericVector y,
    NumericMatrix X,
    const NumericMatrix locs,
    List NNlist) {
  
  NumericVector ll(1);
  NumericVector grad(covparms.size());
  NumericVector betahat(X.ncol());
  NumericMatrix info(covparms.size(), covparms.size());
  NumericMatrix betainfo(X.ncol(), X.ncol());

  synthesize_grouped(covparms, covfun_name, locs, NNlist, y, X,
                     &ll, &betahat, &grad, &info, &betainfo, true, true);
  
 
  List ret = List::create(Named("loglik") = ll, 
                          Named("betahat") = betahat,
                          Named("grad") = grad, 
                          Named("info") = info, 
                          Named("betainfo") = betainfo);
  return ret;
}


#include <Rcpp.h>
#include <cmath>
#include <cstring>
#ifdef _WIN32
#include <malloc.h>
#endif
#include <cstdlib>
using namespace std;
using namespace Rcpp;



/*A struct containing all the information relevant to the data point*/
typedef struct point {
  int Id;
  int level;
  int d;
  double* x;
  int NChildren;
  int firstChildKey;
  struct point* parent;
  /*Heap Stuff:*/
  double distSQ;
  int hlevel;
  struct point** hnode;
  int hId;
  int leaf;
} point;


/*creates a set of n^2 points forming 2-dimensional grid*/
double* create_coords2d(const int n) {
  double h = 1.0 / ((double)(n + 1));
  int k, l;
  int N = pow(n, 2);
  /*allocates a N times 2 array*/
  double* coords = (double*)malloc(sizeof(double) * 2 * N);
  //if (coords == NULL) exit(1);
  /*Fill the allocated array with the right coordinates*/
  for (k = 0; k < n; ++k) {
    for (l = 0; l < n; ++l) {
      coords[2 * (k*n + l) + 0] = h * (double)(l + 1);
      coords[2 * (k*n + l) + 1] = h * (double)(k + 1);
    }
  }
  return coords;
}

/*creates point structs from an input of coordinates*/
point* create_Points(double* const x, const int d, const int N) {
  //int k, l;
  int k;
  point* points = (point*)malloc(sizeof(point) * N);
  //if (points == NULL) exit(1);
  for (k = 0; k < N; ++k) {
    points[k].Id = k;
    points[k].d = d;
    points[k].x = &x[d * k];
    points[k].parent = NULL;
    points[k].firstChildKey = 0;
    points[k].NChildren = 0;
    points[k].distSQ = 10000.;
  }
  return points;
}

/*destructor for points*/
void destruct_coords(double* coords) {
  free(coords);
}


/*A max heap to keep track of the next point to be updated*/
typedef struct heap {
  point** elements;
  int N;
} heap;

/*Function comparing squared distances*/
int compareSQ(const void* p1, const void*p2) {
  double diff = ((double)(*((point**)p1))->distSQ - (double)(*((point**)p2))->distSQ);
  if (diff > 0) return -1;
  else return 1;
}

/*Function comparing levels*/
int compareLevel(const void* p1, const void*p2) {
  return ((point*)p1)->level - ((point*)p1)->level;
}


 /*A struct that provides storage for the children of the individual nodes*/
 typedef struct daycare {
   int size;
   int sizeBuffer;
   point** data;
 } daycare;

void daycare_init(daycare* dc, const int N) {
  // int i;
  dc->sizeBuffer = 2 * N;
  dc->size = 0;
  dc->data = (point**)malloc(dc->sizeBuffer * sizeof(point*));
  //if (dc->data == NULL) exit(1);
}

void daycare_add(daycare* dc, point* addendum) {
  if (dc->size == dc->sizeBuffer) {
    dc->sizeBuffer *= 2;
    dc->data = (point**)realloc(dc->data, dc->sizeBuffer * sizeof(point*));
    //if (dc->data == NULL) exit(1);
  }
  ++dc->size;
  dc->data[dc->size - 1] = addendum;
}

void daycare_destruct(daycare* dc) {
  free(dc->data);
}

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
// From sortSparse.h & sortSparse.c

typedef struct heapNode {
  /*distance squared to closest point that is already taken out, negative for points that are taken out*/
  double dist;
  struct heapNode** handleHandle;
  struct heapNode* leftChild;
  struct heapNode* rightChild;
  /*might not be needed:*/
  unsigned int Id;
} heapNode;

heapNode* _moveDown(heapNode* const a) {
  /*If the nodes has no children:*/
  if (a->leftChild == NULL) {
    return NULL;
  }
  /*If the node has only one child:*/
  if (a->rightChild == NULL) {
    if (a->dist < a->leftChild->dist) {
      /*swaps a with its left child*/
      const double tempDist = a->dist;
      a->dist = a->leftChild->dist;
      a->leftChild->dist = tempDist;
      *(a->handleHandle) = a->leftChild;
      *(a->leftChild->handleHandle) = a;
      heapNode** const tempHandleHandle = a->handleHandle;
      a->handleHandle = a->leftChild->handleHandle;
      a->leftChild->handleHandle = tempHandleHandle;
      
      const int tempId = a->Id;
      a->Id = a->leftChild->Id;
      a->leftChild->Id = tempId;
      
      return a->leftChild;
    }
    else return NULL;
  }
  /*If the node has two children:*/
  if (a->leftChild->dist > a->rightChild->dist) {
    if (a->dist < a->leftChild->dist) {
      /*swaps a with its left child*/
      const double tempDist = a->dist;
      a->dist = a->leftChild->dist;
      a->leftChild->dist = tempDist;
      *(a->handleHandle) = a->leftChild;
      *(a->leftChild->handleHandle) = a;
      heapNode** const tempHandleHandle = a->handleHandle;
      a->handleHandle = a->leftChild->handleHandle;
      a->leftChild->handleHandle = tempHandleHandle;
      
      const int tempId = a->Id;
      a->Id = a->leftChild->Id;
      a->leftChild->Id = tempId;
      
      return a->leftChild;
    }
    else return NULL;
  }
  else {
    if (a->dist < a->rightChild->dist) {
      /*swaps a with its right child*/
      const double tempDist = a->dist;
      a->dist = a->rightChild->dist;
      a->rightChild->dist = tempDist;
      *(a->handleHandle) = a->rightChild;
      *(a->rightChild->handleHandle) = a;
      heapNode** const tempHandleHandle = a->handleHandle;
      a->handleHandle = a->rightChild->handleHandle;
      a->rightChild->handleHandle = tempHandleHandle;
      
      const int tempId = a->Id;
      //printf( "newId = %d ", a->rightChild->Id);
      a->Id = a->rightChild->Id;
      a->rightChild->Id = tempId;
      
      
      return a->rightChild;
    }
    else return NULL;
  }
}

/*Only works as expected, if the new distance is smaller than the old one*/
void update(heapNode* target, const double newDist) {
  target->dist = newDist;
  while (target != NULL) {
    target = _moveDown(target);
  }
}

/*Initialises array of nodes with proper children and writes it into nodes*/
void heapInit(const unsigned int N, heapNode* const nodes, heapNode** const handles) {
  /*heapNode* ret = (heapNode*) malloc( N * sizeof( heapNode ) );
   if( ret == NULL ) exit(1);*/
  for (unsigned int k = 0; k < N; ++k) {
    if (2 * k + 1 >= N) {
      heapNode newnode = { INFINITY, &handles[k], NULL, NULL };
      memcpy(&(nodes[k]), &newnode, sizeof(heapNode));
    }
    else if (2 * k + 2 >= N) {
      heapNode newnode = { INFINITY, &handles[k], &nodes[2 * k + 1], NULL };
      memcpy(&(nodes[k]), &newnode, sizeof(heapNode));
    }
    else {
      heapNode newnode = { 10000., &handles[k], &nodes[2 * k + 1], &nodes[2 * k + 2] };
      memcpy(&(nodes[k]), &newnode, sizeof(heapNode));
    }
    handles[k] = &nodes[k];
    nodes[k].Id = k;
  }
}

typedef struct ijlookup {
  unsigned int pres_i;
  unsigned int N;
  unsigned int S;
  unsigned int S_Buffer;
  unsigned int* i;
  unsigned int* j;
} ijlookup;

void ijlookup_init(ijlookup* lookup, unsigned int N) {
  lookup->pres_i = 0;
  lookup->N = N;
  lookup->S = 0;
  lookup->S_Buffer = N;
  lookup->i = (unsigned int*)malloc((N + 1) * sizeof(unsigned int));
  //if (lookup->i == NULL) exit(1);
  lookup->j = (unsigned int*)malloc(N * sizeof(unsigned int));
  //if (lookup->j == NULL) exit(1);
  lookup->i[0] = 0;
  lookup->i[1] = 0;
}

void ijlookup_newparent(ijlookup* const lookup) {
  ++lookup->pres_i;
  lookup->i[lookup->pres_i + 1] = lookup->i[lookup->pres_i];
}

void ijlookup_newson(ijlookup* const lookup, const unsigned int Id) {
  ++lookup->S;
  if (lookup->S > lookup->S_Buffer) {
    lookup->S_Buffer *= 2;
    lookup->j = (unsigned int*)realloc(lookup->j, lookup->S_Buffer * sizeof(unsigned int));
    //if (lookup->j == NULL) exit(1);
  }
  lookup->j[lookup->S - 1] = Id;
  ++lookup->i[lookup->pres_i + 1];
}

void ijlookup_destruct(ijlookup* lookup) {
  free(lookup->i);
  free(lookup->j);
}

double dist(const unsigned int i, const unsigned int j, const double* const coords, const unsigned int d) {
  double ret = 0;
  for (int k = 0; k < (int)d; ++k) {
    ret += (coords[d * i + k] - coords[d * j + k]) * (coords[d * i + k] - coords[d * j + k]);
  }
  return sqrt(ret);
}

double dist_2d(const unsigned int i, const unsigned int j, const double* const coords) {
  return sqrt((coords[2 * i] - coords[2 * j]) * (coords[2 * i] - coords[2 * j])
                + (coords[2 * i + 1] - coords[2 * j + 1]) * (coords[2 * i + 1] - coords[2 * j + 1]));
}

double dist_3d(const unsigned int i, const unsigned int j, const double* const coords) {
  return sqrt((coords[3 * i] - coords[3 * j]) * (coords[3 * i] - coords[3 * j])
                + (coords[3 * i + 1] - coords[3 * j + 1]) * (coords[3 * i + 1] - coords[3 * j + 1])
                + (coords[3 * i + 2] - coords[3 * j + 2]) * (coords[3 * i + 2] - coords[3 * j + 2]));
}

double dist2(const unsigned int i, const unsigned int j, const double* const coords, const unsigned int d) {
  double ret = 0;
  for (int k = 0; k < (int)d; ++k) {
    ret += (coords[d * i + k] - coords[d * j + k]) * (coords[d * i + k] - coords[d * j + k]);
  }
  return ret;
}

double dist2_2d(const unsigned int i, const unsigned int j, const double* const coords) {
  return (coords[2 * i] - coords[2 * j]) * (coords[2 * i] - coords[2 * j])
  + (coords[2 * i + 1] - coords[2 * j + 1]) * (coords[2 * i + 1] - coords[2 * j + 1]);
}

double dist2_3d(const unsigned int i, const unsigned int j, const double* const coords) {
  return (coords[3 * i] - coords[3 * j]) * (coords[3 * i] - coords[3 * j])
  + (coords[3 * i + 1] - coords[3 * j + 1]) * (coords[3 * i + 1] - coords[3 * j + 1])
  + (coords[3 * i + 2] - coords[3 * j + 2]) * (coords[3 * i + 2] - coords[3 * j + 2]);
}

//void dist2Blocked(double* results, const unsigned int i, const unsigned int* const j0, const unsigned int n, const double* const coords, const unsigned int d);
//void dist2BlockedVec(double* results, const double* const ivec, const unsigned int* const j0, const unsigned int n, const double* const coords, const unsigned int d);
double dist2fix(const double* x, const unsigned int j, const double* const coords, const unsigned int d);

inline double in_dist2(const unsigned int i, const unsigned int j, const double* const coords, const unsigned int d) {
  double ret = 0;
  for (unsigned int k = 0; k < d; ++k) {
    ret += (coords[d * i + k] - coords[d * j + k]) * (coords[d * i + k] - coords[d * j + k]);
  }
  return ret;
}

void  determineChildren(heapNode* const nodes, heapNode** const handles, ijlookup* const lookup, unsigned int* const parents, const double* const coords, const unsigned int d, const unsigned int N, const unsigned int Id, const unsigned int iter) {
  const double pivotDist = nodes[0].dist;
  /*Need to save the kmin nad kmax beforehand, since otherwise the node might become its own parent, later on */
  const int kmin = lookup->i[parents[Id]];
  const int kmax = lookup->i[parents[Id] + 1];
  ijlookup_newparent(lookup);
  
  
  for (unsigned int k = kmin; (int)k < kmax; ++k) {
    //if( lookup->j[ k ] >= iter ){
    const double tempDist2 = dist2(Id, lookup->j[k], coords, d);
    if (tempDist2 < pivotDist * pivotDist) {
      double jDist = handles[lookup->j[k]]->dist;
      if (tempDist2 < jDist*jDist) {
        update(handles[lookup->j[k]], sqrt(tempDist2));
        jDist = sqrt(tempDist2);
      }
      ijlookup_newson(lookup, lookup->j[k]);
      if (sqrt(tempDist2) + jDist < pivotDist) {
        parents[lookup->j[k]] = iter;
      }
    }
    //}
  }
}

void determineChildren_2d(heapNode* const nodes, heapNode** const handles, ijlookup* const lookup, unsigned int* const parents, const double* const coords, const unsigned int N, const unsigned int Id, const unsigned int iter) {
  const double pivotDist = nodes[0].dist;
  /*Need to save the kmin nad kmax beforehand, since otherwise the node might become its own parent, later on */
  const int kmin = lookup->i[parents[Id]];
  const int kmax = lookup->i[parents[Id] + 1];
  ijlookup_newparent(lookup);
  
  
  for (unsigned int k = kmin; (int)k < kmax; ++k) {
    //if( lookup->j[ k ] >= iter ){
    const double tempDist2 = dist2_2d(Id, lookup->j[k], coords);
    if (tempDist2 < pivotDist * pivotDist) {
      double jDist = handles[lookup->j[k]]->dist;
      if (tempDist2 < jDist*jDist) {
        update(handles[lookup->j[k]], sqrt(tempDist2));
        jDist = sqrt(tempDist2);
      }
      ijlookup_newson(lookup, lookup->j[k]);
      if (sqrt(tempDist2) + jDist < pivotDist) {
        parents[lookup->j[k]] = iter;
      }
    }
    //}
  }
}

void determineChildren_3d(heapNode* const nodes, heapNode** const handles, ijlookup* const lookup, unsigned int* const parents, const double* const coords, const unsigned int N, const unsigned int Id, const unsigned int iter) {
  const double pivotDist = nodes[0].dist;
  /*Need to save the kmin nad kmax beforehand, since otherwise the node might become its own parent, later on */
  const int kmin = lookup->i[parents[Id]];
  const int kmax = lookup->i[parents[Id] + 1];
  ijlookup_newparent(lookup);
  
  
  for (unsigned int k = kmin; (int)k < kmax; ++k) {
    //if( lookup->j[ k ] >= iter ){
    const double tempDist2 = dist2_3d(Id, lookup->j[k], coords);
    if (tempDist2 < pivotDist * pivotDist) {
      double jDist = handles[lookup->j[k]]->dist;
      if (tempDist2 < jDist*jDist) {
        update(handles[lookup->j[k]], sqrt(tempDist2));
        jDist = sqrt(tempDist2);
      }
      ijlookup_newson(lookup, lookup->j[k]);
      if (sqrt(tempDist2) + jDist < pivotDist) {
        parents[lookup->j[k]] = iter;
      }
    }
    //}
  }
}


//void  determineChildrenBlocked(heapNode* const nodes, heapNode** const handles, ijlookup* const lookup, unsigned int* const parents, const double* const coords, const unsigned int d, const unsigned int N, const unsigned int Id, const unsigned int iter);

void create_ordering(unsigned int* P, unsigned int* revP, double* distances, const unsigned int d, const unsigned int N, const double* coords, unsigned int first_node) {
  /*Function to construct the ordering.
   * Inputs:
   *  P:
   *    An N element array containing the hierarchical ordering
   *  revP:
   *    An N element array containing the inverse of the hierarchical ordering
   *  distances:
   *    An N element array containing the distance ( length scale ) of each dof
   *  d:
   *    The number of spatial dimensions
   *  N:
   *    The number of points
   *  coords:
   *    An d*N element array that contains the different points coordinates, with the
   *    coordinates of a given point in contiguous memory
   */
  
  /*Allocate the heap structure:*/
  heapNode* nodes = (heapNode*)malloc(N * sizeof(heapNode));
  //if (nodes == NULL) exit(1);
  heapNode** handles = (heapNode**)malloc(N * sizeof(heapNode*));
  //if (handles == NULL) exit(1);
  /*Initiate the heap structure*/
  heapInit(N, nodes, handles);
  /*initialising lookup*/
  ijlookup lookup;
  ijlookup_init(&lookup, N);
  /*allocate array to store the parents of dof:
   *the i-th entry of parents will contain the number of its parent in the ordering */
  unsigned int* parents = (unsigned int*)malloc(N * sizeof(unsigned int));
  //if (parents == NULL) exit(1);
  
  /* Add the first parent node: */
  /*TODO Make random?*/
  unsigned int rootId = first_node;
  distances[0] = 0.;
  for (unsigned int k = 0; k < N; ++k) {
    ijlookup_newson(&lookup, k);
    if (dist(rootId, k, coords, d) > distances[0]) {
      distances[0] = dist(rootId, k, coords, d);
    }
    update(handles[k], dist(rootId, k, coords, d));
    parents[k] = 0;
  }
  
  for (unsigned int k = 1; k < N; ++k) {
    unsigned int pivotId = nodes[0].handleHandle - handles;
    distances[k] = nodes[0].dist;
    P[k] = pivotId;
    revP[pivotId] = k;
    determineChildren(nodes, handles, &lookup, parents, coords, d, N, pivotId, k);
  }
  
  ijlookup_destruct(&lookup);
  free(parents);
  free(handles);
  free(nodes);
}

void create_ordering_2d(unsigned int* P, unsigned int* revP, double* distances, const unsigned int N, const double* coords, unsigned int first_node) {
  /*Function to construct the ordering.
   * Inputs:
   *  P:
   *    An N element array containing the hierarchical ordering
   *  revP:
   *    An N element array containing the inverse of the hierarchical ordering
   *  distances:
   *    An N element array containing the distance ( length scale ) of each dof
   *  N:
   *    The number of points
   *  coords:
   *    An d*N element array that contains the different points coordinates, with the
   *    coordinates of a given point in contiguous memory
   */
  
  /*Allocate the heap structure:*/
  heapNode* nodes = (heapNode*)malloc(N * sizeof(heapNode));
  //if (nodes == NULL) exit(1);
  heapNode** handles = (heapNode**)malloc(N * sizeof(heapNode*));
  //if (handles == NULL) exit(1);
  /*Initiate the heap structure*/
  heapInit(N, nodes, handles);
  /*initialising lookup*/
  ijlookup lookup;
  ijlookup_init(&lookup, N);
  /*allocate array to store the parents of dof:
   *the i-th entry of parents will contain the number of its parent in the ordering */
  unsigned int* parents = (unsigned int*)malloc(N * sizeof(unsigned int));
  //if (parents == NULL) exit(1);
  
  /* Add the first parent node: */
  /*TODO Make random?*/
  unsigned int rootId = first_node;
  distances[0] = 0.;
  for (unsigned int k = 0; k < N; ++k) {
    ijlookup_newson(&lookup, k);
    if (dist_2d(rootId, k, coords) > distances[0]) {
      distances[0] = dist_2d(rootId, k, coords);
    }
    update(handles[k], dist_2d(rootId, k, coords));
    parents[k] = 0;
  }
  
  for (unsigned int k = 1; k < N; ++k) {
    unsigned int pivotId = nodes[0].handleHandle - handles;
    distances[k] = nodes[0].dist;
    P[k] = pivotId;
    revP[pivotId] = k;
    determineChildren_2d(nodes, handles, &lookup, parents, coords, N, pivotId, k);
  }
  
  ijlookup_destruct(&lookup);
  free(parents);
  free(handles);
  free(nodes);
}


void create_ordering_3d(unsigned int* P, unsigned int* revP, double* distances, const unsigned int N, const double* coords, unsigned int first_node) {
  /*Function to construct the ordering.
   * Inputs:
   *  P:
   *    An N element array containing the hierarchical ordering
   *  revP:
   *    An N element array containing the inverse of the hierarchical ordering
   *  distances:
   *    An N element array containing the distance ( length scale ) of each dof
   *  N:
   *    The number of points
   *  coords:
   *    An d*N element array that contains the different points coordinates, with the
   *    coordinates of a given point in contiguous memory
   */
  
  /*Allocate the heap structure:*/
  heapNode* nodes = (heapNode*)malloc(N * sizeof(heapNode));
  //if (nodes == NULL) exit(1);
  heapNode** handles = (heapNode**)malloc(N * sizeof(heapNode*));
  //if (handles == NULL) exit(1);
  /*Initiate the heap structure*/
  heapInit(N, nodes, handles);
  /*initialising lookup*/
  ijlookup lookup;
  ijlookup_init(&lookup, N);
  /*allocate array to store the parents of dof:
   *the i-th entry of parents will contain the number of its parent in the ordering */
  unsigned int* parents = (unsigned int*)malloc(N * sizeof(unsigned int));
  //if (parents == NULL) exit(1);
  
  /* Add the first parent node: */
  /*TODO Make random?*/
  unsigned int rootId = first_node;
  distances[0] = 0.;
  for (unsigned int k = 0; k < N; ++k) {
    ijlookup_newson(&lookup, k);
    if (dist_3d(rootId, k, coords) > distances[0]) {
      distances[0] = dist_3d(rootId, k, coords);
    }
    update(handles[k], dist_3d(rootId, k, coords));
    parents[k] = 0;
  }
  
  for (unsigned int k = 1; k < N; ++k) {
    unsigned int pivotId = nodes[0].handleHandle - handles;
    distances[k] = nodes[0].dist;
    P[k] = pivotId;
    revP[pivotId] = k;
    determineChildren_3d(nodes, handles, &lookup, parents, coords, N, pivotId, k);
  }
  
  ijlookup_destruct(&lookup);
  free(parents);
  free(handles);
  free(nodes);
}


//[[Rcpp::export]]
IntegerVector MaxMincpp(NumericMatrix locations)
{
  unsigned int N = locations.nrow();
  int dim = locations.ncol();
  IntegerVector res(N);
  unsigned int* P = (unsigned int*)malloc(N * sizeof(unsigned int));
  if (P == NULL) return res;
  unsigned int* revP = (unsigned int*)malloc(N * sizeof(unsigned int));
  if (revP == NULL) return res;
  double* distances = (double*)malloc(N * sizeof(double));
  if (distances == NULL) return res;
  double *coords = (double*)malloc(dim * N * sizeof(double));
  
  // Find the average point.
  unsigned int first_node;
  double *average_arr, cur_dist2, min_dist2;
  average_arr = new double[dim];
  for(int j = 0; j < dim; j++)
  {
    average_arr[j] = 0.0;
  }
  for (int i = 0; i < (int)N; i++)
  {
    for(int j = 0; j < dim; j++)
    {
      average_arr[j] += (coords[dim * i + j] = locations(i, j));
    }
  }
  for(int j = 0; j < dim; j++)
  {
    average_arr[j] /= N;
  }
  min_dist2 = -1;
  first_node = -1;
  for (int i = 0; i < (int)N; i++)
  {
    cur_dist2 = 0;
    for(int j = 0; j < dim; j++)
    {
      cur_dist2 += (coords[dim * i + j] - average_arr[j]) * (coords[dim * i + j] - average_arr[j]);
    }
    if (min_dist2 < 0 || cur_dist2 < min_dist2)
    {
      min_dist2 = cur_dist2;
      first_node = i;
    }
  }
  
  delete[] average_arr;
  if (dim == 2)
  {
    create_ordering_2d(P, revP, distances, N, coords, first_node);
  }
  else if (dim == 3)
  {
    create_ordering_3d(P, revP, distances, N, coords, first_node);
  }
  else if (dim >= 1)
  {
    create_ordering(P, revP, distances, dim, N, coords, first_node);
  }
  else
  {
    free(P);
    free(revP);
    free(distances);
    destruct_coords(coords);
    return res;
  }
  res[0] = first_node + 1;
  for (int i = 1; i < (int)N; i++)
    res[i] = P[i] + 1;
  free(P);
  free(revP);
  free(distances);
  destruct_coords(coords);
  return res;
}
