// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// computeWeightedY
arma::vec computeWeightedY(const arma::mat& X, const arma::vec& y, List params);
RcppExport SEXP _DSWE_computeWeightedY(SEXP XSEXP, SEXP ySEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< List >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(computeWeightedY(X, y, params));
    return rcpp_result_gen;
END_RCPP
}
// predictGP
arma::vec predictGP(const arma::mat& X, const arma::vec& weightedY, const arma::mat& Xnew, List params);
RcppExport SEXP _DSWE_predictGP(SEXP XSEXP, SEXP weightedYSEXP, SEXP XnewSEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type weightedY(weightedYSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type Xnew(XnewSEXP);
    Rcpp::traits::input_parameter< List >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(predictGP(X, weightedY, Xnew, params));
    return rcpp_result_gen;
END_RCPP
}
// computeLogLikGP_
double computeLogLikGP_(const arma::mat& X, const arma::vec& y, const List& params);
RcppExport SEXP _DSWE_computeLogLikGP_(SEXP XSEXP, SEXP ySEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const List& >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(computeLogLikGP_(X, y, params));
    return rcpp_result_gen;
END_RCPP
}
// computeLogLikGradGP_
arma::vec computeLogLikGradGP_(const arma::mat& X, const arma::vec& y, List params);
RcppExport SEXP _DSWE_computeLogLikGradGP_(SEXP XSEXP, SEXP ySEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< List >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(computeLogLikGradGP_(X, y, params));
    return rcpp_result_gen;
END_RCPP
}
// computeLogLikGradGPZeroMean_
arma::vec computeLogLikGradGPZeroMean_(const arma::mat& X, const arma::vec& y, List params);
RcppExport SEXP _DSWE_computeLogLikGradGPZeroMean_(SEXP XSEXP, SEXP ySEXP, SEXP paramsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< List >::type params(paramsSEXP);
    rcpp_result_gen = Rcpp::wrap(computeLogLikGradGPZeroMean_(X, y, params));
    return rcpp_result_gen;
END_RCPP
}
// computeDiffCov_
List computeDiffCov_(const arma::mat& X1, const arma::vec y1, const arma::mat& X2, const arma::vec y2, const arma::mat& XT, arma::vec theta, double sigma_f, double sigma_n, double beta);
RcppExport SEXP _DSWE_computeDiffCov_(SEXP X1SEXP, SEXP y1SEXP, SEXP X2SEXP, SEXP y2SEXP, SEXP XTSEXP, SEXP thetaSEXP, SEXP sigma_fSEXP, SEXP sigma_nSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X1(X1SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type y1(y1SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type X2(X2SEXP);
    Rcpp::traits::input_parameter< const arma::vec >::type y2(y2SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type XT(XTSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_f(sigma_fSEXP);
    Rcpp::traits::input_parameter< double >::type sigma_n(sigma_nSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(computeDiffCov_(X1, y1, X2, y2, XT, theta, sigma_f, sigma_n, beta));
    return rcpp_result_gen;
END_RCPP
}
// computeConfBand_
arma::vec computeConfBand_(const arma::mat& diffCovMat, double confLevel);
RcppExport SEXP _DSWE_computeConfBand_(SEXP diffCovMatSEXP, SEXP confLevelSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type diffCovMat(diffCovMatSEXP);
    Rcpp::traits::input_parameter< double >::type confLevel(confLevelSEXP);
    rcpp_result_gen = Rcpp::wrap(computeConfBand_(diffCovMat, confLevel));
    return rcpp_result_gen;
END_RCPP
}
// matchcov
arma::vec matchcov(arma::mat& ref, arma::mat& obj, arma::rowvec& thres, arma::rowvec& circ_pos, int flag);
RcppExport SEXP _DSWE_matchcov(SEXP refSEXP, SEXP objSEXP, SEXP thresSEXP, SEXP circ_posSEXP, SEXP flagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat& >::type ref(refSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type obj(objSEXP);
    Rcpp::traits::input_parameter< arma::rowvec& >::type thres(thresSEXP);
    Rcpp::traits::input_parameter< arma::rowvec& >::type circ_pos(circ_posSEXP);
    Rcpp::traits::input_parameter< int >::type flag(flagSEXP);
    rcpp_result_gen = Rcpp::wrap(matchcov(ref, obj, thres, circ_pos, flag));
    return rcpp_result_gen;
END_RCPP
}
// matern15_scaledim
arma::mat matern15_scaledim(arma::vec covparms, arma::mat locs);
RcppExport SEXP _DSWE_matern15_scaledim(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(matern15_scaledim(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// d_matern15_scaledim
arma::cube d_matern15_scaledim(arma::vec covparms, arma::mat locs);
RcppExport SEXP _DSWE_d_matern15_scaledim(SEXP covparmsSEXP, SEXP locsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type locs(locsSEXP);
    rcpp_result_gen = Rcpp::wrap(d_matern15_scaledim(covparms, locs));
    return rcpp_result_gen;
END_RCPP
}
// vecchia_grouped_profbeta_loglik_grad_info
List vecchia_grouped_profbeta_loglik_grad_info(NumericVector covparms, StringVector covfun_name, NumericVector y, NumericMatrix X, const NumericMatrix locs, List NNlist);
RcppExport SEXP _DSWE_vecchia_grouped_profbeta_loglik_grad_info(SEXP covparmsSEXP, SEXP covfun_nameSEXP, SEXP ySEXP, SEXP XSEXP, SEXP locsSEXP, SEXP NNlistSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type covparms(covparmsSEXP);
    Rcpp::traits::input_parameter< StringVector >::type covfun_name(covfun_nameSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type locs(locsSEXP);
    Rcpp::traits::input_parameter< List >::type NNlist(NNlistSEXP);
    rcpp_result_gen = Rcpp::wrap(vecchia_grouped_profbeta_loglik_grad_info(covparms, covfun_name, y, X, locs, NNlist));
    return rcpp_result_gen;
END_RCPP
}
// MaxMincpp
IntegerVector MaxMincpp(NumericMatrix locations);
RcppExport SEXP _DSWE_MaxMincpp(SEXP locationsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type locations(locationsSEXP);
    rcpp_result_gen = Rcpp::wrap(MaxMincpp(locations));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_DSWE_computeWeightedY", (DL_FUNC) &_DSWE_computeWeightedY, 3},
    {"_DSWE_predictGP", (DL_FUNC) &_DSWE_predictGP, 4},
    {"_DSWE_computeLogLikGP_", (DL_FUNC) &_DSWE_computeLogLikGP_, 3},
    {"_DSWE_computeLogLikGradGP_", (DL_FUNC) &_DSWE_computeLogLikGradGP_, 3},
    {"_DSWE_computeLogLikGradGPZeroMean_", (DL_FUNC) &_DSWE_computeLogLikGradGPZeroMean_, 3},
    {"_DSWE_computeDiffCov_", (DL_FUNC) &_DSWE_computeDiffCov_, 9},
    {"_DSWE_computeConfBand_", (DL_FUNC) &_DSWE_computeConfBand_, 2},
    {"_DSWE_matchcov", (DL_FUNC) &_DSWE_matchcov, 5},
    {"_DSWE_matern15_scaledim", (DL_FUNC) &_DSWE_matern15_scaledim, 2},
    {"_DSWE_d_matern15_scaledim", (DL_FUNC) &_DSWE_d_matern15_scaledim, 2},
    {"_DSWE_vecchia_grouped_profbeta_loglik_grad_info", (DL_FUNC) &_DSWE_vecchia_grouped_profbeta_loglik_grad_info, 6},
    {"_DSWE_MaxMincpp", (DL_FUNC) &_DSWE_MaxMincpp, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_DSWE(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
