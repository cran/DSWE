#The code has been derived from https://github.com/cran/GpGp/tree/master.
#We did not import the package directly as a dependency because we applied some minor changes to use
#the functions in our algorithm.
#The following is the copyright notice for the GpGp package. 
#
# MIT License
#
# Copyright (c) 2018 Joseph Guinness
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#' @useDynLib DSWE, .registration = TRUE


get_linkfun <- function(covfun_name){
  
  link <- exp
  dlink <- exp
  invlink <- log
  lonlat <- FALSE
  space_time <- FALSE
  
  return(list(
    link = link, dlink = dlink, invlink = invlink,
    lonlat = lonlat, space_time = space_time
  ))
}

find_ordered_nn <- function(locs,m, st_scale = NULL){
  
  # if locs is a vector, convert to matrix
  if( is.null(ncol(locs)) ){
    locs <- as.matrix(locs)
  }
  
  # number of locations
  n <- nrow(locs)
  m <- min(m,n-1)
  mult <- 2
  
  # FNN::get.knnx has strange behavior for exact matches
  # so add a small amount of noise to each location
  ee <- min(apply( locs, 2, stats::sd ))
  locs <- locs + matrix( ee*1e-4*stats::rnorm(n*ncol(locs)), n, ncol(locs) )    
  
  
  if( !is.null(st_scale) ){ 
    d <- ncol(locs)-1
    locs[ , 1:d] <- locs[ , 1:d]/st_scale[1]
    locs[ , d+1] <- locs[ , d+1]/st_scale[2]
  }
  
  # to store the nearest neighbor indices
  NNarray <- matrix(NA,n,m+1)
  
  # to the first mult*m+1 by brutce force
  maxval <- min( mult*m + 1, n )
  NNarray[1:maxval,] <- find_ordered_nn_brute(locs[1:maxval,,drop=FALSE],m)
  
  query_inds <- min( maxval+1, n):n
  data_inds <- 1:n
  
  msearch <- m
  
  while( length(query_inds) > 0 ){
    msearch <- min( max(query_inds), 2*msearch )
    data_inds <- 1:min( max(query_inds), n )
    NN <- FNN::get.knnx( locs[data_inds,,drop=FALSE], locs[query_inds,,drop=FALSE], msearch )$nn.index
    less_than_k <- t(sapply( 1:nrow(NN), function(k) NN[k,] <= query_inds[k]  ))
    sum_less_than_k <- apply(less_than_k,1,sum)
    ind_less_than_k <- which(sum_less_than_k >= m+1)
    
    NN_m <- t(sapply(ind_less_than_k,function(k) NN[k,][less_than_k[k,]][1:(m+1)] ))
    
    NNarray[ query_inds[ind_less_than_k], ] <- NN_m
    
    query_inds <- query_inds[-ind_less_than_k]
    
  }
  
  return(NNarray)
}

get_penalty <- function(y,X,locs,covfun_name){
  
  fitlm <- stats::lm(y ~ X - 1 )
  vv <- summary(fitlm)$sigma^2
  # by default, no penalty
  pen <- function(x) 0.0
  dpen <- function(x) rep(0,length(x))
  ddpen <- function(x) matrix(0,length(x),length(x))
  # nugget penalty
  pen_nug <- function(x,j){ pen_loglo(x[j],.1,log(0.001)) }
  dpen_nug <- function(x,j){
    dpen <- rep(0,length(x))
    dpen[j] <- dpen_loglo(x[j],.1,log(0.001))
    return(dpen)
  }
  ddpen_nug <- function(x,j){
    ddpen <- matrix(0,length(x),length(x))
    ddpen[j,j] <- ddpen_loglo(x[j],.1,log(0.001))
    return(ddpen)
  }

  # dangerous because vv could get redefined
  pen_var <- function(x,j){ pen_hi(x[j]/vv,1,6) }
  dpen_var <- function(x,j){
    dpen <- rep(0,length(x))
    dpen[j] <- 1/vv*dpen_hi(x[j]/vv,1,6)
    return(dpen)
  }
  ddpen_var <- function(x,j){
    ddpen <- matrix(0,length(x),length(x))
    ddpen[j,j] <- 1/vv^2*ddpen_hi(x[j]/vv,1,6)
    return(ddpen)
  }
  ###for matern15_scaledim
    d <- ncol(locs)
    pen <- function(x){  pen_nug(x,d+2)  +   pen_var(x,1)   }
    dpen <- function(x){ dpen_nug(x,d+2)  +  dpen_var(x,1)  }
    ddpen <- function(x){ ddpen_nug(x,d+2) + ddpen_var(x,1) }
  
  return( list( pen = pen, dpen = dpen, ddpen = ddpen ) )
}

group_obs <- function(NNarray, exponent = 2){
  n <- nrow(NNarray)
  m <- ncol(NNarray)-1
  
  clust <- vector("list",n)
  for(j in 1:n) clust[[j]] <- j
  for( ell in 2:(m+1) ){  # 2:(m+1)?
    sv <- which( NNarray[,1] - NNarray[,ell] < n )
    for(j in sv){
      k <- NNarray[j,ell]
      if( length(clust[[k]]) > 0){
        nunique <- length(unique(c(NNarray[c(clust[[j]],clust[[k]]),])))
        
        # this is the rule for deciding whether two groups
        # should be combined
        if( nunique^exponent <= length(unique(c(NNarray[clust[[j]],])))^exponent + length(unique(c(NNarray[clust[[k]],])))^exponent ) {
          clust[[j]] <- c(clust[[j]],clust[[k]])
          clust[[k]] <- numeric(0)
        }
      }
    }
  }
  zeroinds <- unlist(lapply(clust,length)==0)
  clust[zeroinds] <- NULL
  nb <- length(clust)
  all_inds <- rep(NA,n*(m+1))
  last_ind_of_block <- rep(NA,length(clust))
  last_resp_of_block <- rep(NA,length(clust))
  #num_resp_inblock <- rep(NA,length(clust))
  local_resp_inds <- rep(NA,n)
  global_resp_inds <- rep(NA,n)
  
  last_ind <- 0
  last_resp <- 0
  for(j in 1:nb){
    resp <- sort(clust[[j]])
    last_resp_of_block[j] <- last_resp + length(resp)
    global_resp_inds[(last_resp+1):last_resp_of_block[j]] <- resp
    
    inds_inblock <- sort( unique( c(NNarray[resp,]) ) )
    last_ind_of_block[j] <- last_ind + length(inds_inblock)
    all_inds[(last_ind+1):last_ind_of_block[j]] <- inds_inblock
    last_ind <- last_ind_of_block[j]
    
    local_resp_inds[(last_resp+1):last_resp_of_block[j]] <- which( inds_inblock %in% resp )
    last_resp <- last_resp_of_block[j]
  }
  all_inds <- all_inds[ !is.na(all_inds) ]
  NNlist <- list( all_inds = all_inds, last_ind_of_block = last_ind_of_block,
                  global_resp_inds = global_resp_inds, local_resp_inds = local_resp_inds,
                  last_resp_of_block = last_resp_of_block )
  return(NNlist)
}
calculatePairwiseDistances <- function(set1, set2) {
  n1 <- nrow(set1)
  n2 <- nrow(set2)
  distances <- matrix(0, n1, n2)
  for (i in 1:n1) {
    for (j in 1:n2) {
      
      distance <- sqrt(sum((set1[i,] - set2[j,])^2))
      distances[i,j] <- distance
    }
  }
  return(distances)
}
find_ordered_nn_brute <- function( locs, m ){

  n <- dim(locs)[1]
  m <- min(m,n-1)
  NNarray <- matrix(NA,n,m+1)
  for(j in 1:n ){
    distvec <- c(calculatePairwiseDistances(locs[1:j,,drop=FALSE],locs[j,,drop=FALSE]) )
    NNarray[j,1:min(m+1,j)] <- order(distvec)[1:min(m+1,j)]
  }
  return(NNarray)
}

test_likelihood_object <- function(likobj){
  
  pass <- TRUE
  allvals <- c( likobj$loglik, likobj$grad, c(likobj$info) )
  if( sum(is.na(allvals)) > 0  ||  sum( abs(allvals) == Inf ) > 0 ){
    pass <- FALSE
  }
  return(pass)
}

condition_number <- function(info){
  # assumes that information matrix has finite numbers in it
  if(max(diag(info))/min(diag(info)) > 1e6){
    return( max(diag(info))/min(diag(info)) )
  } else {
    ee <- eigen(info)
    return( max(ee$values)/min(ee$values) )
  }
}    

fisher_scoring <- function( likfun, start_parms, link, 
                            silent = FALSE, convtol = 1e-4, max_iter = 40 ){
  
  # function for checking wolfe conditions
  wolfe_check <- function(likobj0,likobj1,logparms,step,both){
    c1 <- 1e-4
    c2 <- 0.9
    tol <- 0.1
    ll0 <- likobj0$loglik
    gr0 <- likobj0$grad
    ll1 <- likobj1$loglik
    gr1 <- likobj1$grad
    if(!both){
      satfd <- ll1 <= ll0 + c1*crossprod(step,gr0) + tol
    } else {
      satfd <- ll1 <= ll0 + c1*crossprod(step,gr0) + tol &&
        -crossprod(step,gr1) <= -c2*crossprod(step,gr0) + tol
    }
    return(satfd)
  }
  
  # evaluate function at initial values
  logparms <- start_parms
  likobj <- likfun(logparms)
  
  # test likelihood object    
  if( !test_likelihood_object(likobj) ){
    logparms <- 0.1*logparms
    likobj <- likfun(logparms)
  }
  
  # assign loglik, grad, and info
  loglik <- likobj$loglik        
  grad <- likobj$grad
  info <- likobj$info
  
  # add a small amount of regularization
  diag(info) <- diag(info) + 0.1*min(diag(info))
  
  # print some stuff out
  if(!silent){
    cat(paste0("Iter ",0,": \n"))
    cat("pars = ",  paste0(round(link(logparms),4)), "  \n" )
    cat(paste0("loglik = ", round(-loglik,6),         "  \n"))
    cat("grad = ")
    cat(as.character(round(-grad,3)))
    cat("\n\n")
  }
  
  for(j in 1:max_iter){
    
    likobj0 <- likobj
    
    # if condition number of info matrix large, 
    # then gradient descent
    tol <- 1e-10
    if (condition_number(info) > 1 / tol) {
      if (!silent) cat("Cond # of info matrix > 1/tol \n")
      #info <- 1.0*max(likobj0$info)*diag(nrow(likobj0$info))
      # regularize
      ee <- eigen(info)
      ee_ratios <- ee$values/max(ee$values)
      ee_ratios[ ee_ratios < 1e-5 ] <- 1e-5
      ee$values <- max(ee$values)*ee_ratios
      info <- ee$vectors %*% diag(ee$values) %*% t(ee$vectors)
      
      #diag(info) <- diag(info) + tol*max(diag(info))
    }
    
    # calculate fisher step 
    step <- -solve(info, grad)
    
    # if step size large, then make it smaller
    if (mean(step^2) > 1) {
      if(!silent) cat("##\n")
      step <- step/sqrt(mean(step^2))
    }
    
    # take step and calculate loglik, grad, and info
    newlogparms <- logparms + step
    likobj <- likfun(newlogparms)
    
    # check for Inf, NA, or NaN
    cnt <- 1
    while (!test_likelihood_object(likobj)) {
      if (!silent) cat("inf or na or nan in likobj\n")
      step <- 0.5 * step
      newlogparms <- logparms + step
      likobj <- likfun(newlogparms)
      if (cnt == 10) { stop("could not avoid inf, na or nan\n") }
    }
    
    # check whether negative likelihood decreased
    # take smaller stepsize
    if( likobj$loglik > likobj0$loglik ){
      step <- 0.25*step
      newlogparms <- logparms + step
      likobj <- likfun(newlogparms)
    }
    
    # check again, move along gradient
    if( likobj$loglik > likobj0$loglik ){
      info0 <- diag( rep(mean(diag(info)),nrow(info)) )
      step <- -solve(info0,grad)
      newlogparms <- logparms + step
      likobj <- likfun(newlogparms)
    }
    
    # check once move, take smaller step along gradient
    if( likobj$loglik > likobj0$loglik ){
      info0 <- diag( rep(max(diag(info)),nrow(info)) )
      step <- -solve(info0,grad)
      newlogparms <- logparms + step
      likobj <- likfun(newlogparms)
    }
    
    # Check the wolfe conditions
    # if not satisfied, shrink fisher step
    # after some iterations, switch to gradient
    cnt <- 1
    no_decrease <- FALSE
    both <- FALSE
    mult <- 1.0
    #if(!wolfe_check(likobj0,likobj,logparms,newlogparms-logparms,both) &&
    #        !no_decrease ){
    #    cat("@@\n")
    #    step <- -0.5*sqrt(mean(step^2))*grad/sqrt(sum(grad^2))
    #}
    #while (!wolfe_check(likobj0,likobj,logparms,newlogparms-logparms,both) &&
    #        !no_decrease ){
    #    info <- 1/mult*max(likobj$info)*diag(nrow(likobj0$info))
    #    step <- -solve(info,grad)
    #    if(!silent) cat("**\n") 
    #    if ( sqrt(sum(step^2)) < 1e-4 ){ no_decrease <- TRUE }  # maybe we should throw error here?
    #    newlogparms <- logparms + step
    #    likobj <- likfun(newlogparms)
    #    cnt <- cnt + 1
    #    mult <- mult*0.5
    #}
    stepgrad <- c(crossprod(step,grad))
    
    # redefine logparms, loglik, grad, info
    logparms <- logparms + step
    loglik <- likobj$loglik        
    grad <- likobj$grad
    info <- likobj$info
    
    # print some stuff out
    if(!silent){
      cat(paste0("Iter ",j,": \n"))
      cat("pars = ",  paste0(round(link(logparms),4)), "  \n" )
      cat(paste0("loglik = ", round(-loglik,6),         "  \n"))
      cat("grad = ")
      cat(as.character(round(grad,4)),"\n")
      cat("step dot grad = ",stepgrad,"\n")
      cat("\n")
    }
    
    # if gradient is small, then break and return the results        
    if( abs(stepgrad) < convtol || no_decrease ){
      break
    }
  }
  
  # collect information to return
  betahatinfo <- likobj        
  betahat <- as.vector(betahatinfo$betahat)
  betacov <- solve(betahatinfo$betainfo)
  sebeta <- sqrt(diag(betacov))
  tbeta <- betahat/sebeta
  
  ret <- list(
    covparms = link(logparms), 
    logparms = logparms,
    betahat = betahat, 
    sebeta = sebeta,
    betacov = betacov,
    tbeta = tbeta,
    loglik = loglik,
    no_decrease = no_decrease,
    grad = likobj$grad,
    info = likobj$info,
    conv = ( abs(stepgrad) < convtol || no_decrease )
  )
  return(ret)
}

order_maxmin_exact<-function(locs){
  ord<-MaxMincpp(locs)
  return(ord)
}

expit <- function(x){ exp(x)/(1+exp(x)) }

intexpit <- function(x){ log(1+exp(x)) }

pen_hi <- function(x,tt,aa){ -tt*intexpit(x-aa) }

dpen_hi <- function(x,tt,aa){ -tt*expit(x-aa) }

ddpen_hi <- function(x,tt,aa){ -tt*expit(x-aa)/(1+exp(x-aa)) }

pen_lo <- function(x,tt,aa){ -tt*intexpit(-x+aa) }

dpen_lo <- function(x,tt,aa){ +tt*expit(-x+aa) }

ddpen_lo <- function(x,tt,aa){ -tt*expit(-x+aa)/(1+exp(-x+aa)) }

pen_loglo <- function(x,tt,aa){ 
  if(x==0){ return(0.0) 
  } else { 
    return( pen_lo(log(x),tt,aa) )
  }
}

dpen_loglo <- function(x,tt,aa){ 
  if( x==0 ){
    return(0.0) 
  } else {
    return( dpen_lo(log(x),tt,aa)/x )
  }
}

ddpen_loglo <- function(x,tt,aa){ 
  if( x==0 ){
    return( 0.0 )
  } else {
    return( ddpen_lo(log(x),tt,aa)/x^2 - dpen_lo(log(x),tt,aa)/x^2 )
  }
}
