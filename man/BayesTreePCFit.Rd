% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/BayesTreePCFit.R
\name{BayesTreePCFit}
\alias{BayesTreePCFit}
\title{Tree based power curve estimate}
\usage{
BayesTreePCFit(trainX, trainY, testX, nTree = 50)
}
\arguments{
\item{trainX}{a matrix or dataframe to be used in modelling}

\item{trainY}{a numeric or vector as a target}

\item{testX}{a matrix or dataframe, to be used in computing the predictions}

\item{nTree}{a numeric value specifying number of trees to be constructed in model}
}
\value{
a vector or numeric predictions on user provided test data
}
\description{
Tree based power curve estimate
}
\examples{

data = data1
trainX = as.matrix(data[c(1:100),2])
trainY = data[c(1:100),7]
testX = as.matrix(data[c(100:110),2])

Bart_prediction = BayesTreePCFit(trainX, trainY, testX)

}
