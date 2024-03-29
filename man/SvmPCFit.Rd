% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SvmPCFit.R
\name{SvmPCFit}
\alias{SvmPCFit}
\title{SVM based power curve modelling}
\usage{
SvmPCFit(trainX, trainY, testX, kernel = "radial")
}
\arguments{
\item{trainX}{a matrix or dataframe to be used in modelling}

\item{trainY}{a numeric or vector as a target}

\item{testX}{a matrix or dataframe, to be used in computing the predictions}

\item{kernel}{default is 'radial' else can be 'linear', 'polynomial' and 'sigmoid'}
}
\value{
a vector or numeric predictions on user provided test data
}
\description{
SVM based power curve modelling
}
\examples{

data = data1
trainX = as.matrix(data[c(1:100),2])
trainY = data[c(1:100),7]
testX = as.matrix(data[c(101:110),2])

Svm_prediction = SvmPCFit(trainX, trainY, testX)

}
