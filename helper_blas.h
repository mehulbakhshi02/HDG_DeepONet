#ifndef HELPERBLAS_H
#define HELPERBLAS_H

// Commented as it is defined in NGSolve 6
// extern "C" {
// /* Declaration of routines from BLAS and LAPACK */
//   extern void dgetrf_( int *m, int *n,
//                        double *A, int *lda, int *ipiv,
//                        int *info );

//   extern void dgetri_( int *m,
//                        double *A, int *lda, int *ipiv,
//                        double *workspace, int * lwork, int *info );
  
//   extern void dgetrs_( char *trans, int *n, int *nrhs,
//                        double *A, int *lda, int *ipiv,
//                        double *B, int *ldb,
//                        int *info );
		    
//   extern void dgesv_( int *n, int *nrhs,
//                       double *A, int *lda, int *ipiv,
//                       double *B, int *ldb,
//                       int *info );

//   extern void dgels_( char *trans, int *m, int *n, int *nrhs,
//                       double *A, int *lda, 
//                       double *B, int *ldb,
//                       double *work, int *lwork,
//                       int *info );  
		    		    
//   extern void dgemv_( char *trans, int *m, int *n,
//                       double *alpha, double *a, int *lda, double *x,
//                       int *incx, double *beta, double *y, int *incy);		    

//   extern double ddot_( int *n, double *dx, int *incx, double *dy, int *incy);
  
//   extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k, 
//                      double *alpha, double *a, int *lda, double *b, int *ldb, 
//                      double *beta, double *c, int *ldc);

//   extern void daxpy_(int *n, double *da, double *dx, int *incx, double *dy, int *incy);

//   extern double dlange_(char *norm, int *m, int *n, double *a, int *lda, double *work);

//   extern void dsyev_(char *jobz, char *uplo, int *n, double *a, int *lda, double *w,
//                      double *work, int *lwork, int *info);

//   extern void dgeev_( char* jobvl, char* jobvr, int* n, double* a,
//                 int* lda, double* wr, double* wi, double* vl, int* ldvl,
//                 double* vr, int* ldvr, double* work, int* lwork, int* info );
// }

// Wrapper for the standard gemv
void mygemv(char transa, int m, int n, double alpha, double * A, double * x, double beta, double * y) {
  int lda = m;
  int ione = 1;

  dgemv_(&transa, &m, &n, &alpha, A, &lda, x, &ione, &beta, y, &ione);
}


// Wrapper for the standard (column-oriented) gemm
void mygemm(char transa, char transb, int m, int n, int k, double alpha, double * A, double * B, double beta, double * C) {
  int lda, ldb, ldc;  

  if (transa == 'n')
    lda = m;
  else
    lda = k;
  if (transb == 'n')
    ldb = k;
  else
    ldb = n;
  
  ldc = m;

  dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

//#define DGEMM_ROWMAJOR(A,B,C,m,n,k,alpha,beta,transf_A,transf_B, lda, ldb, ldc) 
//                 DGEMM(transf_B, transf_A, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)

// Wrapper for the row-oriented gemm
void mygemm_row(char transa, char transb, int m, int n, int k, double alpha, double * A, double * B, double beta, double * C) {
  int lda, ldb, ldc;

  if (transa == 'n') {
    lda = k;
  } else {
    lda = m;
  }
  if (transb == 'n') {
    ldb = n;
  } else {
    ldb = k;
  }
  ldc = n;
  
  dgemm_(&transb, &transa, &n, &m, &k, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

#endif
