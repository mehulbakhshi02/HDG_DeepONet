#include "../helper_blas.h"
/** @file
 * @brief We take in two metrics and intersect them
 * @param[in] - metric_1 - Contains the first metric. Size = (D*(D+1)/2 double
 * @param[in] - metric_2 - Contains the second metric. Size = Dx(D+1)/2 double
 * @param[out] - metric_int - Contains the intersected metric. Size = Dx(D+1)/2 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::IntersectTwoMetric(const vector<double> & metric_1, const vector<double> & metric_2, vector<double> & metric_int) {

    double M1[D*D]={0.0};// We just need to do this convertion because dsyev takes only array pointers
    double M2[D*D]={0.0};// THe second metric stroed as required by the code
    // Here we store it as row major
    int index = 0;
    for(int j = 0; j<D; ++j)
    {
      for(int i = 0; i <= j; ++i)
      {
        M1[i*D+j] = metric_1[index];
        M1[j*D+i] = metric_1[index];
        M2[i*D+j] = metric_2[index];
        M2[j*D+i] = metric_2[index];
        index++;
      }
    }
     // Some parameters form dsyev blas routine
    char jobz = 'V', uplo = 'L';
    int size = D;
    double D_vec[D]={0.0};
    double wkopt;
    int lwork = -1;// Not sure what this does
    int info;
    // Query size and allocate optimal workspace
    // eig contains the eigenvalues in ascending order and corresponding eigen vectors
    dsyev_(&jobz, &uplo, &size, M1, &size, D_vec, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    double* work = (double*)malloc( lwork*sizeof(double) );
    // Solve the eigen value problem
    // This rotation matrix is mostly in increasing order of eigen values and also a Column major vector
    // Because M1 is symmetric it doesnt matter if we store it as column or row major
    dsyev_(&jobz, &uplo, &size, M1, &size, D_vec, work, &lwork, &info);
    // Here we have M1 which is the rotation matrix after eigen value decomposition and D_vec which are eigen vectors
    double D1[D*D] = {0.0};// The matrix containg the eigen values of the first metric
    for(int i=0;i<D;++i)
    {
    	D1[i*D+i] = D_vec[i];
    }
    // Parameters for dgemm blas routine
    double M2p_temp[D*D]={0.0};// Used to store the temporary matrix product
    double M2p[D*D]={0.0};// Used to store the temporary matrix product
    double alpha = 1.0, beta = 0.0;
    char trans1 = 'T', trans2 = 'N';
    // The R is column major as if comes out of a Blas routine.lambda_met is diagonal
    // Compute the rotation of M2p_temp = M2*R1(here M1). Because we need the transpose we have trans2='T'
    dgemm_(&trans2, &trans2, &size, &size, &size, &alpha, M2, &size, M1, &size, &beta, M2p_temp, &size);
    // M2p_temp is column major and M1 is also column major
    // Compute the rotation of M2p = R1^T*M2p_temp (M2p_temp = M2*R1(here M1))
    dgemm_(&trans1, &trans2, &size, &size, &size, &alpha, M1, &size, M2p_temp, &size, &beta, M2p, &size);
    // M2p is column major but it is symmetric

	vector<int> ipiv(D);
    lwork = size * size;
    double D1_inv[D*D] = {0.0};
    for(int i=0;i<D;++i)
    {

        for(int j=0;j<D;++j)
        {
            D1_inv[i*D+j] = D1[i*D+j];
        }
    }
    dgetrf_(&size, &size, D1_inv, &size, &ipiv[0], &info);
    // Compute the  inverse of the diagonal matrix D1
    dgetri_(&size, D1_inv, &size, &ipiv[0], &work[0], &lwork, &info);
    // Compute N=D1^-1*M2p

    double N[D*D]={0.0};// Used to store the N matrix which is N=D1^{-1}*M2p
    dgemm_(&trans2, &trans2, &size, &size, &size, &alpha, D1_inv, &size, M2p, &size, &beta, N, &size);

    lwork = -1;
    char jobvl = 'N', jobvr = 'V';
    double D_vec_im[size], vl[size*size], P[size*size];// D_vec_im should contain only zeros and so should vl as we do not calculate them
	dgeev_(&jobvl, &jobvr, &size, N, &size, D_vec, D_vec_im, vl, &size, P, &size, &wkopt, &lwork, &info );
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
	dgeev_(&jobvl, &jobvr, &size, N, &size, D_vec, D_vec_im, vl, &size, P, &size, work, &lwork, &info);

    // // Query size and allocate optimal workspace
    // // Compute the eigen value decomposition of the matrix N

    // Now N contains the eigen vectors while D_vec contains the eigen values of N(called P in the paper)
    double L[D*D]={0.0};// Used to store the temporary matrix product
    double L_temp[D*D]={0.0};// Used to store the temporary matrix product
    // Compute the rotation of L_temp = D1*P
    dgemm_(&trans2, &trans2, &size, &size, &size, &alpha, D1, &size, P, &size, &beta, L_temp, &size);
    // L_temp is column major and N is also column major
    // Compute the rotation of L = P^{T}*L_temp (L_temp = D1*P)
    dgemm_(&trans1, &trans2, &size, &size, &size, &alpha, P, &size, L_temp, &size, &beta, L, &size);


    double U[D*D]={0.0};// Used to store the temporary matrix product
    double U_temp[D*D]={0.0};// Used to store the temporary matrix product
    // Compute the rotation of U_temp = M2p*P
    dgemm_(&trans2, &trans2, &size, &size, &size, &alpha, M2p, &size, P, &size, &beta, U_temp, &size);
    // U_temp is column major and N is also column major
    // Compute the rotation of U = P^{T}*U_temp (U_temp = M2p*P)
    dgemm_(&trans1, &trans2, &size, &size, &size, &alpha, P, &size, U_temp, &size, &beta, U, &size);
    double D_fin[D*D] = {0.0};

   //  Here is a hack to fix the case when the metrics are exactly the same
   //  The matrix D in that case is not diagonal
    for(int i=0;i<D;++i)
    {
        for(int j=0;j<D;++j)
        {
            D_fin[i*D+j] = max(L[i*D+j], U[i*D+j]);
        }
    }
	double P_inv[D*D] = {0.0};
    for(int i=0;i<D;++i)
    {
        for(int j=0;j<D;++j)
        {
            P_inv[i*D+j] = P[i*D+j];
        }
    }
    dgetrf_(&size, &size, P_inv, &size, &ipiv[0], &info);
    // Compute the  inverse of the diagonal matrix D1
    dgetri_(&size, P_inv, &size, &ipiv[0], &work[0], &lwork, &info);

    double MIp[D*D]={0.0};// Used to store the temporary matrix product
    double MIp_temp[D*D]={0.0};// Used to store the temporary matrix product
    // Compute the rotation of MIp_temp = D_fin*P^-1
    dgemm_(&trans2, &trans2, &size, &size, &size, &alpha, D_fin, &size, P_inv, &size, &beta, MIp_temp, &size);
    // Compute the product of MIp = P^{-T}*MIp_temp (MIp_temp = D_fin*P^-1)
    dgemm_(&trans1, &trans2, &size, &size, &size, &alpha, P_inv, &size, MIp_temp, &size, &beta, MIp, &size);

    double MI[D*D]={0.0};// Used to store the temporary matrix product
    double MI_temp[D*D]={0.0};// Used to store the temporary matrix product
    // Compute the rotation of MI_temp = MIp*R1^T(here called M1)
    dgemm_(&trans2, &trans1, &size, &size, &size, &alpha, MIp, &size, M1, &size, &beta, MI_temp, &size);
    // Compute the product of MIp = R1*MI_temp (MI_temp = MIp*R1^T(here called M1))
    dgemm_(&trans2, &trans2, &size, &size, &size, &alpha, M1, &size, MI_temp, &size, &beta, MI, &size);

    // index = 0;
    // for(int i = 0; i<D; ++i)
    // {
    //   for(int j = i; j<D; ++j)
    //   {
    //     metric_int[index] = MI[i*D+j];
    //     index++;
    //   }
    // }
    index = 0;
    for(int j = 0; j<D; ++j)
    {
      for(int i = 0; i <= j; ++i)
      {
        metric_int[index] = MI[i*D+j];
        index++;
      }
    }
}
