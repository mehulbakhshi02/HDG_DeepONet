#include "../helper_blas.h"
template <int D>
class MeshMetric {

  // metric components
/*!
 * The metric is defined as a vector of vector of doubles whose size is decided based on the number of elements and
 * dimension of the problem.
 */
  vector<double> Metric;
public:  
// /*!
//  * The rotation matrix is computed using the dimension information also using the euler angles
//  * to the rotation matrix computation. The rule used is Rz*Ry*Rx. The angles are stored in that
//  * format. We have to separate the values for different dimensions because we need to compute
//  * differente sized matrices
//  */
// // 
//   void ComputeRotationMatrix(const vector<double> & theta, double * R)
//   {
//     if(D==2)
//     {
//       R[0] = cos(theta[0]);
//       R[1] = sin(theta[0]);
//       R[2] = -sin(theta[0]);
//       R[3] = cos(theta[0]);
//     }
//     else if(D==3)
//     {

//       double Rx[D*D]={0.0}, Ry[D*D]={0.0}, Rz[D*D]={0.0};
//       // Theta is stored in the order z y x
//       // Initialize the rotation wrt x axis
//       Rx[0] = 1.0;
//       Rx[4] = cos(theta[2]);
//       Rx[5] = sin(theta[2]);
//       Rx[7] = -sin(theta[2]);
//       Rx[8] = cos(theta[2]);
//       // Initialize rotation wrt y axis
//       Ry[0] = cos(theta[1]);
//       Ry[2] = -sin(theta[1]);
//       Ry[4] = 1.0;
//       Ry[6] = sin(theta[1]);
//       Ry[8] = cos(theta[1]);
//       // Initialize rotation wrt z axis
//       Rz[0] = cos(theta[0]);
//       Rz[1] = sin(theta[0]);
//       Rz[3] = -sin(theta[0]);
//       Rz[4] = cos(theta[0]);
//       Rz[8] = 1.0;

//       // Multiplying the rotation matrices to compute the final rotation matrix
//       // Values used in degmm blas routine
//       double Rot_temp[D*D]={0.0};
//       int size = D;
//       double alpha = 1.0, beta = 0.0;
//       char trans = 'N';
      
//       // Compute the rotation of Rot_temp = Ry*Rx
//       dgemm_(&trans, &trans, &size, &size, &size, &alpha, Ry, &size, Rx, &size, &beta, Rot_temp, &size);
//       // Compute the rotation of R = Rz*Rot_temp(Rot_temp = Ry*Rx)
//       dgemm_(&trans, &trans, &size, &size, &size, &alpha, Rz, &size, Rot_temp, &size, &beta, R, &size);
//       // Finally computed R is column major

//     }
//   }

  /*!
   * The metric components are computed using the lambda and theta values. This is to be written
   * generally for any dimension
 */

  void ComputeMetric(const vector<double> & lambda, const vector<double> & q_met, vector<double> & metric_loc)
  {
    double lambda_met[D*D]={0.0};
    // Diagonal matrix so it doesnt matter if it is row or column major
    for(int dd = 0; dd<D; ++dd)
    {
      int index = dd*D + dd;// Sets the diagonals to the value of lambda matrix
      lambda_met[index] = lambda[dd];
    }
    // Compute the rotation matrix from the vector
    double R[D*D]={0.0};
    int index = 0;
    for(int ll = 0; ll < D; ++ll)
    {
      for(int kk = 0; kk < D; ++kk) 
      {         
        R[index] = q_met[index];
        index++;
      }
    }
    // ComputeRotationMatrix(theta, R);
    // The rotation matrix is Column major


    // The full symmetric matrix
    double M_full[D*D]={0.0};
    // Parameters for dgemm blas routine
    double M_temp[D*D]={0.0};// Used to store the temporary matrix product
    int size = D;
    double alpha = 1.0, beta = 0.0;
    char trans1 = 'N', trans2 = 'T';
    // The R is column major as if comes out of a Blas routine.lambda_met is diagonal
    // Compute the rotation of M_temp = lambda_met*R^T. Because we need the transpose we have trans2='T'
    dgemm_(&trans1, &trans2, &size, &size, &size, &alpha, lambda_met, &size, R, &size, &beta, M_temp, &size);
    // M_temp is column major and R is also column major
    // Compute the rotation of M_full = R*M_temp (M_temp = R*lambda_met)
    dgemm_(&trans1, &trans1, &size, &size, &size, &alpha, R, &size, M_temp, &size, &beta, M_full, &size);
    // M_full is column major but it is symmetric

    // We store only half of the full metric
    // M_full is symmetric so it doesnt matter if it is column
    // or row major
    index = 0;
    for(int j = 0; j<D; ++j)
    {
      for(int i = 0; i <= j; ++i)
      {
        metric_loc[index] = M_full[i*D+j];
        index++;
      }
    }
  }
/*!
 * The metric components are used to do an eigen value decomposition of the metric. 
 * We need to then do some conversion to Euler angles
 * We need to arrange the lambda's in decreasing order and also arrange the thetas according
 * to this order
 */
  void EigenDecomposition(vector<double> & lambda, vector<double> & q_met, const vector<double> & metric_loc) const
  {
    // Must do eigen value decomposition of metric_loc and store the eigen values in ascending
    // order in
    // Compute the upper triangular matrix for blas routine dsyev to compute
    // the eigen value decomposition

    double Vec[D*D]={0.0};// We just need to do this convertion because dsyev takes only array pointers
    // Here we store it as row major
    int index = 0;
    for(int j = 0; j<D; ++j)
    {
      for(int i = 0; i <= j; ++i)
      {
        Vec[i*D+j] = metric_loc[index];
        Vec[j*D+i] = metric_loc[index];
        index++;
      }
    }
    // Some parameters form dsyev blas routine
    char jobz = 'V', uplo = 'L';
    int size = D;
    double eig[D]={0.0};
    double wkopt;
    int lwork = -1;// Not sure what this does
    int info;
    // Query size and allocate optimal workspace
    // eig contains the eigenvalues in ascending order and corresponding eigen vectors
    dsyev_(&jobz, &uplo, &size, Vec, &size, eig, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    double work[lwork];
    memset( work, 0, lwork*sizeof(double) );
    // Solve the eigen value problem
    // This rotation matrix is mostly in increasing order of eigen values and also a Column major vector
    // Because Vec is symmetric it doesnt matter if we store it as column or row major
    dsyev_(&jobz, &uplo, &size, Vec, &size, eig, work, &lwork, &info);

    index = 0;
    for(int ll = 0; ll < D; ++ll)
    {
      for(int kk = 0; kk < D; ++kk) 
      {         
        q_met[index] = Vec[index];
        index++;
      }
    }

    // // Compute the euler angles from the orthonormal eigen vectors
    // // Vec is column major
    // ComputeRotationAngle(theta, Vec);
    // The eigen values are stored in the lambda vector
    // The we mostly decompose a metric which is SPD which means that the eigen values have to be positive ONLY
    // if not we have some error
    for(int dd = 0; dd < D; ++dd)
    {
      if(eig[dd]<0)
      {
        cout<<"Eigen values of an SPD matrix not positive!"<<endl;
        cout<<eig[dd]<<endl;
        exit(1);
      }
      lambda[dd] = eig[dd];
    }
  }


/*!
 * Set the size size of the metric arrays is number of elements times the D*(D+1)/2
 * @param[in] n - This is the number of elements for which we want to calculate
 * the metric. This is not the size of the metric array. The metric array is of size
 * n*(D)*(D+1)/2. Ex: For a 2D system we have 3 components and for a 3D system we have 6 components
 */
  MeshMetric(int n) : Metric(n*D*(D+1)/2) {}

    /*!
   * We try to compute the angles of rotation from the rotation matrix. In case of 2D
   * we might need to use the atan function while in case of 3D we can use the euler angles
   * rotation
 */

  // void ComputeRotationAngle(vector<double> & theta, const double * R) const
  // {
  //   if(D==2)
  //   {
  //     double y = R[2];
  //     double x = R[0];
  //     theta[0] = atan2(y,x);
  //     if(theta[0]<0)
  //       theta[0] = theta[0] + M_PI;
  //   }
  //   else if(D==3)
  //   {
  //     double I_temp[D*D]={0.0};// Used to store the temporary matrix product which should be indentity
  //     double R_temp[D*D]={0.0};// Should convert const to non const because we need to pass it to check for valid rotation matrix
  //     int size = D;
  //     double alpha = 1.0, beta = 0.0;
  //     char trans1 = 'T', trans2 = 'N';
  //     for(int i=0;i<D;++i)
  //     {
  //       for(int j=0;j<D;++j)
  //         R_temp[i*D+j] = R[i*D+j];
  //     }
  //     // Compute R^T*R. This shoudl be I
  //     // This is true for both column and row major
  //     // so it doesnt matter
  //     dgemm_(&trans1, &trans2, &size, &size, &size, &alpha, R_temp, &size, R_temp, &size, &beta, I_temp, &size);
  //     // Parameters for the frobenius norm of the supposed identity matrix
  //     char norm='F';
  //     double *work;
  //     double norm_val = dlange_(&norm, &size, &size, I_temp, &size, work);
  //     // If it is not a valid rotation matrix then exit the program
  //     if(fabs(norm_val-sqrt(D))>1e-8)
  //     {
  //       cout<<"Incorrect rotation matrix!"<<endl;
  //       cout<<norm_val<<endl;
  //       exit(1);
  //     }
  //     // Has to be changed because we have R as Column
  //     // major
  //     double sy = sqrt(R[0]*R[0]+R[1]*R[1]);
  //     bool singular = sy<1e-6;
  //     if(!singular)
  //     {
  //       // j*D+i
  //       theta[0] = atan2(R[0*D+1], R[0*D+0]);//rotation about z
  //       theta[1] = atan2(-R[0*D+2], sy);//rotation about y
  //       theta[2] = atan2(R[1*D+2], R[2*D+2]);//rotation about x
  //     }
  //     else
  //     {
  //       // j*D+i
  //       theta[0] = 0.0;//rotation about z
  //       theta[1] = atan2(-R[0*D+2], sy);//rotation about y
  //       theta[2] = atan2(-R[2*D+1], R[1*D+1]);//rotation about x        
  //     }
  //     // if(theta[0]<0)
  //     //   theta[0] = theta[0] + M_PI;
  //     // if(theta[1]<0)
  //     //   theta[1] = theta[1] + M_PI;
  //     // if(theta[2]<0)
  //     //   theta[2] = theta[2] + M_PI;
  //   }
  // }
  
/*!
 * Return the value of each component of the local metric
 * @param[in] i - The index of the position where we want the metric. Size = 1 int
 * @param[out] metric_loc - The local metric of the cell which is to be computed from the global array. Size = D*(D+1)/2 double
 * We should store the metric as follows: a, b, c for 2D and a, b, c, d, e, f for 3D
 * M = [a b; b c] for 2D 
 * M = [a b c; b d e; c e f] for 3D while we only store the upper triangle
 */
  void GetComponents(const int i, vector<double> & metric_loc) const 
  {
    if(metric_loc.size()!=D*(D+1)/2)
    {
      cout<<"Incorrect size for local metric vector. Resizing..."<<endl;
      metric_loc.resize(D*(D+1)/2);
    }

    for(int t = 0; t < D * (D+1)/2; ++t)
    {
      int index = i*D*(D+1)/2 + t;
      metric_loc[t] = Metric[index];
    }
  }
    /*!
 * Set the value of each component of the global metric array
 * @param[in] i - The index of the position where we want to set the metric. Size = 1 int
 * @param[out] metric_loc - Setting the global metric array from the local array. Size = D*(D+1)/2 double
 */
  void SetComponents(const int i, const vector<double> & metric_loc) {

    if(metric_loc.size()!=D*(D+1)/2)
    {
      cout<<"Incorrect size for local metric vector"<<endl;
      exit(1);
    }
    for(int t = 0; t < D * (D+1)/2; ++t)
    {
      int index = i*D*(D+1)/2 + t;// Gives the location of the first position for the current cell i
      Metric[index] = metric_loc[t];
    }
  }
/*!
 * Get the principle components of each metric i.e, the anisotropy and size components.
 * @param[in] i - Index of the metric. Size = 1 int
 * @param[out] size - The maximum eigen value of the eigen value decomposition of the metric. Size = 1 double
 * @param[out] beta - The anisotropic ratio of maximum and minimum eigenvalues of the metric. Always greater
 * than 1. Size = D double
 * @param[out] theta - The orientation of the maximum eigen value of the metric. Size = 1 if D=2 and 3 if D=3
 */  
  void GetPrincipal(const int i, double & size, vector<double> & beta, vector<double> & q_met) const {

    vector<double> metric_loc(D*(D+1)/2,0.0);
    GetComponents(i, metric_loc);
    vector<double> lambda(D, 0.0);
    EigenDecomposition(lambda, q_met, metric_loc);// lambda and theta have to be arranged such that
    // the lambda are in increasing order
    vector<double> h_val(D, 0.0); // Contains the major and minor axis of the metric
    for(int dd = 0; dd<D; ++dd)
    {
      h_val[dd] = 1.0/sqrt(lambda[dd]); // h_i = 1.0/sqrt(lambda_i) where h1>h2>h3 so lambda1<lambda2<lambda3
    }
    size = 1.0;// Initialize the size to be 1
    for(int dd = 0; dd<D; ++dd)
    {
      size *= h_val[dd];// size = h1*h2*h3 or size = h1*h2
    }
    for(int dd = 0; dd<D-1; ++dd)
    {
      beta[dd] = h_val[0]/h_val[dd+1];// Contains beta which is h1/h2 and h1/h3 in case of 3d and h1/h2 in case of 2d
    }
  }

  /*!
 * Set the principle components of each metric using the anisotropy and size.
 * @param[in] i - Index of the metric. Size = 1 int
 * @param[in] size - The product of major and minor axes of the ellipses/ellipsoids with no scaling factor. \f$size=h_{1}h_{2}\f for 2D and $\f$size=h_{1}h_{2}h_{3}\f$ for 3D with \f$h_{1}\geq h_{2}\geq h_{3}\f$. Size = 1 double
 * @param[in] beta - The anisotropic ratio(s) relating the major axis to the first
 * and the second minor axis. \f$\beta_1=\frac{h_{1}}{h_{2}}, \beta_{2}=\frac{h_{1}}{h_{3}}\f$. Always greater than 1. \f$\beta_{i}\geq 1\f$. Size = 1 or 2 double based on dimension
 * @param[in] theta - The orientation of the major, and two minor axis. Size = 1 or 3 double depending on the dimension
 */

  void SetPrincipal(const int i, const double size, const vector<double> & beta, const vector<double> & q_met) {

    if(beta.size() != (D-1) && q_met.size() != D*D)
    {
      cout<<"Incorrect size for anisotropy parameters beta and q_met."<<endl;
      exit(1);
    }

    vector<double> h_val(D,0.0);
    h_val[0] = size;// Initialize size of major axis to size. They are stored in increasing order h1>h2>h3
    for(int dd = 0; dd<D-1 ; ++dd)
    {
      h_val[0] *= beta[dd];// Multiply size by anisotropy to get major axis ^D
    }
    h_val[0] = pow(h_val[0], 1.0/(double)D);// Compute the exact value of major axis
    for(int dd = 0; dd<D-1 ; ++dd)
    {
      h_val[dd+1] = h_val[0] / beta[dd];// Divide the major axis with aspect ratio to get major and minor axis
    }

    vector<double> lambda(D,0.0);// The lambdas are stored in decreasing order. lambda1<lambda2<lambda3
    for(int dd = 0; dd<D ; ++dd)
    {// This has to be done to have cases where the metric size is 0
      double tol = 1e-200;
      if(h_val[dd]>tol)
      {     
       lambda[dd] = 1.0/(pow(h_val[dd],2.0));// lambda_i = 1/h_i^2 so if h1 is maximum then lambda1 is minimum
      }
      else
      {     
        lambda[dd] = 0.0;
      }
    }

    vector<double> metric_loc(D*(D+1)/2,0.0);
    ComputeMetric(lambda, q_met, metric_loc);
    SetComponents(i, metric_loc);
  }

  /*!
 * Get the scaling factor for the metric based on the type of adaptation and also the best generator being used
 * The two integers required tell what type of adaptation we are running and what is the mesh generator being used. Both are set
 * in the PDE file. The scaling has to be done because the error estimates are computed based on certain assumptions of how the triangle
 * and ellipse are related. 
 * - 0(default) - BAMG
 * - 1 - Angener
 * - 2 - MMG2D
 * - 3 - Omegah
 * - 4 - MMG3D
 * In case of 3D we have only MMG3D now and the scaling factor is yet to be found out. As of now it is set to be 1.0. HACK NEEDS TO BE CORRECTED
 */
double ScalingFactor(const int opt_strategy, const int mesh_generator){
  double factor = 1.0;
  if (D==2)// 2D mesh generators
  {
    if(opt_strategy == 0 && mesh_generator == 1)// Old adaptation and BAMG
    {
      factor = 1.0;
    }
    else if(opt_strategy == 0 && mesh_generator == 2)// Old adaptation and Angener
    {
      factor = 3.0;
    }
    else if(opt_strategy == 0 && mesh_generator == 3)// Old adaptation and MMG2d
    {
      factor = 1.0;
    }
    else if(opt_strategy == 0 && mesh_generator == 4)// Old adaptation and Omega_h
    {
      factor = 1.0;
    }
    else if((opt_strategy == 1|| opt_strategy == 2) && mesh_generator == 1)// Analytic opt and BAMG
    {
      factor = 1.0/3.0;
    }
    else if(opt_strategy == 1 && mesh_generator == 2)// Analytic opt and Angener
    {
      factor = 1.0;
    }
    else if(opt_strategy == 1 && mesh_generator == 3)// Analytic opt and MMG2d
    {
      factor = 1.0/3.0;
    }
    else if(opt_strategy == 1 && mesh_generator == 4)// Analytic opt and Madlib
    {
      factor = 1.0/3.0;
    }
    else
    {
      cout<<"Default mesh generator BAMG"<<endl;
      if(opt_strategy == 1)
        factor = 1.0/3.0;
      else
        factor = 1.0;
    }
  }
  else if(D==3)// 3D mesh generators
  {
    if(opt_strategy == 1 && mesh_generator == 5)// Analytic opt and MMG3D
    {
      factor = (3.0*sqrt(3.0))/(16.0*sqrt(2.0));//HACK NEEDS TO BE COMPUTED
    }
    else if(opt_strategy == 1 && mesh_generator == 6)// Analytic opt and MMG3D
    {
      factor = (1.0);//HACK NEEDS TO BE COMPUTED
    }
    else if(opt_strategy == 1 && mesh_generator == 4)// Analytic opt and Madlib
    {
      factor = (1.0);//HACK NEEDS TO BE COMPUTED
    }

  }

  return factor;
}

  /*!
 * Compute the scaled metric by multiplying each component of the metric by a scaling factor
 * @param[in] scale - Scaling factor for each component of the metric. Size = 1 double
 */
  void Scale(const double scale) {
    for (int i = 0; i < Metric.size()/(D*(D+1)/2); ++i) {
      for(int j = 0; j < D*(D+1)/2; ++j)
      {
        int index = i*D*(D+1)/2+j;
        Metric[index] *= scale;
      }

    }
  }

/*!
 * Take a metric and raise it to a given power
 * @param[in] i - Index of the metric. Size = 1 int
 * @param[in] i - Power to which the metric eigen values have to be raised. Size = 1 double
 * @param[out] metric_out - The original metric raised to a given power. Size = D*(D+1)/2 double
 */  
void MetricPower(const int i, double power, vector<double> & metric_out) {

    vector<double> metric_loc(D*(D+1)/2,0.0);
    GetComponents(i, metric_loc);// Computes the original metric at that point
    vector<double> q_met(D*D,0.0); // Compute the angles which represent the eigenvectors
    vector<double> lambda(D, 0.0);// Compute the eigen values
    EigenDecomposition(lambda, q_met, metric_loc);
    for(int dd = 0; dd<D; ++dd)
    {
      lambda[dd] = pow(lambda[dd], power); // Raise lambda to the required power
    }
    ComputeMetric(lambda, q_met, metric_out);
  }
/*!
 * Take a metric and log it
 * @param[in] i - Index of the metric. Size = 1 int
 * @param[out] metric_out - The original metric raised to a given power. Size = D*(D+1)/2 double
 */  
void MetricLog(const int i, vector<double> & metric_out) {

    vector<double> metric_loc(D*(D+1)/2,0.0);
    GetComponents(i, metric_loc);// Computes the original metric at that point
    vector<double> q_met(D*D,0.0); // Compute the angles which represent the eigenvectors
    vector<double> lambda(D, 0.0);// Compute the eigen values
    EigenDecomposition(lambda, q_met, metric_loc);
    for(int dd = 0; dd<D; ++dd)
    {
      lambda[dd] = log(lambda[dd]); // Log the eigen values
    }
    ComputeMetric(lambda, q_met, metric_out);
  }
/*!
 * Take a metric and exp it
 * @param[in] i - Index of the metric. Size = 1 int
 * @param[out] metric_out - The original metric raised to a given power. Size = D*(D+1)/2 double
 */  
void MetricExp(const int i, vector<double> & metric_out) {

    vector<double> metric_loc(D*(D+1)/2,0.0);
    GetComponents(i, metric_loc);// Computes the original metric at that point
    vector<double> q_met(D*D,0.0); // Compute the angles which represent the eigenvectors
    vector<double> lambda(D, 0.0);// Compute the eigen values
    EigenDecomposition(lambda, q_met, metric_loc);
    for(int dd = 0; dd<D; ++dd)
    {
      lambda[dd] = exp(lambda[dd]); // Log the eigen values
    }
    ComputeMetric(lambda, q_met, metric_out);
  }
/*!
 * Take an index and compute the determinant of the metric at that point
 * @param[in] i - Index of the metric. Size = 1 int
 * @param[out] det - The determinant of the metric at that point. Size = D*(D+1)/2 double
 */  
double MetricDet(const int i) {

    double det;
    vector<double> metric_loc(D*(D+1)/2,0.0);
    GetComponents(i, metric_loc);// Computes the original metric at that point
    if(D==2)
    {
      double a = metric_loc[0];
      double b = metric_loc[1];
      double c = metric_loc[2];
      det = a*c-b*b;
    }
    else if(D==3)
    {
      double a = metric_loc[0];
      double b = metric_loc[1];
      double c = metric_loc[2];
      double d = metric_loc[3];
      double e = metric_loc[4];
      double f = metric_loc[5];
      det = a*(d*f-e*e) - b*(b*f-c*e) + c*(b*e-c*d);
    }
    else
    {
      cout<<"Error! Incorrect dimension!"<<endl;
      exit(1);
    }
    return det;
  }

/*!
 * Compute the number of elements whose anisotropy is stored
 * @param[out] n - Computes the number of elements whose anisotropy is stored. Size = 1 int
 */
  int GetSize() const {
    double n;
    n = Metric.size()/(D*(D+1)/2);
    return (int)n;
  }

  /*************************************** Implemented only for 2D*******************************************************/
  /*!
 * Compute the implied metric using the nodes of the triangle. Only implemented for 2D
 * @param[in] i - Index of the metric. Size = 1 int
 * @param[in] x1 - X-coordinate of first point. Size = 1 double
 * @param[in] y1 - Y-coordinate of first point. Size = 1 double
 * @param[in] x2 - X-coordinate of second point. Size = 1 double
 * @param[in] y2 - Y-coordinate of second point. Size = 1 double
 * @param[in] x3 - X-coordinate of third point. Size = 1 double
 * @param[in] y3 - Y-coordinate of third point. Size = 1 double
 */
  void SetNodes(const int i, const double x1, const double y1, const double x2, const double y2, const double x3, const double y3) {

    double A[3][3], invA[3][3];
    
    A[0][0] = pow(x2-x1,2);
    A[0][1] = 2.*(x2-x1)*(y2-y1);
    A[0][2] = pow(y2-y1,2);
    A[1][0] = pow(x3-x2,2);
    A[1][1] = 2.*(x3-x2)*(y3-y2);
    A[1][2] = pow(y3-y2,2);
    A[2][0] = pow(x1-x3,2);
    A[2][1] = 2.*(x1-x3)*(y1-y3);
    A[2][2] = pow(y1-y3,2);

    double detA    = A[0][0]*(A[1][1]*A[2][2]-A[2][1]*A[1][2]) - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    double invdetA = 1/detA;
  
    invA[0][0] =  (A[1][1]*A[2][2]-A[2][1]*A[1][2])*invdetA;
    invA[0][1] = -(A[0][1]*A[2][2]-A[0][2]*A[2][1])*invdetA;
    invA[0][2] =  (A[0][1]*A[1][2]-A[0][2]*A[1][1])*invdetA;
    invA[1][0] = -(A[1][0]*A[2][2]-A[1][2]*A[2][0])*invdetA;
    invA[1][1] =  (A[0][0]*A[2][2]-A[0][2]*A[2][0])*invdetA;
    invA[1][2] = -(A[0][0]*A[1][2]-A[1][0]*A[0][2])*invdetA;
    invA[2][0] =  (A[1][0]*A[2][1]-A[2][0]*A[1][1])*invdetA;
    invA[2][1] = -(A[0][0]*A[2][1]-A[2][0]*A[0][1])*invdetA;
    invA[2][2] =  (A[0][0]*A[1][1]-A[1][0]*A[0][1])*invdetA;

    double a = invA[0][0]+invA[0][1]+invA[0][2];
    double b = invA[1][0]+invA[1][1]+invA[1][2];
    double c = invA[2][0]+invA[2][1]+invA[2][2];
    vector<double> metric_loc(2*(2+1)/2, 0.0);
    metric_loc[0] = a;
    metric_loc[1] = b;
    metric_loc[2] = c;
    SetComponents(i, metric_loc);
  }


};
