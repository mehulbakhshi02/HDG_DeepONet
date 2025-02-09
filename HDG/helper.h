#ifndef HELPER_HDG
#define HELPER_HDG
/** @file
 * @brief This file contains the data structures used for solution, residual, adjoint
 * and so on. It also contains the declaration of some variables that can be used throught
 * the code.
*/

/** @struct Residual
 *  @brief This structure contains the details of the residual for the three main variables
 *  @var Residual::resQ 
 *  The residual for the gradient variable \f$\nabla u\f$. Size = 1 double
 *  @var Residual::resW
 *  The residual for the primal solution \f$u\f$. Size = 1 double
 *  @var Residual::resL
 *  The residual for the hybrid variable that lies on the skeleton - \f$\lambda\f$. Size = 1 double
 */
struct Residual {
  double resQ;
  double resW;
  double resL;
};
/** @struct Solution
 *  @brief This structure contains the details of the solution for the three main variables
 *  @var Solution::vecQ 
 *  The solution for the gradient variable \f$\nabla u\f$. Size = COMPxDxndof double
 *  @var Solution::vecW
 *  The solution for the primal solution \f$u\f$. Size = COMPxndof double
 *  @var Solution::vecL
 *  The solution for the hybrid variable that lies on the skeleton - \f$\lambda\f$. <b>Size = COMPxndof_skeleton double. Needs
 *  to be corrected</b>
 */
struct Solution {
  vector<double> vecQ;
  vector<double> vecW;
  vector<double> vecL;
};
/** @struct Space
 *  @brief This structure contains the details of the FE space for the three variables
 *  @var Space::fes_q 
 *  The space for the gradient variable called \f$\tau\f$. <b>Size = COMPxDxndof double. Needs
 *  to be corrected</b>
 *  @var Space::fes_w
 *  The space for the primal solution called \f$v\f$. <b>Size = COMPxndof double. Needs
 *  to be corrected</b>
 *  @var Space::fes_l
 *  The space for the hybrid variable that lies on the skeleton called \f$\mu\f$. <b>Size = COMPxndof_skeleton double. Needs
 *  to be corrected</b>
 */
struct Space {
  shared_ptr<FESpace> fes_q;
  shared_ptr<FESpace> fes_w;
  shared_ptr<FESpace> fes_l;
};

shared_ptr<FESpace>  fes_scalar;
/// The nodal grid function in case we want to save the bamg solution. This only works for triangular meshes in 2D
shared_ptr<FESpace>  fes_nodal;

/** @struct GFunction
 *  @brief It is used to transfer the coefficients of the solution from our data structure to the netgen data
 *  structure which are ultimately used for visualization.
 *  @var GFunction::gf_w 
 *  This is the transfer function for the solution \f$u\f$.
 *  @var GFunction::gf_q
 *  This is the transfer function for the solution \f$\nabla u\f$.
 */
struct GFunction {
  shared_ptr<GridFunction> gf_w;
  shared_ptr<GridFunction> gf_q;
};

shared_ptr<GridFunction> gf_wd;

/// The number of faces including boundary faces. Size = 1 int
int nf;
/// The number of elements. Size = 1 int
int ne;
/// The number of faces per element. Used in cases when we have hybrid meshes. Size = 1 int
int nf_max;
/// Order of the polynomial. This is the maximum order in case of \f$hp\f$ used to allocate memory. Size = 1 int
int order;
/*! \name Contain the minimum and maximum order of the polynomial in the domain.
*/
//@{
int min_order, max_order;
//@}
/*! \name Contain the total degrees of freedom for solution, gradient and hybrid variable
*/
//@{
int ndof_w_total, ndof_q_total, ndof_l_total;
//@}

/*! \name Contain the total degrees of freedom for solution, gradient and hybrid variable at the time of computing the solution
this changes because we move to a richer space for adaptation and adjoint
*/
//@{
int ndof_w_solve, ndof_q_solve, ndof_l_solve;
//@}

int ndof_w_max, ndof_q_max, ndof_l_max;

int nip_max;

double resL2_min;

double stab_conv, stab_visc;

// Are we in the assembly or reconstruction phase?
enum AssemblyMode {
  AM_None,
  AM_AssemblyPrimal,
  AM_AssemblyAdjoint,
  AM_BackSolvePrimal,
  AM_BackSolveAdjoint,
  AM_ErrorEstimation // only for adjoint  
};

AssemblyMode assemblyMode = AM_None;
//AssemblyMode adjointMode = AM_None;

vector<vector<int> > ndof_converter_segm;
vector<vector<int> > ndof_converter_tri;
vector<vector<int> > ndof_converter_quad;
vector<vector<int> > ndof_converter_tet;
vector<vector<int> > ndof_converter_hex;

// Temporary arrays
vector<double> temp, temp2, temp3, temp4, temp5, insert, insert2, temp_lift, temp_lift2;
vector<double> matA, matB, matC, matD;
vector<double> matL, matM;
vector<double> matQ, matW;

double * sol;
vector<double> adj_rhs_l;

int get_ndof(const int order, const int type) {

  if (type == ET_SEGM)
    return (order + 1);
  else if (type == ET_TRIG)
    return (order + 1) * (order + 2) / 2;
  else if (type == ET_QUAD)
    return (order + 1) * (order + 1);
  else if (type == ET_TET)
    return (order + 1) * (order + 2) * (order + 3) / 6;
  else if (type == ET_HEX)
    return (order + 1) * (order + 1) * (order + 1);

  cout << "Element type not found: " << type << endl;
  exit(0);
}

void transfer_ndof(int order, int order_precond, int type, int* transfer) {

  int ndof_precond = get_ndof(order_precond, type);

  if (order == order_precond) {
    // Transfer is identity
    for (int i = 0; i < ndof_precond; ++i)
      transfer[i] = i;
    return;
  }

  if (type == ET_SEGM) {
    for (int i = 0; i < ndof_precond; ++i)
      transfer[i] = i;
  } else if (type == ET_TRIG) {

    int delta = order - order_precond;
    int index1 = 0;
    int index2 = 0;
    
    for (int i = 0; i <= order_precond; ++i) {
      for (int j = 0; j <= order_precond-i; ++j) {
        transfer[index2] = index1;
        index1++; index2++;
      }
      index1 += delta;
    }
  } else if (type == ET_QUAD) {
    for (int i = 0, ii = 0; i <= order_precond; ++i)
      for (int j = 0; j <= order_precond; ++j)
        transfer[ii++] = j+i*(order+1);
  } else if (type == ET_TET) {
    int delta = order - order_precond;
    int index1 = 0;
    int index2 = 0;

    for (int i = 0; i <= order_precond; ++i) {
      for (int j = 0; j <= order_precond-i; ++j) {
        for (int k = 0; k <= order_precond-i-j; ++k) {
          transfer[index2] = index1;
          index1++; index2++;
        }
        index1 += delta;
      }
      index1 = (order-i+1) * (order-i+2) / 2;
    }
  } else if (type == ET_HEX) {
    for (int i = 0, ii = 0; i <= order_precond; ++i)
      for (int j = 0; j <= order_precond; ++j)
        for (int k = 0; k <= order_precond; ++k)
          transfer[ii++] = k+(order+1)*(j+(order+1)*i);
  }
}


#endif
