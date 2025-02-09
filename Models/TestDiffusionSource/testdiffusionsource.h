#ifndef TestDiffusionSource_H
#define TestDiffusionSource_H

template <int D>
class TestDiffusionSource : virtual public DummyModel<D, 1> {
 public:

  // Characterization of the model
  /// The number of components in the problem is 1
  static const int  Components = 1;
  /// There is no convection in this problem
  static const bool Convection = false;
  /// This is a purely diffusiv problem
  static const bool Diffusion  = true;
  /// There could be a source term in this equation
  static const bool Source     = true;
  /// This sets the type of initial condition. \c IT_Zero means that we set it to zero
  /// initial condition
  static const InitType inittype = IT_Function; //Function;
  // User-defined parameters
  /// \f$\epsilon\f$ is the parameter that scales the viscous flux
  static double epsilon;
  /// \f$d0\f$ is a constant used to define the exact solution
  static double alpha;
  /// \f$d1\f$ is a constant used to define the exact solution
  static double beta;

    /*!
 * <b>DO NOT KNOW WHAT THIS FUNCTION DOES</b>
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] rhs - <b>DO NOT KNOW WHAT THIS VARIABLE IS</b>. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate res. Size=D
 * @param[out] res - <b>DO NOT KNOW WHAT THIS FUNCTION RETURNS</b>. Size = 1xD
 */
  template <typename SCAL>
  static void ApplyK(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, Mat<1, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    res = epsilon * rhs;
  }  
    /*!
 * The function returns the difussion flux as a vector
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * diffusive flux. Size=D
 * @param[out] res - Contains the diffusive flux \f$f_{c}(u,\nabla u)\f$. Size = 1xD
 */
  template <typename SCAL>  
  static void EvalDiffFlux(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    res = epsilon * grad;

  }
/*!
 * The initial condition for the problem
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * initial condition. Size=D
 * @param[out] res - Contains the initial \f$u_{0}(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalInitial(SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
    double r0 = beta;
    double r = 0.0;
    res = 1.0;
    for (int dd = 0; dd < D; ++dd)
    {
      r += pow(sparam.pos(dd), 2.0);
      res *= sparam.pos(dd)*(1.0-sparam.pos(dd));
    }
    r = sqrt(r);
    res *= (atan(alpha*(r-r0))+atan(alpha*r0));
  }

  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<1, SCAL> & bcstate) {
    EvalInitial(sparam, bcstate);
  }
  /*!
 * The function returns the source term
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * source. Size=D
 * @param[out] res - Contains the source at \f$s(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalSource(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
    double r0 = beta;
    double r = 0.0;
    for (int dd = 0; dd < D; ++dd)
    {
      r += pow(sparam.pos(dd), 2.0);
    }
    r = sqrt(r);
    double eps = epsilon;
    if(D==2)
    {
      double x = sparam.pos(0);
      double y = sparam.pos(1);
      res[0] = -eps*(x*(atan(alpha*r0)-atan(alpha*(r0-r)))*(x-1.0)*2.0
               +(alpha*x*(y*y)*1.0/r*(x-1.0)*2.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
               -(alpha*x*(y*y*y)*1.0/pow(r,3.0)*(x-1.0)*(y-1.0))/((alpha*alpha)*pow(r0-r,2.0)+1.0)
               +(alpha*x*y*1.0/r*(x-1.0)*(y-1.0)*3.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
               +((alpha*alpha*alpha)*x*(y*y*y)*1.0/pow((alpha*alpha)*pow(r0-r,2.0)+1.0,2.0)*(r0-r)*(x-1.0)*(y-1.0)*2.0)/(pow(r,2.0)))
               -eps*(y*(atan(alpha*r0)-atan(alpha*(r0-r)))*(y-1.0)*2.0
               +(alpha*(x*x)*y*1.0/r*(y-1.0)*2.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
               -(alpha*(x*x*x)*y*1.0/pow(r,3.0)*(x-1.0)*(y-1.0))/((alpha*alpha)*pow(r0-r,2.0)+1.0)
               +(alpha*x*y*1.0/r*(x-1.0)*(y-1.0)*3.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
               +((alpha*alpha*alpha)*(x*x*x)*y*1.0/pow((alpha*alpha)*pow(r0-r,2.0)+1.0,2.0)*(r0-r)*(x-1.0)*(y-1.0)*2.0)/(pow(r,2.0)));
    }
    if(D==3)
    {
      double x = sparam.pos(0);
      double y = sparam.pos(1);
      double z = sparam.pos(2);
      res[0] = eps*(x*y*(x-1.0)*(y-1.0)*(atan(alpha*r0)-atan(alpha*(r0-r)))*2.0
                +(alpha*x*y*(z*z)*(x-1.0)*(y-1.0)*1.0/r*2.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                -(alpha*x*y*(z*z*z)*(x-1.0)*(y-1.0)*(z-1.0)*1.0/pow(r,3.0))/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                +(alpha*x*y*z*(x-1.0)*(y-1.0)*(z-1.0)*1.0/r*3.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                +((alpha*alpha*alpha)*x*y*(z*z*z)*1.0/pow((alpha*alpha)*pow(r0-r,2.0)+1.0,2.0)*(r0-r)*(x-1.0)*(y-1.0)*(z-1.0)*2.0)/(pow(r,2.0)))
                +eps*(x*z*(x-1.0)*(z-1.0)*(atan(alpha*r0)-atan(alpha*(r0-r)))*2.0
                +(alpha*x*(y*y)*z*(x-1.0)*(z-1.0)*1.0/r*2.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                -(alpha*x*(y*y*y)*z*(x-1.0)*(y-1.0)*(z-1.0)*1.0/pow(r,3.0))/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                +(alpha*x*y*z*(x-1.0)*(y-1.0)*(z-1.0)*1.0/r*3.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                +((alpha*alpha*alpha)*x*(y*y*y)*z*1.0/pow((alpha*alpha)*pow(r0-r,2.0)+1.0,2.0)*(r0-r)*(x-1.0)*(y-1.0)*(z-1.0)*2.0)/(pow(r,2.0)))
                +eps*(y*z*(y-1.0)*(z-1.0)*(atan(alpha*r0)-atan(alpha*(r0-r)))*2.0
                +(alpha*(x*x)*y*z*(y-1.0)*(z-1.0)*1.0/r*2.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                -(alpha*(x*x*x)*y*z*(x-1.0)*(y-1.0)*(z-1.0)*1.0/pow(r,3.0))/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                +(alpha*x*y*z*(x-1.0)*(y-1.0)*(z-1.0)*1.0/r*3.0)/((alpha*alpha)*pow(r0-r,2.0)+1.0)
                +((alpha*alpha*alpha)*(x*x*x)*y*z*1.0/pow((alpha*alpha)*pow(r0-r,2.0)+1.0,2.0)*(r0-r)*(x-1.0)*(y-1.0)*(z-1.0)*2.0)/(pow(r,2.0)));

    }
  }

  /*!
 * The function returns the diffusive flux at the boundary
 * @param[in] bcnr - Contains the type of boundary. Size = 1
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
 * flux. Size=D
 * @param[out] bcstate - Contains the boundary diffusive flux at \f$(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalBdryDiffFlux(int bcnr, Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {

    Vec<1, SCAL> bcstate;
    EvalBdryState(bcnr, state, sparam, bcstate);
    Mat<1, D, SCAL> flux;
    EvalDiffFlux(state, grad, sparam, flux);
    res = flux * sparam.normal;
  }
  static void LoadParameters(shared_ptr<PDE> apde) {
    
    if (apde->ConstantUsed("epsilon"))
      epsilon = apde->GetConstant("epsilon");
    
    if (apde->ConstantUsed("alpha"))
      alpha = apde->GetConstant("alpha");

    if (apde->ConstantUsed("beta"))
      beta = apde->GetConstant("beta");
  }
  
  

};
template<int D> double TestDiffusionSource<D>::epsilon;
template<int D> double TestDiffusionSource<D>::alpha;
template<int D> double TestDiffusionSource<D>::beta;

#endif
