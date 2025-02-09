#ifndef ADVECTION_H
#define ADVECTION_H

template <int D>
class Advection : virtual public DummyModel<D, 1> {
 public:

  // Characterization of the model
  /// The number of components in the problem is 1
  static const int  Components = 1;
  /// This is pure convection in the problem
  static const bool Convection = true;
  /// There is no diffusion in this problem
  static const bool Diffusion  = false;
  /// This is pure advection with no source term
  static const bool Source     = false;
  /// The number of volume functionals for the adjoint equation
  // static const int NumVolFunctionals = 1;
  /// The number of surface functionals for the adjoint equation
  static const int NumBdryFluxWeight = 0;
  /// This sets the type of initial condition. \c IT_Zero means that we set it to zero
  /// initial condition
  static const InitType inittype = IT_Zero;

/*!
 * GIves the advection speed \f$\beta\f$
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * direction. Size = D
 * @param[out] dir - Contains the direction \f$\beta(x,t)\f$. Size = D
 */
  static void GetDirection (SpatialParams<D> & sparam, Vec<D> & dir) {
    //    dir(0) = 1.;
    //dir(1) = 2. * pos(0);
    // dir = 1.; 

    // Using the setup from Hartmann thesis chapter 4 page 79
    // if(sparam.pos(0)<1.0)
    // {
    //   double beta_mag = sqrt(pow(2.5*sparam.pos(0)*sparam.pos(1),2.0)+pow(1.0-sparam.pos(0),2.0));
    //   dir(0) = (2.5*sparam.pos(0)*sparam.pos(1))/beta_mag;
    //   dir(1) = ((1.0-sparam.pos(0)))/beta_mag;
    // }
    // else
    // {
    //   double beta_mag = sqrt(pow(2.5*sparam.pos(0)*sparam.pos(1),2.0)+pow(sparam.pos(0)-1.0,2.0));
    //   dir(0) = ((2.5*sparam.pos(0)*sparam.pos(1)))/beta_mag;
    //   dir(1) = ((sparam.pos(0)-1.0))/beta_mag;
    // }
    // Hartmann thesis original
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x<1.0)
    // {
    //   double beta_mag = sqrt(pow(y, 2.0)+ pow(1.0-x,2.0));
    //   dir(0) = y/beta_mag;
    //   dir(1) = (1.0-x)/beta_mag;
    // }
    // else
    // {
    //   double beta_mag = sqrt(pow(2.0-y, 2.0)+ pow(x-1.0,2.0));
    //   dir(0) = (2.0-y)/beta_mag;
    //   dir(1) = (x-1.0)/beta_mag;
    // }
    
    // Using the setup from Caripio, Preito second test case
    dir(0) = sparam.pos(1);
    dir(1) = -sparam.pos(0);
  }
/*!
 * The function returns the convective flux
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * diffusive flux. Size=D
 * @param[out] res - Contains the diffusive flux \f$f_{c}(u,\nabla u)=\beta^{T}u\f$. Size = 1xD
 */
  template <typename SCAL>
  static void EvalConvFlux(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    Vec<D> dir;
    GetDirection(sparam, dir);
    res = state * Trans(dir);
  }
  /*!
 * Returns the derivative of convective flux with respect to solution
 * @param[in] state - Contains \f$u\f$ vector. Size = COMP double
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * diffusive flux. Size=D
 * @param[out] res - Contains the derivative of the convective flux \f$\beta\f$. Size = 1xD
 */
  // template <typename SCAL>
  // static void DerConvFlux(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<D> & res) {
  //   Vec<D> dir;
  //   GetDirection(sparam, dir);
  //   res = dir;
  // }

/*!
 * (trivial) spectral decomposition for the linear, scalar case
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * \c eig. Size = D
 * @param[in] fix - entropy fix (not used here). Size = 1
 * @param[out] eig - eigenvalue (normal advection speed). Size = 1
 * @param[in] eigvr - left eigenvectors (trivial). Size = 1x1
 * @param[in] eigvl - right eigenvectors (trivial). Size = 1x1
 */
  template <typename SCAL>
  static void EvalConvEigenSystem(Vec<1, SCAL> & state, SpatialParams<D> & sparam, double fix, Vec<1, SCAL> & eig, Mat<1, 1, SCAL> & eigvr, Mat<1, 1, SCAL> & eigvl) {
    Vec<D> dir;
    GetDirection(sparam, dir);
    eig = 0.;
    Vec<D, SCAL> adir(dir);
    for (int dd = 0; dd < D; ++dd)
      eig(0) += adir(dd) * sparam.normal(dd);
    eigvr = 1.;
    eigvl = 1.;
  }
  /*!
 * The function returns the boundary condition for the given problem
 * @param[in] bcnr - Contains the type of boundary condition. Size = 1
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * boundary value. Size=D
 * @param[out] bcstate - Contains the boundary condition at \f$(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<1, SCAL> & bcstate) {
    Vec<D> dir;
    GetDirection(sparam, dir);
    double dirn = 0.;
    for (int dd = 0; dd < D; ++dd)
      dirn += dir(dd) * sparam.normal(dd);

    // if (dirn > 0.) // Outflow
    //   bcstate = state;
    // else {
    //   if (bcnr == 0)
    //     bcstate(0) = 1. - sparam.pos(1);
    //   else if (bcnr == 1)
    //     bcstate(0) = sparam.pos(0) - 1.;
    // }
    // if (dirn > 0.) // Outflow
    //   bcstate = state;
    // else {
    //   // if((sparam.pos(0)>1.0/8.0 && sparam.pos(0)<3.0/4.0) && sparam.pos(1)==0)
    //   // {
    //   //   bcstate(0) = 1.0;
    //   // }
    //   // else
    //   // {
    //   //   bcstate(0) = 0.0;
    //   // }
    //   // Using the setup from Caripio, Preito second test case
    //   if(sparam.pos(0)==0)
    //   {
    //     bcstate(0) = 1.0;
    //   }
    //   else
    //   {
    //     bcstate(0) = 0.0;
    //   }
    // if (dirn > 0.) // Outflow
    //   bcstate(0) = state(0);
    // else {
    //   // double x = sparam.pos(0);
    //   // double y = sparam.pos(1);
    //   // bcstate(0) = cos(2*M_PI*x)*cos(2*M_PI*y);
    //   double x = sparam.pos(0);
    //   double y = sparam.pos(1);
    //   double alpha = 1000.0;
    //   // if(x == 0.0 || y == 0.0)
    //   {
    //     double a = fabs(x-y);
    //     bcstate(0) = a*(1-a)*(atan(alpha*(a-0.25))+atan(alpha*(0.75-a)));
    //   }
    // if (dirn > 0.) // Outflow
    //   bcstate(0) = state(0);
    // else {

    //   double x = sparam.pos(0);
    //   double y = sparam.pos(1);

    //   if(y == 0)
    //   {
    //     if(x > 1.0/8.0 && x < 3.0/4.0)
    //       bcstate(0) = 1.0;
    //     else
    //       bcstate(0) = 0.0;
    //   }
    //   if(x == 0)
    //     bcstate(0) = 0.0;
    // }

    if (dirn > 0.) // Outflow
      bcstate = state;
    else 
    {
      // Using the setup from Caripio, Preito second test case
      if(sparam.pos(0)==0)
      {
        // bcstate(0) = 1.0;
        // double x = sparam.pos(0);
        // double y = sparam.pos(1);
        // double alpha = 2000;
        // double r = sqrt(pow(x,2.0)+pow(y,2.0));
        // bcstate(0) = atan(1000*(r-1.0/2.0))/M_PI+atan(1000*(4.0/5.0-r))/M_PI;
        // bcstate(0) = atan(alpha*(r-2.0))/M_PI+atan(alpha*(4.0-r))/M_PI;
        bcstate(0) = 1.0;
      }
      else
      {
        bcstate(0) = 0.0;
      }
    }
  }
/*!
 * The function returns the convective flux at the boundary
 * @param[in] bcnr - Contains the type of boundary. Size = 1
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
 * flux. Size=D
 * @param[out] bcstate - Contains the boundary convective flux at \f$(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalBdryConvFlux(int bcnr, Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
    Vec<1, SCAL> bcstate;
    Mat<1, D, SCAL> flux;
    EvalBdryState(bcnr, state, sparam, bcstate);
    EvalConvFlux(bcstate, sparam, flux);
    res = flux * sparam.normal;
  }
// /*!
//  * The function returns the volume integral which is the target function
//  * @param[in] state - Contains \f$u\f$ vector. Size = 1
//  * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
//  * flux. Size=D
//  * @param[out] res - Contains the functional to be computed. Size = 1
//  */
//   template <typename SCAL>
//   static void EvalVolFunctionals(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<NumVolFunctionals, SCAL> & res) {
//     // // Peak at the point (xc, yc) with steepness alpha
//     double alpha = 100;
//     double xc = 0.8;
//     double yc = 0.2;
//     // double r0 = 0.001;
//     double x = sparam.pos(0);
//     double y = sparam.pos(1);
//     double dist = sqrt(pow(x-xc,2.0)+pow(y-yc,2.0));
//     // res = exp(-1.0/(1.0-(dist*dist)))*state(0);
//     res = exp(-alpha*dist*dist)*state(0);
//     // res = state(0)*state(0);
//   }

// /*!
//  * The function returns the surface integral which is the target function
//  * @param[in] bcnr - Contains the type of boundary condition. Size = 1
//  * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
//  * flux. Size=D
//  * @param[out] res - Contains the functional to be computed. Size = 1
//  */
  static void EvalBdryFluxWeight(int bcnr, SpatialParams<D> & sparam, Mat<NumBdryFluxWeight, 1> & res) {
    // hack for scalar and square test case
    // if(sparam.pos(0) == 1)
    //   res = 1.0;
    // else
    //   res = 0.0;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);

    // if(x == 2.0)
    // {
    //   // res = exp(pow(3.0/8.0,-2.0)-pow(pow(y-(5.0/8.0),2.0)-(3.0/8.0),-2.0));
    //   // Johan thesis modification
    //   double s = 2000.0;
    //   double z = 0.1;
    //   double weight = 0.5*exp(-s*pow(y-z-0.5, 2.0))*(1.0-cos(2.0*M_PI*(y-z)));
    //   res = weight;
    // }
    // else
    //   res = 0;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x == 1 || y == 1)
    // {
    //   double a = fabs(x-y);
    //   res = sin(M_PI*a);
    // }
    // else
    // {
    //   res = 0.0;
    // }
    if(sparam.pos(0) == 4.0)
      res = 1.0;
    else
      res = 0.0;
    //   Adjoint paper
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x == 1.0 || y == 0.0)
    // {
    //   res = cos(2.0*M_PI*x)*cos(2.0*M_PI*y);
    // }
    // else
    //   res = 0.0;
  }
/*!
 * The function returns the volume volume integral which is the target function
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
 * flux. Size=D
 * @param[out] res - Contains the functional to be computed. Size = 1
 */
  // template <typename SCAL>
  // static void EvalVolFunctionals(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<NumVolFunctionals, SCAL> & res) {
  //   // // // Peak at the point (xc, yc) with steepness alpha
  //   double delta = 1000.0;
  //   double eps = 0.001;
  //   double xc = 0.5-eps;
  //   double yc = 1.0-eps;//sqrt(pow(beta,2.0)-pow(xc,2.0));
  //   double x = sparam.pos(0);
  //   double y = sparam.pos(1);
  //   double z = sparam.pos(2);
  //   double dist = sqrt(pow(x-xc,2.0)+pow(y-yc,2.0));
  //   res = exp(-delta*dist*dist)*state(0);//*state(0)*state(0);
  // }
/*!

 * The function retursn the analytic solution for the problem if it exsits. Is used in case
 * of manufactured solution cases to compute the error.
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * analytical solution. Size=D
 * @param[out] res - Contains the analytic \f$u(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalAnalyticSol(Vec<1, SCAL> & w, Mat<1, D, SCAL> & q, SpatialParams<D> & sparam) {
    // Hack for scalar and square case
    double x = sparam.pos(0);
    double y = sparam.pos(1);
    double a = fabs(x-y);
    w(0) = sin(M_PI*a);
    q = 0.0;

  }

  static void LoadParameters(shared_ptr<PDE> apde) {
  }
  
};

#endif
