#ifndef POISSON_H
#define POISSON_H

template <int D>
class Poisson : virtual public DummyModel<D, 1> {
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
  /// The number of volume functionals for the adjoint equation
  static const int NumVolFunctionals = 1;
  /// The number of surface functionals for the adjoint equation
  // static const int NumBdryFluxWeight = 0;
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
 * The initial condition for the problem
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * initial condition. Size=D
 * @param[out] res - Contains the initial \f$u_{0}(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalInitial(SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
      // Scalar boundary layer
    // double eps = Poisson<D>::epsilon;
    // res = 1.0;
    // for (int dd = 0; dd < D; ++dd)
    // {
    //   double pos = sparam.pos(dd);
    //   res *= (pos+(exp(pos/eps)-1.0)/(1.0-exp(1.0/eps)));
    // }
    // res = 0.0;
    // double x_new = sparam.pos(0)-0.5;
    // double y_new = sparam.pos(1)-0.5;
    // res = pow(x_new, 2.0)+pow(y_new,2.0)+pow(x_new,3.0)/10.0+pow(y_new,4.0)/100000.0;
    // double temp = 0.;
    // for (int dd = 0; dd < D; ++dd)
    //   temp += pow(sparam.pos(dd) - d1, 2.);
    // res = exp(-d0 * temp);
    // double alpha = Poisson<D>::alpha;
    // double eps = Poisson<D>::epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double x1 = 1.0/3.0;
    // double x2 = 2.0/3.0;
    // double y1 = x1;
    // double y2 = x2;
    // res = (atan(alpha*(x-x1))+atan(alpha*(x2-x)))*(atan(alpha*(y-y1))+atan(alpha*(y2-y)));
    // Addition of two inverse tangents
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
    // double alpha = alp;
    // double r0 = beta;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double z = sparam.pos(2);
    // double xc = -0.05;
    // double yc = -0.05;
    // double zc = -0.05;
    // double r = sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0));
    // double eps = epsilon;
    // res = atan(alpha*(r-r0));
    // Bl
    // double eps = epsilon;
    // res = 1.0;
    // for (int dd = 0; dd < D; ++dd)
    // {
    //   double pos = sparam.pos(dd);
    //   res *= (pos+(exp(pos/eps)-1.0)/(1.0-exp(1.0/eps)));
    // }
    // Exact solution is x^2+y^2
    // double eps = epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double z = sparam.pos(2);
    // res[0] = pow(x, 3.0)+pow(y,3.0)+pow(z,3.0);
    // // res[0] = 1;
    // // if(x<y)
    // //   res[0] = 1;
    // // if(x>=y)
    // //   res[0] = 0;
    // // res[0] = eps*(pow(x,2.0)+pow(y,2.0));
    // res[0] = pow(x, 1.0)+pow(y, 2.0)+pow(z, 3.0);
    // Interior line singularity
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // // double z = sparam.pos(2);
    // if(x<=0.5 + beta*y)
    // {
    //   res = cos(M_PI*(y-0.5));
    // }
    // else
    // {
    //   res = cos(M_PI*(y-0.5)) + pow(x-beta*y-0.5, alpha);
    // }
    // Multiple polynomial orders
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x<=0.5 && y<=0.5)
    // {
    //   res = pow(x,2.0)*pow(y,2.0);
    // }
    // // else if(x>0.5 && y<=0.5)
    // // {
    // //   res = pow(x,6.0)*pow(y,2.0);
    // // }
    // // else if(x>0.5 && y>0.5)
    // // {
    // //   res = pow(x, 0.0)*pow(y, 1.0);
    // // }
    // else
    // {
    //   res = pow(x,2.0)*pow(y,3.0);
    // }
    // Square jump
    // res = 1.0;
    // double alpha = Poisson<D>::alpha;
    // double eps = Poisson<D>::epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double x1 = 1.0/3.0;
    // double x2 = 2.0/3.0;
    // double y1 = x1;
    // double y2 = x2;
    // res = (atan(alpha*(x-x1))+atan(alpha*(x2-x)))*(atan(alpha*(y-y1))+atan(alpha*(y2-y)));
    // Fichera corner
    // double q_exp = beta;
    // double r = 0.0;
    // res = 0.0;
    // for (int dd = 0; dd < D; ++dd)
    // {
    //   r += pow(sparam.pos(dd), 2.0);
    // }
    // r = sqrt(r);
    // res = pow(r, q_exp);
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
 * Returns the derivative of diffusive flux with respect to solution
 * @param[in] state - Contains \f$u\f$ vector. Size = COMP double
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * diffusive flux. Size=D
 * @param[out] res - Contains the derivative of the convective flux \f$a(\zeta)\f$. Size = 1xD
 */
  // template <typename SCAL>
  // static void DerDiffFlux(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<D> & res) {
  //   res = epsilon;
  // }
  /*!
 * Returns the derivative of coefficient of the diffusive flux
 * @param[in] state - Contains \f$u\f$ vector. Size = COMP double
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * diffusive flux. Size=D
 * @param[out] res - Contains the derivative of the convective flux \f$a'(\zeta)\f$. Size = 1xD
 */
  // template <typename SCAL>
  // static void DerDiffFluxCoeff(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<D> & res) {
  //   res = 0.0;
  // }


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
 * The function returns the source term
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * source. Size=D
 * @param[out] res - Contains the source at \f$s(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalSource(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {

  // The exact solution for this source is \f$u(x,y) = e^{-d0((x-d1)^2+(y-d1)^2)}\f$
    // double temp = 0.;
    // for (int dd = 0; dd < D; ++dd)
    //   temp += pow(sparam.pos(dd) - d1, 2.);
    // temp = exp(-d0 * temp);

    // double temp2 = 0.;
    // for (int dd = 0; dd < D; ++dd)
    //   temp2 += -epsilon * (1. - 2. * d0 * pow(sparam.pos(dd) - d1, 2.));
    // res = -2. * temp * d0 * temp2;
    // Addition of two inverse tangents
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
    //     // Fichera corner
    // double q_exp = beta;
    // double eps = epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double z = sparam.pos(2);
    // res = -eps*(q_exp*pow(x*x+y*y+z*z,q_exp/2.0-1.0)*3.0+q_exp*(x*x)*(q_exp/2.0-1.0)*pow(x*x+y*y+z*z,q_exp/2.0-2.0)*2.0+q_exp*(y*y)*(q_exp/2.0-1.0)*pow(x*x+y*y+z*z,q_exp/2.0-2.0)*2.0+q_exp*(z*z)*(q_exp/2.0-1.0)*pow(x*x+y*y+z*z,q_exp/2.0-2.0)*2.0);
    // The exact solution is x^2+y^2 which gives the analytical solution as
    // 2 + 2
    // double eps = epsilon;
    // res[0] = -eps*(2.0 + 2.0);
    // Mild wave front from Mitchell paper
    // double alpha = alp;
    // double r0 = beta;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double z = sparam.pos(2);
    // double xc = -0.05;
    // double yc = -0.05;
    // double zc = -0.05;
    // double r = sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0));
    // double eps = epsilon;
    // res[0] = eps * ((alpha*1.0/sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0))*-3.0)/((alpha*alpha)*pow(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)),2.0)+1.0)+(alpha*pow(x-xc,2.0)*1.0/pow(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0),3.0/2.0))/((alpha*alpha)*pow(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)),2.0)+1.0)+(alpha*pow(y-yc,2.0)*1.0/pow(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0),3.0/2.0))/((alpha*alpha)*pow(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)),2.0)+1.0)+(alpha*pow(z-zc,2.0)*1.0/pow(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0),3.0/2.0))/((alpha*alpha)*pow(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)),2.0)+1.0)-((alpha*alpha*alpha)*1.0/pow((alpha*alpha)*pow(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)),2.0)+1.0,2.0)*(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)))*pow(x-xc,2.0)*2.0)/(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0))-((alpha*alpha*alpha)*1.0/pow((alpha*alpha)*pow(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)),2.0)+1.0,2.0)*(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)))*pow(y-yc,2.0)*2.0)/(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0))-((alpha*alpha*alpha)*1.0/pow((alpha*alpha)*pow(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)),2.0)+1.0,2.0)*(r0-sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)))*pow(z-zc,2.0)*2.0)/(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0)));
     // // Interior line singularity
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double eps = epsilon;
    // if(alpha!=1.0)
    // { 
    //   if(x<=0.5+beta*y)
    //   {
    //     res[0] = eps*(M_PI*M_PI)*cos(M_PI*(y-1.0/2.0));
    //   }
    //   else
    //   {
    //     res[0] = eps*((M_PI*M_PI)*cos(M_PI*(y-1.0/2.0))-alpha*(beta*beta)*(alpha-1.0)*pow(x-beta*y-1.0/2.0,alpha-2.0))-alpha*eps*(alpha-1.0)*pow(x-beta*y-1.0/2.0,alpha-2.0);
    //   }
    // }
    // else
    // {
    //     res[0] = (M_PI*M_PI)*eps*cos(M_PI*(y-1.0/2.0));
    // }
    // Scalar boundary layer
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double eps = epsilon;
    // res[0] = eps*((1.0/(eps*eps)*exp(y/eps)*(x-(exp(x/eps)-1.0)/(exp(1.0/eps)-1.0)))/(exp(1.0/eps)-1.0)+(1.0/(eps*eps)*exp(x/eps)*(y-(exp(y/eps)-1.0)/(exp(1.0/eps)-1.0)))/(exp(1.0/eps)-1.0));
    // res[0] = 0.0;
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
    EvalInitial(sparam, bcstate);
    // bcstate = 0.0;
    // if(sparam.pos(0)==1.0)
    //   bcstate = 1.0;
    // else
    //   bcstate = 0.0;
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
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x == 4.0 || y == 0.0)
    // {
    //   res = 0.0;
    // }
    // else
    {
      Vec<1, SCAL> bcstate;
      EvalBdryState(bcnr, state, sparam, bcstate);
      Mat<1, D, SCAL> flux;
      EvalDiffFlux(state, grad, sparam, flux);
      res = flux * sparam.normal;      
    }

  }
/*!
 * The function returns the volume volume integral which is the target function
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
 * flux. Size=D
 * @param[out] res - Contains the functional to be computed. Size = 1
 */
  template <typename SCAL>
  static void EvalVolFunctionals(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<NumVolFunctionals, SCAL> & res) {
    // // // Peak at the point (xc, yc) with steepness alpha
    // double delta = alpha;
    // double xc = 0.99;
    // double yc = 0.5;//sqrt(pow(beta,2.0)-pow(xc,2.0));
    // // double yc = sqrt(pow(beta,2.0)-pow(xc,2.0));
    // // double zc = 0.5;//sqrt(pow(beta,2.0)-pow(xc,2.0));
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // // double z = sparam.pos(2);
    // // double dist_x = max(pow(x-xc, 2.0), pow(0.1, 2.0))-0.01;
    // // double dist_y = max(pow(y-yc, 2.0), pow(0.1, 2.0))-0.01;
    // // double dist = sqrt(dist_x+dist_y);
    // double dist = sqrt(pow(x-xc,2.0)+pow(y-yc,2.0));
    // // double dist = sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0));
    // res = exp(-delta*dist*dist)*state(0);//*state(0)*state(0);
    // // double x = sparam.pos(0);
    // double y = sparam.pos(1);

    // // Source for same boundary layer test case
    // res = state(0)*temp;
    // Anisotropic opposite boundary layer
    // double eps = epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double beta1 = 1.0;
    // double beta2 = 1.0;
    // double temp = -eps*(((beta2*beta2)*1.0/(eps*eps)*exp(-(beta2*(y-1.0))/eps)*(x+(exp(-(beta1*(x-1.0))/eps)-1.0)/(exp(beta1/eps)-1.0)-1.0))/(exp(beta2/eps)-1.0)+((beta1*beta1)*1.0/(eps*eps)*exp(-(beta1*(x-1.0))/eps)*(y+(exp(-(beta2*(y-1.0))/eps)-1.0)/(exp(beta2/eps)-1.0)-1.0))/(exp(beta1/eps)-1.0))+beta2*((beta2*exp(-(beta2*(y-1.0))/eps))/(eps*(exp(beta2/eps)-1.0))-1.0)*(x+(exp(-(beta1*(x-1.0))/eps)-1.0)/(exp(beta1/eps)-1.0)-1.0)+beta1*((beta1*exp(-(beta1*(x-1.0))/eps))/(eps*(exp(beta1/eps)-1.0))-1.0)*(y+(exp(-(beta2*(y-1.0))/eps)-1.0)/(exp(beta2/eps)-1.0)-1.0);
    // res = state(0)*temp;
    // double eps = epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double temp = (exp(-(y-1.0)/eps)/(eps*(exp(1.0/eps)-1.0))-1.0)*(x+(exp(-(x-1.0)/eps)-1.0)/(exp(1.0/eps)-1.0)-1.0)+(exp(-(x-1.0)/eps)/(eps*(exp(1.0/eps)-1.0))-1.0)*(y+(exp(-(y-1.0)/eps)-1.0)/(exp(1.0/eps)-1.0)-1.0)-eps*((1.0/(eps*eps)*exp(-(y-1.0)/eps)*(x+(exp(-(x-1.0)/eps)-1.0)/(exp(1.0/eps)-1.0)-1.0))/(exp(1.0/eps)-1.0)+(1.0/(eps*eps)*exp(-(x-1.0)/eps)*(y+(exp(-(y-1.0)/eps)-1.0)/(exp(1.0/eps)-1.0)-1.0))/(exp(1.0/eps)-1.0));
    // res = state(0)*temp;
    // Opp boundary layer
    // double eps = epsilon;
    // double temp = 0.0;
    // for (int dd = 0; dd < D; ++dd)
    // {
    //   double res_loc = 1.0; 
    //   for(int ll = 0; ll < D-1; ++ll)
    //   {
    //     int index = (dd+ll+1)%D;
    //     double pos = sparam.pos(index);
    //     res_loc *= exp(-(pos-1.0)/eps)/(exp(1.0/eps)-1.0)*(exp(pos/eps)+pos*exp((pos-1.0)/eps)-pos*exp(pos/eps)-1.0);
    //   }
    //   temp += res_loc;
    // }
    // res = state(0)*temp;
    // Solution within box
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x > 0.3 && x < 0.5 && y > 0.3 && y < 0.5)
    //   res = state(0);
    // else
    //   res = 0.0;
    // Mild wave 
    // double alp = alpha;
    // double r0 = 0.5;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double z = sparam.pos(2);
    // double xc = 0.05;
    // double yc = 0.05;
    // double zc = 0.05;
    // double r = sqrt(pow(x-xc,2.0)+pow(y-yc,2.0)+pow(z-zc,2.0));
    // res = atan(alp*(r-r0)) * state(0);
    // L shaped
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);

    // if(x > 2.5 && x < 3.5 && y > 2.5 && y < 3.5)
    //   res = state(0);
    // else
    //   res = 0.0;
  }
/*!
 * The function returns the surface integral which is the target function
 * @param[in] bcnr - Contains the type of boundary condition. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
 * flux. Size=D
 * @param[out] res - Contains the functional to be computed. Size = 1
 */
  // static void EvalBdryFluxWeight(int bcnr, SpatialParams<D> & sparam, Mat<NumBdryFluxWeight, 1> & res) {  
  //   double x = sparam.pos(0);
  //   double y = sparam.pos(1);
  //   if(y == 0.0)
  //     res = 1.0;
  //   else
  //     res = 0.0;

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
    EvalInitial(sparam, w);
    q = 0.0;
    // For 2D
    // double r0 = beta;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // q(0,0) = x*y*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y))))*(y-1.0)+y*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y))))*(x-1.0)*(y-1.0)+(alpha*(x*x)*y*1.0/sqrt(x*x+y*y)*(x-1.0)*(y-1.0))/((alpha*alpha)*pow(r0-sqrt(x*x+y*y),2.0)+1.0);
    // q(0,1) = x*y*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y))))*(x-1.0)+x*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y))))*(x-1.0)*(y-1.0)+(alpha*x*(y*y)*1.0/sqrt(x*x+y*y)*(x-1.0)*(y-1.0))/((alpha*alpha)*pow(r0-sqrt(x*x+y*y),2.0)+1.0);
    // // // For 3D
    // double alpha = alp;
    // double r0 = beta;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double z = sparam.pos(2);
    // q(0,0) = -x*y*z*(y-1.0)*(z-1.0)*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y+z*z))))-y*z*(x-1.0)*(y-1.0)*(z-1.0)*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y+z*z))))-(alpha*(x*x)*y*z*(x-1.0)*(y-1.0)*(z-1.0)*1.0/sqrt(x*x+y*y+z*z))/((alpha*alpha)*pow(r0-sqrt(x*x+y*y+z*z),2.0)+1.0);
    // q(0,1) = -x*y*z*(x-1.0)*(z-1.0)*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y+z*z))))-x*z*(x-1.0)*(y-1.0)*(z-1.0)*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y+z*z))))-(alpha*x*(y*y)*z*(x-1.0)*(y-1.0)*(z-1.0)*1.0/sqrt(x*x+y*y+z*z))/((alpha*alpha)*pow(r0-sqrt(x*x+y*y+z*z),2.0)+1.0);
    // q(0,2) = -x*y*z*(x-1.0)*(y-1.0)*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y+z*z))))-x*y*(x-1.0)*(y-1.0)*(z-1.0)*(atan(alpha*r0)-atan(alpha*(r0-sqrt(x*x+y*y+z*z))))-(alpha*x*y*(z*z)*(x-1.0)*(y-1.0)*(z-1.0)*1.0/sqrt(x*x+y*y+z*z))/((alpha*alpha)*pow(r0-sqrt(x*x+y*y+z*z),2.0)+1.0);
    // double eps = Poisson<D>::epsilon;
    // w(0) = 1.0;
    // for (int dd = 0; dd < D; ++dd)
    // {
    //   double pos = sparam.pos(dd);
    //   w(0) *= (pos+(exp(pos/eps)-1.0)/(1.0-exp(1.0/eps)));
    // }
    // // Interior line singularity
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x<=0.5 + beta*y)
    // {
    //   q(0,0) = 0.0;
    //   q(0,1) = -M_PI*sin(M_PI*(y-1.0/2.0));
    // }
    // else
    // {
    //   q(0,0) = alpha*pow(x-beta*y-1.0/2.0,alpha-1.0);
    //   q(0,1) = -M_PI*sin(M_PI*(y-1.0/2.0))-alpha*beta*pow(x-beta*y-1.0/2.0,alpha-1.0);
    // }

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

// Default parameter settings (have to be outside the class)
template<int D> double Poisson<D>::epsilon;
template<int D> double Poisson<D>::alpha;
template<int D> double Poisson<D>::beta;

#endif
