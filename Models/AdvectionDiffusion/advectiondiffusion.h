#ifndef ADVECTIONDIFFUSION_H
#define ADVECTIONDIFFUSION_H

#include "../Advection/advection.h"
#include "../Poisson/poisson.h"

template <int D>
class AdvectionDiffusion : public Advection<D>, public Poisson<D> {
 public:

  // Charaterization of the model
  static const int  Components = 1;
  static const bool Convection = true;
  static const bool Diffusion  = true;
  static const bool Source     = true;

  static const InitType inittype = IT_Function; //Function;

  static const int NumDerVar = 1; //

  static const int NumParam = 3;
  static const int NumVolFunctionals = 1;
  // Both Advection and Poisson provide these functions
  // Therefore, we have to make a decision
  // using Poisson<D>::EvalInitial;
  // using Poisson<D>::EvalBdryState;
  using Poisson<D>::ApplyK;
  // using Poisson<D>::EvalVolFunctionals;

  using Advection<D>::EvalBdryFluxWeight;

  static void GetDerVarName(int dervar, string & name) {
        name = "Error";
  }

  // User-defined parameters
  /// \f$\epsilon\f$ is the parameter that scales the viscous flux
  // static double epsilon;

  static void GetParameters(Vec<NumParam> & param) {


    double alpha = Poisson<D>::alpha;
    double beta = Poisson<D>::beta;
    double eps = Poisson<D>::epsilon;
    param(0) = alpha;
    param(1) = beta;
    param(2) = eps;

  }

  static void GetFilename(stringstream & oss) {

    double alpha = Poisson<D>::alpha;
    double beta = Poisson<D>::beta;
    double eps = Poisson<D>::epsilon;
    oss << "-eps-" << eps << "-beta1-"<< alpha <<"-beta2-"<<beta;
  }

/*!
 * Gives the advection speed \f$\beta\f$
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * direction. Size = D
 * @param[out] dir - Contains the direction \f$\beta(x,t)\f$. Size = D
 */
  static void GetDirection (SpatialParams<D> & sparam, Vec<D> & dir) {
       // dir(0) = 1.;
       // dir(1) = 0.;
       // dir(2) = 0.;
    //dir(1) = 2. * pos(0);
    // dir = 1.;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // dir(0) = 2.0*y*(1.0-pow(x,2.0));
    // dir(1) = -2.0*x*(1.0-pow(y,2.0));
    // double alpha = Poisson<D>::alpha;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    double alpha = Poisson<D>::alpha;
    double beta = Poisson<D>::beta;

    dir(0) = alpha;
    dir(1) = beta;
    // dir(0) = y;
    // dir(1) = -x;
    // dir(2) = 1.0;
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
 * The function returns the difussion flux as a vector
 * @param[in] state - Contains \f$u\f$ vector. Size = 1
 * @param[in] grad - Contains \f$\nabla u\f$. Size = 1xD
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * diffusive flux. Size=D
 * @param[out] res - Contains the diffusive flux \f$f_{c}(u,\nabla u)\f$. Size = 1xD
 */
  template <typename SCAL>  
  static void EvalDiffFlux(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    res = Poisson<D>::epsilon * grad;

  }
 //  /*!
 // * Returns the derivative of diffusive flux with respect to solution
 // * @param[in] state - Contains \f$u\f$ vector. Size = COMP double
 // * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 // * diffusive flux. Size=D
 // * @param[out] res - Contains the derivative of the convective flux \f$a(\zeta)\f$. Size = 1xD
 // */
 //  template <typename SCAL>
 //  static void DerDiffFlux(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<D> & res) {
 //    res = Poisson<D>::epsilon;
 //  }
 //  /*!
 // * Returns the derivative of coefficient of the diffusive flux
 // * @param[in] state - Contains \f$u\f$ vector. Size = COMP double
 // * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 // * diffusive flux. Size=D
 // * @param[out] res - Contains the derivative of the convective flux \f$a'(\zeta)\f$. Size = 1xD
 // */
 //  template <typename SCAL>
 //  static void DerDiffFluxCoeff(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<D> & res) {
 //    res = 0.0;
 //  }
  /*!
 * The initial condition for the problem
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the
 * initial condition. Size=D
 * @param[out] res - Contains the initial \f$u_{0}(x,y)\f$. Size = 1
 */
  template <typename SCAL>
  static void EvalInitial(SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
  	// Anisotropic boundary layer
    double eps = Poisson<D>::epsilon;
    Vec<D> dir;
    GetDirection(sparam, dir);
    double beta1 = dir(0);
    double beta2 = dir(1);
    double x = sparam.pos(0);
    double y = sparam.pos(1);
  	res = (x-(exp((beta1*x)/eps)-1.0)/(exp(beta1/eps)-1.0))*(y-(exp((beta2*y)/eps)-1.0)/(exp(beta2/eps)-1.0));
    // Internal layer
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(y>0.5*(1.0+x))
    //   res = 1.0;
    // else 
    //   res = 0.0;
    // L2 projection
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // // double z = sparam.pos(2);
    // // // if(x < 0.5)
    //   res = 1e-3/60.*pow(x,5.0)+ 1./6.*pow(y,3.0);
    // else
      // res = pow(x,2.0)*pow(y,2.0)*pow(z,1.0);

    // Scalar boundary layer
    // double eps = Poisson<D>::epsilon;
    // res = 1.0;
    // for (int dd = 0; dd < D; ++dd)
    // {
    //   double pos = sparam.pos(dd);
    //   res *= (pos+(exp(pos/eps)-1.0)/(1.0-exp(1.0/eps)));
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
    // Eriksson–Johnson problem
    // double eps = Poisson<D>::epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double r1 = (-1.0+sqrt(1.0+4.0*eps*eps*M_PI*M_PI))/(-2.0*eps);
    // double r2 = (-1.0-sqrt(1.0+4.0*eps*eps*M_PI*M_PI))/(-2.0*eps);
    // res = sin(M_PI*y)*(exp(r1*(x-1.0))-exp(r2*(x-1.0))/(exp(-r1)-exp(-r2)));
    // res = 0.0;
    // Caprio
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double r = sqrt(pow(x,2.0)+pow(y,2.0));
    // if(r > 2.0 && r < 4.0)
    //   res = 1.0;
    // else 
    //   res = 0.0;

  }

  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<1, SCAL> & bcstate) {
    EvalInitial(sparam, bcstate);
    // bcstate = 0.0;
    // Internal layer
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x == 0 && y < 0.5)
    //   bcstate = 0.0;
    // else if(x == 0 && y > 0.5)
    //   bcstate = 1.0;
    // else
    //   bcstate = 0.0;
    // double alpha = Poisson<D>::alpha;
    // if(x == 0)
    //   bcstate = 0.5+atan(alpha*(y-0.5))/M_PI;
    // else
      // bcstate = 0.0;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // if(x == 0.0)
    // {
    //   bcstate(0) = 1.0;
    // }
    // else if(x == 4.0 || y == 0.0)
    // {
    //   bcstate(0) = state(0);
    // }
    // else
    // {
    //   bcstate(0) = 0.0;
    // }
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double tol = 1e-13;
    // if(y < tol && (x > 1.0/3.0 && x < 2.0/3.0))
    // {
    //   bcstate(0) = 1.0;
    // }
    // else if(x < tol)
    // {
    //   bcstate(0) = state(0);
    // }
    // else
    // {
    //   bcstate(0) = 0.0;
    // }
  }

  // template <typename SCAL>
  // static void EvalBdryGradient(int bcnr, Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<1, D, SCAL> & bcgrad) {
  //     // double x = sparam.pos(0);
  //     // double y = sparam.pos(1);
  //     // if(x == 4.0 || y == 0.0)
  //     //   bcgrad = 0.0;
  //     // else 
  //       bcgrad = grad;
  // }  

  template <typename SCAL>
  static void EvalSource(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {

    // double eps = Poisson<D>::epsilon;
    // // // // // Source for boundary layer test case
    // double temp = 0.0;
    // for (int dd = 0; dd < D; ++dd)
    // {
    //   double res_loc = 1.0; 
    //   for(int ll = 0; ll < D-1; ++ll)
    //   {
    //     int index = (dd+ll+1)%D;
    //     double pos = sparam.pos(index);
    //     res_loc *= (pos+(exp(pos/eps)-1.0)/(1.0-exp(1.0/eps)));
    //   }
    //   temp += res_loc;
    // }
    // res = temp;
    // Eriksson–Johnson problem
    // double eps = Poisson<D>::epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double r1 = (-1.0+sqrt(1.0+4.0*eps*eps*M_PI*M_PI))/(-2.0*eps);
    // double r2 = (-1.0-sqrt(1.0+4.0*eps*eps*M_PI*M_PI))/(-2.0*eps);
    // res = -eps*(sin(M_PI*y)*((r1*r1)*exp(r1*(x-1.0))-((r2*r2)*exp(r2*(x-1.0)))/(exp(-r1)-exp(-r2)))-(M_PI*M_PI)*sin(M_PI*y)*(exp(r1*(x-1.0))-exp(r2*(x-1.0))/(exp(-r1)-exp(-r2))))+sin(M_PI*y)*(r1*exp(r1*(x-1.0))-(r2*exp(r2*(x-1.0)))/(exp(-r1)-exp(-r2)));
  	// Anisotropic boundary layer

    double eps = Poisson<D>::epsilon;
    Vec<D> dir;
    GetDirection(sparam, dir);
    double beta1 = dir(0);
    double beta2 = dir(1);
    double x = sparam.pos(0);
    double y = sparam.pos(1);
  	res = eps*(((beta2*beta2)*1.0/(eps*eps)*exp((beta2*y)/eps)*(x-(exp((beta1*x)/eps)-1.0)/(exp(beta1/eps)-1.0)))/(exp(beta2/eps)-1.0)+((beta1*beta1)*1.0/(eps*eps)*exp((beta1*x)/eps)*(y-(exp((beta2*y)/eps)-1.0)/(exp(beta2/eps)-1.0)))/(exp(beta1/eps)-1.0))-beta2*(x-(exp((beta1*x)/eps)-1.0)/(exp(beta1/eps)-1.0))*((beta2*exp((beta2*y)/eps))/(eps*(exp(beta2/eps)-1.0))-1.0)-beta1*(y-(exp((beta2*y)/eps)-1.0)/(exp(beta2/eps)-1.0))*((beta1*exp((beta1*x)/eps))/(eps*(exp(beta1/eps)-1.0))-1.0);
    // Internal layer
    // res = 0.0;
    // Square jump
    // double alpha = Poisson<D>::alpha;
    // double eps = Poisson<D>::epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double x1 = 1.0/3.0;
    // double x2 = 2.0/3.0;
    // double y1 = x1;
    // double y2 = x2;
    // res = (atan(alpha*(x-x1))-atan(alpha*(x-x2)))*(alpha/((alpha*alpha)*pow(y-y1,2.0)+1.0)-alpha/((alpha*alpha)*pow(y-y2,2.0)+1.0))+(atan(alpha*(y-y1))-atan(alpha*(y-y2)))*(alpha/((alpha*alpha)*pow(x-x1,2.0)+1.0)-alpha/((alpha*alpha)*pow(x-x2,2.0)+1.0))+eps*((atan(alpha*(y-y1))-atan(alpha*(y-y2)))*((alpha*alpha*alpha)*(x*2.0-x1*2.0)*1.0/pow((alpha*alpha)*pow(x-x1,2.0)+1.0,2.0)-(alpha*alpha*alpha)*(x*2.0-x2*2.0)*1.0/pow((alpha*alpha)*pow(x-x2,2.0)+1.0,2.0))+(atan(alpha*(x-x1))-atan(alpha*(x-x2)))*((alpha*alpha*alpha)*(y*2.0-y1*2.0)*1.0/pow((alpha*alpha)*pow(y-y1,2.0)+1.0,2.0)-(alpha*alpha*alpha)*(y*2.0-y2*2.0)*1.0/pow((alpha*alpha)*pow(y-y2,2.0)+1.0,2.0)));
    // res = 0.0;
  }

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
    // w = 0.0;
    // q = 0.0;
    // double eps = Poisson<D>::epsilon;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // q(0,0) = -(exp(x/eps)/(eps*(exp(1.0/eps)-1.0))-1.0)*(y-(exp(y/eps)-1.0)/(exp(1.0/eps)-1.0));
    // q(0,1) = -(exp(y/eps)/(eps*(exp(1.0/eps)-1.0))-1.0)*(x-(exp(x/eps)-1.0)/(exp(1.0/eps)-1.0));
    

    // Anisotropic boundary layer
    double eps = Poisson<D>::epsilon;
    Vec<D> dir;
    GetDirection(sparam, dir);
    double beta1 = dir(0);
    double beta2 = dir(1);
    double x = sparam.pos(0);
    double y = sparam.pos(1);
    q(0, 0) = (y + (exp(beta2*y/eps) - 1)/(1 - exp(beta2/eps)))*(beta1*exp(beta1*x/eps)/(eps*(1 - exp(beta1/eps))) + 1);
    q(0, 1) = (x + (exp(beta1*x/eps) - 1)/(1 - exp(beta1/eps)))*(beta2*exp(beta2*y/eps)/(eps*(1 - exp(beta2/eps))) + 1);
// 
    // // // // // w(0) = 1.0;
    // // // // // for (int dd = 0; dd < D; ++dd)
    // // // // // {
    // // // // //   double pos = sparam.pos(dd);
    // // // // //   w(0) *= (pos+(exp(pos/eps)-1.0)/(1.0-exp(1.0/eps)));
    // // // // // }
    // // // // // //Written only for 2D
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // // // double z = sparam.pos(2);
    // double ep = 1.0/eps;
    
    // // // double u = x + (exp(x*ep) - 1.0)/(1.0-exp(ep));
    // // // double u_x = 1.0 + ep * exp(x*ep)/ (1.0-exp(ep));

    // // // double v = y + (exp(y*ep) - 1.0)/(1.0-exp(ep));
    // // // double v_y = 1.0 + ep * exp(y*ep)/ (1.0-exp(ep));

    // // // double t = z + (exp(z*ep) - 1.0)/(1.0-exp(ep));
    // // // double t_z = 1.0 + ep * exp(z*ep)/ (1.0-exp(ep));

    // // // q(0,0) = u_x * v * t;
    // // // q(0,1) = u * v_y * t;
    // // // q(0,2) = u * v * t_z;
    // double x_new = sparam.pos(0)-0.5;
    // double y_new = sparam.pos(1)-0.5;
    // w(0) = pow(x_new, 2.0)+pow(y_new,2.0)+pow(x_new,3.0)/10.0+pow(y_new,4.0)/100000.0;
    // q = 0.0;
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);
    // double dist = sqrt(pow(x,2.0)+pow(y,2.0));
    // if(dist > 1.0/3.0 && dist < 2.0/3.0)
    //   w = 1.0;
    // else
    //   w = 0.0;
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

    Vec<1, SCAL> bcstate(0.);
    EvalBdryState(bcnr, state, sparam, bcstate);

    // Mat<1, D, SCAL> bcgrad(0.);
    // EvalBdryGradient(bcnr, state, grad, sparam, bcgrad);

    Mat<1, D, SCAL> flux(0.);
    EvalDiffFlux(bcstate, grad, sparam, flux);    
    res = flux * sparam.normal;
  }
/*!
 * The function returns the function whose derivative we want to compute
 * @param[in] State - Contains the type of boundary. Size = 1
 * @param[in] sparam - Contains the position \f$(x,y)\f$ where we want to evaluate the boundary
 * flux. Size=D
 * @param[out] dervar - Contains the variables which will be reconstructed. Size = COMP doubles
 */
  template <typename SCAL>
  static void EvalDerVar(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<NumDerVar, SCAL> & dervar) {
    // testing
    Vec<1> w(0.);
    Mat<1, D> q(0.);
    EvalAnalyticSol(w, q, sparam);

    //testing ends
    dervar(0) = state(0)-w(0);

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
    // L2 error is the target
    Vec<1> w_ex(0.);
    EvalInitial(sparam, w_ex);

    // res = state(0)*temp;
    res = (state(0)-w_ex(0))*(state(0)-w_ex(0));

  }

  static void LoadParameters(shared_ptr<PDE> apde) {
    Advection<D>::LoadParameters(apde);
    Poisson<D>::LoadParameters(apde);
  }
  // static void GetBCName(int bcnr, stringstream & oss) {
  //   oss << "Interior";
  // }
};


#endif
