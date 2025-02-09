#ifndef EULER_H
#define EULER_H

#include "../dummymodel.h"

template <int D, int COMP = D+2>
class CompressibleEuler : virtual public DummyModel<D, COMP> {

 public:

  // Characterization of the model
  static const int  Components = COMP;
  static const bool Convection = true;
  static const bool Diffusion  = false;
  static const bool Source     = false;

  static const InitType inittype = IT_Constant; // IT_Function;

  static const int NumDerVar = 3; // Mach, pressure, entropy
  static const int NumBdryFluxWeight = 1;
  static const int NumBdryCoeffs     = 1;
  static const int NumVolFunctionals = 0;

  // User-defined parameters
  static double mach;
  static Vec<D-1> angle;
  static double gamma;
  static double chord;

  // Dependent variables
  static double rho_infty;
  static Vec<D> v_infty;
  static double E_infty;
  static double p_infty;
  static double c_infty;
  static double C_infty;
  static double H_infty;
  static double cvTtot;
  static double ptot;
  static Vec<COMP> w_infty;
  
  template <typename SCAL>
  static void EvalInitial(SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {
    /*** From freestream conditions ***/
    res = w_infty;

    /*** Isentropic Vortex (time-dependent) ***/
    // double x = sparam.pos(0);
    // double y = sparam.pos(1);

    // double epsilon = 5.;
    // double rSqr = pow(x,2.) + pow(y,2.);

    // double deltaT = -(gamma - 1.0)/(8.*gamma*pow(M_PI,2.))*pow(epsilon,2.)*exp(1. - rSqr);
    // double deltaU = -y*epsilon/(2.*M_PI)*exp(0.5*(1.-rSqr));
    // double deltaV =  x*epsilon/(2.*M_PI)*exp(0.5*(1.-rSqr));

    // double T   = 1. + deltaT;
    // double rho = pow(T,1./(gamma-1.));
    // double p   = pow(T,gamma/(gamma-1.));
    // double u   = 1. + deltaU;
    // double v   = 1. + deltaV;

    // res(0) = rho;
    // res(1) = rho*u;
    // res(2) = rho*v;
    // res(3) = p/(gamma - 1.0) + 0.5*rho*(pow(u,2.0) + pow(v,2.0));
  }

  /*** For Isentropic Vortex (time-dependent)'*/
  template <typename SCAL>
  static void EvalAnalyticSol(Vec<COMP> & w, Mat<COMP, D, SCAL> & q, SpatialParams<D> & sparam) {
    EvalInitial(sparam, w);
  }

  static void GetFilename(stringstream & oss) {

    oss << "-Ma" << mach << "-Al" << angle(0) / M_PI * 180;
    if (D == 3)
      oss << "-Be" << angle(1) / M_PI * 180;
  }

  static void GetDerVarName(int dervar, string & name) {
    switch (dervar) {
      case 0:
        name = "Ma";
        break;
      case 1:
        name = "p";
        break;
      case 2:
        name = "s";
        break;
      default:
        cout << "Derived variable #" << dervar << " not found." << endl;
        exit(0);        
    }
  }
  
  template <typename SCAL>
  static void EvalDerVar(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<NumDerVar, SCAL> & dervar) {

    SCAL rho       = state(0);
    Vec<D, SCAL> m = state.Rows(1, D+1);
    SCAL E         = state(D+1);

    Vec<D, SCAL> U = 1. / rho * m;

    SCAL U2 = 0.;
    for (int dd = 0; dd < D; ++dd)
      U2 += U(dd) * U(dd);

    SCAL p = (gamma - 1.) * (E - 0.5 * rho * U2);

    SCAL c2 = gamma * p / rho;
    SCAL Ma = sqrt(U2 / c2);
    SCAL GAM = SCAL(gamma);
    SCAL s  = p * pow(rho, -GAM);

    dervar(0) = Ma;
    dervar(1) = p;
    dervar(2) = s;
    
  }

  template <typename SCAL>
  static void EvalPrim(Vec<COMP, SCAL> & state, Vec<D, SCAL> & U, SCAL & p) {

    SCAL rho       = state(0);
    Vec<D, SCAL> m = state.Rows(1, D+1);
    SCAL E         = state(D+1);

    U = 1. / rho * m;

    SCAL U2 = 0.;
    for (int dd = 0; dd < D; ++dd)
      U2 += U(dd) * U(dd);

    p = (gamma - 1.) * (E - 0.5 * rho * U2);
  }

  template <typename SCAL>
  static int GetConstrConsVar(int i) {
    return 0;
  }

  template <typename SCAL>
  static SCAL EvalConstrPrimVar(int i, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam) {

    SCAL p;
    Vec<D, SCAL> U;

    EvalPrim(state, U, p);

    return p;
  }  
  
  template <typename SCAL>
  static void EvalConvFlux(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res) {

    SCAL p;
    Vec<D, SCAL> U;
    EvalPrim(state, U, p);

    res = state * Trans(U);
    res.Rows(1, D+1) += p * Id<D>();
    res.Row(D+1)     += p * U;
  }

  template <typename SCAL>
  static void EvalMaxEigenvalue(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, SCAL & res);
  
  template <typename SCAL>
    static void EvalConvEigenSystem(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, double fix, Vec<COMP, SCAL> & eig, Mat<COMP, COMP, SCAL> & eigvr, Mat<COMP, COMP, SCAL> & eigvl);

  template <typename SCAL>
  static void EvalFarfield(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalSlipwall(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalInflow(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalOutflow(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);
  
  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalBdryConvFlux(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

    Vec<COMP, SCAL> bcstate(0.);
    EvalBdryState(bcnr, state, sparam, bcstate);

    Mat<COMP, D, SCAL> flux(0.);
    EvalConvFlux(bcstate, sparam, flux);
    res = flux * sparam.normal;
  }

  template <typename SCAL>
  static void EvalBdryCoefficients(int bcnr, Vec<COMP, SCAL> & fcn, Vec<COMP, SCAL> & fvn, SpatialParams<D> & sparam, Vec<NumBdryCoeffs, SCAL> & res);
  
  static void EvalBdryFluxWeight(int bcnr, SpatialParams<D> & sparam, Mat<NumBdryFluxWeight, COMP> & res);

  template <typename SCAL>
  static void EvalVolFunctionals(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<NumVolFunctionals, SCAL> & res) {

    Vec<NumDerVar, SCAL> dervar;
    EvalDerVar(state, sparam, dervar);
    // SCAL power = 2.0;
    // res = pow(dervar(2) - 1., power);
    if(sparam.pos[0]>=8. && sparam.pos[0]<=11.)
    {
      double yc = -10.0;
      double y = sparam.pos[1];
      double dist = sqrt(pow(y-yc,2.0));
      double delta = 1000.;
      res = exp(-delta*dist*dist)*(dervar(1)-p_infty)/p_infty*0.5;
    }
  }  

  static void LoadParameters(shared_ptr<PDE> apde);


  static void GetBCName(int bcnr, stringstream & oss);

};

// Default values
template<int D, int COMP> double    CompressibleEuler<D, COMP>::mach      = 0.0; // no default value for that
template<int D, int COMP> Vec<D-1>  CompressibleEuler<D, COMP>::angle     = 0.0;
template<int D, int COMP> double    CompressibleEuler<D, COMP>::gamma     = 1.4;
template<int D, int COMP> double    CompressibleEuler<D, COMP>::chord     = 1.;

template<int D, int COMP> double    CompressibleEuler<D, COMP>::rho_infty = 1.;
template<int D, int COMP> double    CompressibleEuler<D, COMP>::p_infty   = 1.;

// The following will be initialized in LoadParameters
template<int D, int COMP> double    CompressibleEuler<D, COMP>::c_infty   = 0.;
template<int D, int COMP> Vec<D>    CompressibleEuler<D, COMP>::v_infty   = 0.;
template<int D, int COMP> double    CompressibleEuler<D, COMP>::E_infty   = 0.;
template<int D, int COMP> double    CompressibleEuler<D, COMP>::C_infty   = 0.;
template<int D, int COMP> double    CompressibleEuler<D, COMP>::H_infty   = 0.;
template<int D, int COMP> double    CompressibleEuler<D, COMP>::cvTtot    = 0.;
template<int D, int COMP> double    CompressibleEuler<D, COMP>::ptot      = 0.;
template<int D, int COMP> Vec<COMP> CompressibleEuler<D, COMP>::w_infty = 0.;

#include "eigensystem.h"
#include "farfield.h"
#include "slipwall.h"
#include "inflow.h"
#include "outflow.h"
#include "bdrystate.h"
#include "bdryfluxweight.h"
#include "loadparam.h"

#endif
