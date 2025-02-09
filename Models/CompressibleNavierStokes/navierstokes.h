#ifndef NAVIERSTOKES_H
#define NAVIERSTOKES_H

#include "../CompressibleEuler/euler.h"

template <int D, int COMP=D+2>
class CompressibleNavierStokes : public CompressibleEuler<D, COMP> {

public:

  // Characterization of the model
  static const int  Components = COMP;
  static const bool Convection = true;
  static const bool Diffusion  = true;
  static const bool Source     = false;

  static const InitType inittype = IT_Constant;

  static const int NumBdryCoeffs = CompressibleEuler<D, COMP>::NumBdryCoeffs + 1;

  static const int NumParam = 3;

  // User-defined parameters
  static double reynolds;
  static double prandtl;
  static double T0;

  static double stab_visc;

  // Dependent variables
  static double amu;
  static double rr;
  static double mu0;

  // Independent variables
  static double suth_a1;
  static double suth_a2;

  using CompressibleEuler<D, COMP>::mach;
  using CompressibleEuler<D, COMP>::gamma;
  using CompressibleEuler<D, COMP>::chord;
  using CompressibleEuler<D, COMP>::rho_infty;
  using CompressibleEuler<D, COMP>::c_infty;

  using CompressibleEuler<D, COMP>::NumDerVar;
  using CompressibleEuler<D, COMP>::NumBdryFluxWeight;
  
  using CompressibleEuler<D, COMP>::EvalInitial;
  using CompressibleEuler<D, COMP>::GetDerVarName;
  using CompressibleEuler<D, COMP>::EvalDerVar;
  using CompressibleEuler<D, COMP>::EvalConvFlux;
  using CompressibleEuler<D, COMP>::EvalFarfield;
  using CompressibleEuler<D, COMP>::EvalSlipwall;
  using CompressibleEuler<D, COMP>::EvalInflow;
  using CompressibleEuler<D, COMP>::EvalOutflow;
  using CompressibleEuler<D, COMP>::EvalBdryFluxWeight;

  static void GetParameters(Vec<NumParam> & param) {

    static double aoa = CompressibleEuler<D, COMP>::angle(0)* 180. / M_PI;
    param(0) = aoa;
    param(1) = mach;
    param(2) = reynolds;

  }
  
  static void GetFilename(stringstream & oss) {

    CompressibleEuler<D, COMP>::GetFilename(oss);

    oss << "-Re" << reynolds;
        
  }

  template <typename SCAL>
  static int GetConstrConsVar(int i) {
    return 0;
  }

  template <typename SCAL>
  static SCAL EvalConstrPrimVar(int i, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam) {

    SCAL rho(state(0));
    Vec<D, SCAL> m(state.Rows(1, D+1));
    SCAL rhoE(state(D+1));

    Vec<D, SCAL> U(1. / rho * m);

    SCAL U2 = 0.;
    for (int dd = 0; dd < D; ++dd)
      U2 += U(dd) * U(dd);

    return (gamma - 1.) * (rhoE - 0.5 * rho * U2);
  }
  
  
  // Helper routines
  template <typename SCAL>
  static Mat<D, D, SCAL> EvalStrainRate(Mat<D, D, SCAL> & gradU);

  template <typename SCAL>
  static SCAL EvalPhysVisc(SCAL & rho, SCAL & p);
  
  template <typename SCAL>
  static void EvalPrim(Vec<COMP, SCAL> & state, Vec<D, SCAL> & U, SCAL & p, SCAL & T);
  
  template <typename SCAL>
  static void EvalPrim(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, Vec<D, SCAL> & U, SCAL & p, SCAL & T, Mat<D, D, SCAL> & gradU, Vec<D, SCAL> & gradp, Vec<D, SCAL> & gradT);

  template <typename SCAL>
  static void EvalDiffFlux(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res) {

     // Primitive variables and their gradients
    SCAL p(0.), T(0.);
    Vec<D, SCAL> U(0.), gradp(0.), gradT(0.);
    Mat<D, D, SCAL> gradU(0.);
    EvalPrim(state, grad, U, p, T, gradU, gradp, gradT);

    SCAL rho(state(0));
    SCAL visc(EvalPhysVisc(rho, p));
    Mat<D, D, SCAL> S(EvalStrainRate(gradU));
    Mat<D, D, SCAL> tau(visc * S);

    res = 0.;
    res.Rows(1, D+1) = tau;                                 // Momentum
    res.Row(D+1)     = tau * U + visc / prandtl * gradT;    // Energy  
  }

  template <typename SCAL>
  static void ApplyK(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, Mat<COMP, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res);

  template <typename SCAL>
  static void EvalNoslipwallAdiabatic(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);  

  template <typename SCAL>
  static void EvalNoslipwallIsothermal(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);  
  
  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalBdryGradient(int bcnr, Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & bcgrad);    

  template <typename SCAL>
  static void EvalBdryConvFlux(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

    Vec<COMP, SCAL> bcstate(0.);
    EvalBdryState(bcnr, state, sparam, bcstate);

    Mat<COMP, D, SCAL> flux(0.);
    EvalConvFlux(bcstate, sparam, flux);
    res = flux * sparam.normal;
  }  
  
  template <typename SCAL>
  static void EvalBdryDiffFlux(int bcnr, Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

    Vec<COMP, SCAL> bcstate(0.);
    EvalBdryState(bcnr, state, sparam, bcstate);

    Mat<COMP, D, SCAL> bcgrad(0.);
    EvalBdryGradient(bcnr, state, grad, sparam, bcgrad);

    Mat<COMP, D, SCAL> flux(0.);
    EvalDiffFlux(bcstate, bcgrad, sparam, flux);    
    res = flux * sparam.normal;

    // HACK
    if (bcnr == 1) // Adiabatic no-slip wall
      res(D+1) = 0.;
  }

  template <typename SCAL>
  static void EvalBdryCoefficients(int bcnr, Vec<COMP, SCAL> & fcn, Vec<COMP, SCAL> & fvn, SpatialParams<D> & sparam, Vec<NumBdryCoeffs, SCAL> & res) {

    const int n_euler = CompressibleEuler<D, COMP>::NumBdryCoeffs;
    
    Vec<n_euler, SCAL> res_euler;
    CompressibleEuler<D, COMP>::EvalBdryCoefficients(bcnr, fcn, fvn, sparam, res_euler);
    res.Rows(0, n_euler) = res_euler;

    Vec<D> t;
    if (D == 2) {
      t(0) = -sparam.normal(1);
      t(1) = sparam.normal(0);
    }
    
    SCAL f(0.);
    for (int dd = 0; dd < D; ++dd)
      f += fvn(dd+1) * t(dd);

    res(n_euler) = 2. * f / (rho_infty * pow(mach * c_infty, 2.));
  }
  
  static void LoadParameters(shared_ptr<PDE> apde);

  static void GetBCName(int bcnr, stringstream & oss);
};

// Default values
template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::reynolds = 0.; // no default value for that (has to be provided by the user)
template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::prandtl  = 0.72;
template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::T0       = 288.15;

template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::stab_visc = 0.;

template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::suth_a1  = 1.461e-6;
template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::suth_a2  = 110.3;

template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::amu      = 0.;
template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::rr       = 0.;
template<int D, int COMP> double CompressibleNavierStokes<D, COMP>::mu0      = 0.;

#include "helper.h"
#include "loadparam.h"
#include "applyk.h"
#include "bdrystate.h"
#include "noslipwall.h"

#endif
