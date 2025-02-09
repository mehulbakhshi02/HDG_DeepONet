#ifndef KOMEGA_H
#define KOMEGA_H

template <int D>
class CompressibleKOmega : public CompressibleNavierStokes<D> {

public:

  // Characterization of the model
  static const int  Components = D+4;
  static const bool Convection = true;
  static const bool Diffusion  = true;
  static const bool Source     = true;

  static const InitType inittype = IT_Constant;

  static const int NumConstrConsVar  = 1;
  static const int NumConstrPrimVar  = 1;
  static const int NumDerVar         = 4;
  static const int NumBdryFluxWeight = 2;

  // User-defined parameters
  static double prandtl_t;
  static double Tu;
  static double Evr;
  
  // Dependent variables
  static double k_infty;
  static double w_infty;

  // Independent variables
  static const double sigma_k;
  static const double sigma_w;

  static const double alpha_w;
  static const double beta_k;
  static const double beta_w;

  static const double alpha0;
  static const double alphas0;

  static const double Rb;
  static const double Rk;
  static const double Rw;

  using CompressibleNavierStokes<D>::mach;
  using CompressibleNavierStokes<D>::gamma;
  using CompressibleNavierStokes<D>::c_infty;
  using CompressibleNavierStokes<D>::rho_infty;
  using CompressibleNavierStokes<D>::p_infty;
  using CompressibleNavierStokes<D>::prandtl;
  using CompressibleNavierStokes<D>::angle;
  using CompressibleNavierStokes<D>::cvTtot;
  using CompressibleNavierStokes<D>::ptot;
  using CompressibleNavierStokes<D>::H_infty;
  
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
      case 3:
        name = "Evr";
        break;
      default:
        cout << "Derived variable #" << dervar << " not found." << endl;
        exit(0);        
    }
  }  

  template <typename SCAL>
  static void EvalDerVar(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<NumDerVar, SCAL> & dervar) {
    // Mach, pressure, entropy

    SCAL rho       = state(0);
    Vec<D, SCAL> m = state.Rows(1, D+1);
    SCAL rhoE      = state(D+1);
    SCAL rhok      = state(D+2);

    Vec<D, SCAL> U = 1. / rho * m;

    SCAL U2 = 0.;
    for (int dd = 0; dd < D; ++dd)
      U2 += U(dd) * U(dd);

    SCAL p = (gamma - 1.) * (rhoE - 0.5 * rho * U2 - rhok);

    SCAL c2 = gamma * p / rho;
    SCAL Ma = sqrt(U2 / c2);
    SCAL s  = p * pow(rho, -gamma);

    double visc = CompressibleNavierStokes<D>::EvalPhysVisc(state(0), dervar(1));

    dervar(0) = Ma;
    dervar(1) = p;
    dervar(2) = s;
    dervar(3) = state(D+2) / (exp(state(D+3) / state(0)) * visc);
  }

  template <typename SCAL>
  static int GetConstrConsVar(int i) {
    return 0;
  }

  template <typename SCAL>
  static SCAL EvalConstrPrimVar(int i, Vec<D+4, SCAL> & state, SpatialParams<D> & sparam) {

    SCAL rho       = state(0);
    Vec<D, SCAL> m = state.Rows(1, D+1);
    SCAL rhoE      = state(D+1);
    SCAL rhok      = state(D+2);

    Vec<D, SCAL> U = 1. / rho * m;

    SCAL U2 = 0.;
    for (int dd = 0; dd < D; ++dd)
      U2 += U(dd) * U(dd);

    return (gamma - 1.) * (rhoE - 0.5 * rho * U2 - rhok) + 2. / 3. * rhok;
  }

  template <typename SCAL>
  static void EvalInitial(SpatialParams<D> & sparam, Vec<D+4, SCAL> & res) {

    Vec<D+2, SCAL> state_ns;
    CompressibleNavierStokes<D>::EvalInitial(sparam, state_ns);

    res.Rows(0, D+2) = state_ns;
    res(D+2)         = res(0) * k_infty;
    res(D+3)         = res(0) * w_infty;
  }  

  static void GetFilename(stringstream & oss) {
    CompressibleNavierStokes<D>::GetFilename(oss);
    oss << "-Tu" << Tu << "-Evr" << Evr;
  }

  template <typename SCAL>
  static void EvalConvFlux(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Mat<D+4, D, SCAL> & res);

  template <typename SCAL>
  static void EvalMaxEigenvalue(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, SCAL & res);
  
  template <typename SCAL>
  static void EvalConvEigenSystem(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, double fix, Vec<D+4, SCAL> & eig, Mat<D+4, D+4, SCAL> & eigvr, Mat<D+4, D+4, SCAL> & eigvl);

  template <typename SCAL>  
  static void EvalDiffFlux(Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<D+4, D, SCAL> & res);

  template <typename SCAL>
  static void ApplyK(Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, Mat<D+4, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<D+4, D, SCAL> & res);

  template <typename SCAL>
  static SCAL EvalEddyVisc(SCAL & rho, SCAL & rhok, SCAL & omega_r, SCAL & visc, bool includek);

  template <typename SCAL>
  static SCAL EvalLimitedOmega(SCAL & rho, SCAL & rhok, SCAL & rhowhat, SCAL & visc, Mat<D, D, SCAL> & S);
  
  template <typename SCAL>  
  static void EvalSource(Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<D+4, SCAL> & res);

  template <typename SCAL>
  static void EvalFarfield(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate);

  template <typename SCAL>
  static SCAL EvalWallOmega(SCAL rho, SCAL p);
  
  template <typename SCAL>
  static void EvalNoslipwallAdiabatic(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalNoslipwallIsothermal(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate);
  
  template <typename SCAL>
  static void EvalInflow(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalOutflow(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate);
  
  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalBdryGradient(int bcnr, Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<D+4, D, SCAL> & bcgrad);

  template <typename SCAL>
  static void EvalBdryConvFlux(int bcnr, Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & res) {

    Vec<D+4, SCAL> bcstate;
    EvalBdryState(bcnr, state, sparam, bcstate);

    Mat<D+4, D, SCAL> flux;
    EvalConvFlux(bcstate, sparam, flux);
    res = flux * sparam.normal;
  }

  template <typename SCAL>
  static void EvalBdryDiffFlux(int bcnr, Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<D+4, SCAL> & res) {

    Vec<D+4, SCAL> bcstate;
    EvalBdryState(bcnr, state, sparam, bcstate);

    Mat<D+4, D, SCAL> bcgrad;
    EvalBdryGradient(bcnr, state, grad, sparam, bcgrad);

    Mat<D+4, D, SCAL> flux;    
    if (bcnr == 1) {
      Vec<D+4, SCAL> ghoststate(2. * bcstate - state);
      Mat<D+4, D, SCAL> flux1, flux2;
      EvalDiffFlux(state, grad, sparam, flux1);
      EvalDiffFlux(ghoststate, grad, sparam, flux2);
      flux = 0.5 * (flux1 + flux2);
    } else {
      EvalDiffFlux(bcstate, bcgrad, sparam, flux);
    }
    
    res = flux * sparam.normal;

    if (bcnr == 1) {
      res(D+1) = res(D+2);
      //      res(D+2) = 0.;
    }
  }  

  static void EvalBdryFluxWeight(int bcnr, SpatialParams<D> & sparam, Mat<NumBdryFluxWeight, D+4> & res) {

    Mat<NumBdryFluxWeight, D+2> res_ns;
    CompressibleNavierStokes<D>::EvalBdryFluxWeight(bcnr, sparam, res_ns);
    res = 0.;
    res.Cols(0, D+2) = res_ns;
  }

  static void LoadParameters(PDE & apde);

};

// Default values
template<int D> double CompressibleKOmega<D>::prandtl_t     = 0.9;
template<int D> double CompressibleKOmega<D>::Tu            = 0.0004;
template<int D> double CompressibleKOmega<D>::Evr           = 0.009;

template<int D> double CompressibleKOmega<D>::k_infty       = 0.;
template<int D> double CompressibleKOmega<D>::w_infty       = 0.;

template<int D> const double CompressibleKOmega<D>::sigma_k = 0.5;
template<int D> const double CompressibleKOmega<D>::sigma_w = 0.5;

template<int D> const double CompressibleKOmega<D>::alpha_w = 13./25.;
template<int D> const double CompressibleKOmega<D>::beta_k  = 0.09;
template<int D> const double CompressibleKOmega<D>::beta_w  = 9./125.;

template<int D> const double CompressibleKOmega<D>::alpha0  = 1./9.;
template<int D> const double CompressibleKOmega<D>::alphas0 = 0.0708/3.;

template<int D> const double CompressibleKOmega<D>::Rb      = 8.;
template<int D> const double CompressibleKOmega<D>::Rk      = 6.;
template<int D> const double CompressibleKOmega<D>::Rw      = 2.61;

#include "komegaconvflux.h"
#include "komegaviscflux.h"
#include "komegasource.h"
#include "bdrystate.h"
#include "komegafarfield.h"
#include "komeganoslipwall.h"
#include "komegainflow.h"
#include "komegaoutflow.h"

#endif
