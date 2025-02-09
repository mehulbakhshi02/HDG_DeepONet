#ifndef SA_H
#define SA_H

template <int D, int COMP=D+3>
class CompressibleSpalartAllmaras : public CompressibleNavierStokes<D, COMP> {

public:

  // Characterization of the model
  static const int  Components = COMP;
  static const bool Convection = true;
  static const bool Diffusion  = true;
  static const bool Source     = true;

  static const InitType inittype = IT_Constant;

  static const int NumDerVar = CompressibleNavierStokes<D, COMP>::NumDerVar+1;
  static const int NumVolFunctionals = 0;  

  // User-defined parameters
  static double prandtl_t;
  static double nu_t;
  
  // Dependent variables
  static double nu_t_infty;

  // Independent variables
  static const double cb1;
  static const double cb2;
  static const double sigma;
  static const double kappa;
  static const double cw1;
  static const double cw2;
  static const double cw3;
  // Incorporating new wall correction
  static const double cw4;
  static const double cw5;
  static const double cv1;
  static const double cv2;
  static const double cv3;
  static const double rlim;
  static const double cn1;
  static const double ct3;
  static const double ct4;
  
  using CompressibleNavierStokes<D, COMP>::gamma;
  using CompressibleNavierStokes<D, COMP>::amu;
  using CompressibleNavierStokes<D, COMP>::prandtl;
  using CompressibleNavierStokes<D, COMP>::NumBdryFluxWeight;
  using CompressibleNavierStokes<D, COMP>::NumBdryCoeffs;
  using CompressibleNavierStokes<D, COMP>::rho_infty;
  using CompressibleNavierStokes<D, COMP>::v_infty;
  using CompressibleNavierStokes<D, COMP>::E_infty;
  using CompressibleNavierStokes<D, COMP>::p_infty;
  using CompressibleNavierStokes<D, COMP>::w_infty;

  using CompressibleNavierStokes<D, COMP>::EvalInitial;
  using CompressibleNavierStokes<D, COMP>::EvalConvFlux;
  using CompressibleNavierStokes<D, COMP>::EvalMaxEigenvalue;
  using CompressibleNavierStokes<D, COMP>::EvalConvEigenSystem;
  using CompressibleNavierStokes<D, COMP>::EvalBdryGradient;
  using CompressibleNavierStokes<D, COMP>::EvalFarfield;
  using CompressibleNavierStokes<D, COMP>::EvalInflow;
  using CompressibleNavierStokes<D, COMP>::EvalOutflow;  
  
  static void GetDerVarName(int dervar, string & name) {
    switch (dervar) {
      case 0: case 1: case 2:
        CompressibleNavierStokes<D, COMP>::GetDerVarName(dervar, name);
        break;
      case 3:
        name = "NuSA";
        break;
      default:
        cout << "Derived variable #" << dervar << " not found." << endl;
        exit(0);        
    }
  }  

  template <typename SCAL>
  static void EvalDerVar(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<NumDerVar, SCAL> & dervar) {
    // Mach, pressure, entropy
    const int numdervar_ns = CompressibleNavierStokes<D>::NumDerVar;
    Vec<numdervar_ns, SCAL> dervar_ns;
    CompressibleNavierStokes<D, COMP>::EvalDerVar(state, sparam, dervar_ns);

    dervar.Rows(0, numdervar_ns) = dervar_ns;

    SCAL visc(CompressibleNavierStokes<D, COMP>::EvalPhysVisc(state(0), dervar(1)));
    SCAL chi(amu * state(D+2) / visc);
    SCAL visc_t(EvalEddyVisc(state(D+2), chi));
    dervar(NumDerVar-1) = visc_t / visc;
  }

  static int GetConstrConsVar(int i) {
    return 0;
  }

  template <typename SCAL>
  static SCAL EvalConstrPrimVar(int i, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam) {

    return CompressibleNavierStokes<D, COMP>::EvalConstrPrimVar(i, state, sparam);
  }

  static void GetFilename(stringstream & oss) {
    CompressibleNavierStokes<D, COMP>::GetFilename(oss);
    oss << "-Evr" << nu_t;
  }  

  template <typename SCAL>  
  static void EvalDiffFlux(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res);

  template <typename SCAL>
  static void ApplyK(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, Mat<COMP, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res);

  template <typename SCAL>
  static SCAL EvalEddyVisc(SCAL & mu_t, SCAL & chi) {
    if (mu_t > 0.)
      return amu * mu_t * EvalFv1(chi);
    else
      return 0.;
  }

  template <typename SCAL>
  static SCAL EvalFv1(SCAL & chi) {
    SCAL chi3(pow(chi, 3.));
    return chi3 / (chi3 + pow(cv1, 3.));
  }

  template <typename SCAL>
  static SCAL EvalFn(SCAL & chi) {
    if (chi < 0.) {
      SCAL chi3(pow(chi, 3.));
      return amu * (cn1 + chi3) / (cn1 - chi3);
    } else
      return amu;
  }

  template <typename SCAL>
  static SCAL EvalFv2(SCAL & chi) {
    return 1. - chi / (1. + chi * EvalFv1(chi));
  }

  template <typename SCAL>
  static SCAL EvalFw(SCAL & g) {
    return g * pow((1. + pow(cw3, 6.)) / (pow(g, 6.) + pow(cw3, 6.)), 1./6.);
  }

  template <typename SCAL>  
  static void EvalSource(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res);

  template <typename SCAL>
  static void EvalNoslipwallAdiabatic(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalNoslipwallIsothermal(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);
    
  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate);

  template <typename SCAL>
  static void EvalBdryConvFlux(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

    Vec<COMP, SCAL> bcstate;
    EvalBdryState(bcnr, state, sparam, bcstate);

    Mat<COMP, D, SCAL> flux;
    if (bcnr == 1) {
      Vec<COMP, SCAL> ghoststate(2. * bcstate - state);
      Mat<COMP, D, SCAL> flux1, flux2;
      EvalConvFlux(state, sparam, flux1);
      EvalConvFlux(ghoststate, sparam, flux2);

      Vec<COMP, SCAL> eig(0.);
      Mat<COMP, COMP, SCAL> eigvr(0.), eigvl(0.);

      EvalConvEigenSystem(bcstate, sparam, 0.01, eig, eigvr, eigvl);

      for (int ll = 0; ll < COMP; ++ll)
        if (eig(ll) < 0.)
          eig(ll) *= -1.;

      Vec<D+3, SCAL> stab(eigvl * (bcstate - state));
      for (int ll = 0; ll < COMP; ++ll)
        stab(ll) *= eig(ll);

      res = 0.5 * (flux1 + flux2) * sparam.normal - eigvr * stab;
      
    } else {
      EvalConvFlux(bcstate, sparam, flux);
      res = flux * sparam.normal;
    }
  }

  template <typename SCAL>
  static void EvalBdryDiffFlux(int bcnr, Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

    Vec<COMP, SCAL> bcstate;
    EvalBdryState(bcnr, state, sparam, bcstate);

    Mat<COMP, D, SCAL> bcgrad;
    EvalBdryGradient(bcnr, state, grad, sparam, bcgrad);

    Mat<COMP, D, SCAL> flux;    
    if (bcnr == 1) {
      Vec<COMP, SCAL> ghoststate(2. * bcstate - state);
      Mat<COMP, D, SCAL> flux1, flux2;
      EvalDiffFlux(state, grad, sparam, flux1);
      EvalDiffFlux(ghoststate, grad, sparam, flux2);
      flux = 0.5 * (flux1 + flux2);
    } else {
      EvalDiffFlux(bcstate, bcgrad, sparam, flux);
    }
    
    res = flux * sparam.normal;

    if (bcnr == 1)
      res(D+1) = 0.;
  }  

  template <typename SCAL>
  static void EvalBdryCoefficients(int bcnr, Vec<COMP, SCAL> & fcn, Vec<COMP, SCAL> & fvn, SpatialParams<D> & sparam, Vec<NumBdryCoeffs, SCAL> & res) {

    CompressibleNavierStokes<D, COMP>::EvalBdryCoefficients(bcnr, fcn, fvn, sparam, res);
  }
  
  static void EvalBdryFluxWeight(int bcnr, SpatialParams<D> & sparam, Mat<NumBdryFluxWeight, COMP> & res) {

    res = 0.;
    CompressibleNavierStokes<D, COMP>::EvalBdryFluxWeight(bcnr, sparam, res);
  }

  template <typename SCAL>
  static void EvalVolFunctionals(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<NumVolFunctionals, SCAL> & res) {
    res = 0.;
  }

  static void LoadParameters(shared_ptr<PDE> apde) {
    CompressibleNavierStokes<D, COMP>::LoadParameters(apde);

    if (apde->ConstantUsed("eddy_visc_r"))
      nu_t = apde->GetConstant("eddy_visc_r");

    double visc = CompressibleNavierStokes<D, COMP>::EvalPhysVisc(rho_infty, p_infty);

    nu_t_infty = nu_t * visc / (amu * rho_infty);
    w_infty(D+2) = rho_infty * nu_t_infty;
  }
};


template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cb1   = 0.1355;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cb2   = 0.622;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::sigma = 2. / 3.;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::kappa = 0.41;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cw1   = CompressibleSpalartAllmaras<D, COMP>::cb1 / (CompressibleSpalartAllmaras<D, COMP>::kappa * CompressibleSpalartAllmaras<D, COMP>::kappa) + (1. + CompressibleSpalartAllmaras<D, COMP>::cb2) / CompressibleSpalartAllmaras<D, COMP>::sigma;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cw2   = 0.3;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cw3   = 2.;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cw4   = 0.21;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cw5   = 1.5;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cv1   = 7.1;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cv2   = 0.7;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cv3   = 0.9;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::rlim  = 10.;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::cn1   = 16.;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::ct3   = 1.2;
template<int D, int COMP> const double CompressibleSpalartAllmaras<D, COMP>::ct4   = 0.5;

template<int D, int COMP> double CompressibleSpalartAllmaras<D, COMP>::nu_t        = 3.;
template<int D, int COMP> double CompressibleSpalartAllmaras<D, COMP>::nu_t_infty  = 0.;
template<int D, int COMP> double CompressibleSpalartAllmaras<D, COMP>::prandtl_t   = 0.9;

#include "saviscflux.h"
#include "sasource.h"
#include "bdrystate.h"
#include "sanoslipwall.h"

#endif
