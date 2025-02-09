#ifndef DUMMYMODEL_H
#define DUMMYMODEL_H

enum InitType {IT_Zero, IT_Constant, IT_Function, IT_File};

template <int D, int COMP>
class DummyModel {

 public:

  // Characterization of the model
  static const int  Components = COMP;
  static const bool Convection = false;
  static const bool Diffusion  = true;
  static const bool Source     = true;

  static const InitType inittype = IT_Zero;

  static const int NumDerVar = 0;
  static const int NumBdryFluxWeight = 0;
  static const int NumBdryCoeffs     = 0;
  static const int NumVolFunctionals = 0;

  static const int NumParam = 0;

  template <typename SCAL>
  static void EvalInitial(SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {
    res = 0.;
  }


  static void GetParameters(Vec<NumParam> &param) {
  }

  static void GetFilename(stringstream & oss) {
  }

  static void GetDerVarName(int dervar, string & name) {
  }

  template <typename SCAL>
  static void EvalDerVar(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<NumDerVar, SCAL> & dervar) {
  }

  template <typename SCAL>
  static int GetConstrConsVar(int i) {
    return -1;
  }

  template <typename SCAL>
  static SCAL EvalConstrPrimVar(int i, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam) {
    return 0;
  }
  
  template <typename SCAL>
  static void EvalConvFlux(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res) {
    res = 0.;
  }

  template <typename SCAL>
  static void EvalMaxEigenvalue(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, SCAL & res) {
    res = 1.;
  }
  
  template <typename SCAL>
  static void EvalConvEigenSystem(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, double fix, Vec<COMP, SCAL> & eig, Mat<COMP, COMP, SCAL> & eigvr, Mat<COMP, COMP, SCAL> & eigvl) {
    eig = 0.;
    eigvr = 0.;
    eigvl = 0.;
  }

  template <typename SCAL>  
  static void EvalDiffFlux(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res) {
    res = 0.;
  }

  template <typename SCAL>
  static void ApplyK(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, Mat<COMP, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res) {
    res = 0.;
  }

  template <typename SCAL>  
  static void EvalSource(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {
    res = 0.;
  }  
  
  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate) {
    bcstate = 0.;   
  }


  template <typename SCAL>
  static void EvalBdryGradient(int bcnr, Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & bcgrad) {
    bcgrad = grad;
  }  

  template <typename SCAL>
  static void EvalBdryConvFlux(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {
    res = 0.;
  }

  template <typename SCAL>
  static void EvalBdryDiffFlux(int bcnr, Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {
    res = 0.;
  }

  template <typename SCAL>
  static void EvalBdryCoefficients(int bcnr, Vec<COMP, SCAL> & fcn, Vec<COMP, SCAL> & fvn, SpatialParams<D> & sparam, Vec<NumBdryCoeffs, SCAL> & res) {
    res = 0.;
  }
  
  static void EvalBdryFluxWeight(int bcnr, SpatialParams<D> & sparam, Mat<NumBdryFluxWeight, COMP> & res) {
    res = 0.;
  }

  template <typename SCAL>
  static void EvalVolFunctionals(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<NumVolFunctionals, SCAL> & res) {
    res = 0.;
  }
  /// Evaluate the solution \f$u\f$ and the gradient \f$\nabla u\f$ at each point
  template <typename SCAL>
  static void EvalAnalyticSol(Vec<COMP> & w, Mat<COMP, D, SCAL> & q, SpatialParams<D> & sparam) {
    w = 0.;
    q = 0.;
  }

  template <typename SCAL>
  static void DerConvFlux(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<D> & res) {
    res = 0.;
  }
  
  template <typename SCAL>
  static void DerDiffFlux(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<D> & res) {
    res = 0.0;
  }
  
  template <typename SCAL>
  static void DerDiffFluxCoeff(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<D> & res) {
    res = 0.0;
  }

  static void LoadParameters(shared_ptr<PDE> apde) {
  }

  static void GetBCName(int bcnr, stringstream & oss) {
    oss << "Interior";
  }
};

#endif
