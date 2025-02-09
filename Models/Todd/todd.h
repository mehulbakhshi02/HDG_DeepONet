#ifndef TODD_H
#define TODD_H

template <int D>
class Todd : virtual public DummyModel<D, 1> {
 public:

  // Charaterization of the model
  static const int  Components = 1;
  static const bool Convection = false;
  static const bool Diffusion  = true;
  static const bool Source     = true;

  static const InitType inittype = IT_Function;

  static const int NumBdryFluxWeight = 0;
  static const int NumBdryCoeffs     = 0;
  static const int NumVolFunctionals = 2;
  
  // User-defined parameters
  static double epsilon;
  static double c;

  template <typename SCAL>
  static void EvalInitial(SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
    res = 1.;
    for (int dd = 0; dd < D; ++dd)
      res *= sin(M_PI * sparam.pos(dd));
  }
  
  template <typename SCAL>  
  static void EvalDiffFlux(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    res = (epsilon + state(0)) * grad;
  }

  template <typename SCAL>
  static void ApplyK(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, Mat<1, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<1, D, SCAL> & res) {
    res = (epsilon + state(0)) * rhs;
  }  

  template <typename SCAL>
  static void EvalSource(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {

    double u = 1.;
    for (int dd = 0; dd < D; ++dd)
      u *= sin(M_PI * sparam.pos(dd));

    Vec<D> du;
    if (D == 2) {
      du(0) = M_PI * cos(M_PI * sparam.pos(0)) * sin(M_PI * sparam.pos(1));
      du(1) = M_PI * sin(M_PI * sparam.pos(0)) * cos(M_PI * sparam.pos(1));    
    }

    Vec<D> ddu = -M_PI * M_PI * u;

    double du2 = 0.;
    for (int dd = 0; dd < D; ++dd)
      du2 += du(dd) * du(dd);

    double ddu_sum = 0.;
    for (int dd = 0; dd < D; ++dd)
      ddu_sum += ddu(dd);

    double g = -(1. + c) * du2 - (epsilon + u) * ddu_sum;
    
    SCAL gradu2 = 0.;
    for (int dd = 0; dd < D; ++dd)
      gradu2 += grad(0, dd) * grad(0, dd);

    res = c * gradu2 + g;
  }

  template <typename SCAL>
  static void EvalBdryState(int bcnr, Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<1, SCAL> & bcstate) {
    bcstate = 0.;
  }

  template <typename SCAL>
  static void EvalBdryDiffFlux(int bcnr, Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {

    Vec<1, SCAL> bcstate;
    EvalBdryState(bcnr, state, sparam, bcstate);
        
    Mat<1, D, SCAL> flux;
    EvalDiffFlux(state, grad, sparam, flux);
    res = flux * sparam.normal;
  }

  template <typename SCAL>
  static void EvalVolFunctionals(Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<NumVolFunctionals, SCAL> & res) {

    double u = 1.;
    for (int dd = 0; dd < D; ++dd)
      u *= sin(M_PI * sparam.pos(dd));
    
    res(0) = pow(2. * u - state(0), 2.);
    res(1) = pow(u - state(0), 2.);
  }
  

  static void LoadParameters(PDE & apde) {

    //    LoadConstant(apde, "epsilon", epsilon);
    //    LoadConstant(apde, "c", c);
    
  }
  
};

template<int D> double Todd<D>::epsilon = 1.;
template<int D> double Todd<D>::c       = .5;


#endif
