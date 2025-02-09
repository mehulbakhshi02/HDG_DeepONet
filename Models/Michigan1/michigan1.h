#ifndef MICHIGAN1_H
#define MICHIGAN1_H

#include "../Advection/advection.h"
#include "../Poisson/poisson.h"

template <int D>
class Michigan1 : public Advection<D>, public Poisson<D> {
 public:

  // Charaterization of the model
  static const int  Components = 1;
  static const bool Convection = true;
  static const bool Diffusion  = true;
  static const bool Source     = true;

  static const InitType inittype = IT_Zero; //Function;

  static const int NumBdryFluxWeight = 0;
  static const int NumBdryCoeffs     = 0;

  // Both Advection and Poisson provide these functions
  // Therefore, we have to make a decision
  using Advection<D>::EvalInitial;
  using Poisson<D>::EvalBdryState;
  using Poisson<D>::ApplyK;
  using Poisson<D>::EvalBdryDiffFlux;
  
  template <typename SCAL>
  static void EvalSource(Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {


    double x = sparam.pos(0);
    double y = sparam.pos(1);

    double eps = Poisson<D>::epsilon;

    double fx = tanh((1. - x) / eps);
    double fy = tanh((1. - y) / eps);

    double fxp = 1. / eps * (fx * fx - 1.);
    double fyp = 1. / eps * (fy * fy - 1.);

    double fxpp = -2. / (eps * eps) * (1. - fx * fx) * fx;
    double fypp = -2. / (eps * eps) * (1. - fy * fy) * fy;

    double ux = y * fx * fy + x * y * fxp * fy;
    double uy = x * fx * fy + x * y * fx * fyp;

    double uxx = 2. * y * fxp * fy + x * y * fxpp * fy;
    double uyy = 2. * x * fx * fyp + x * y * fx * fypp;    

    res = ux + uy - eps * (uxx + uyy);
  }

  template <typename SCAL>
  static void EvalBdryConvFlux(int bcnr, Vec<1, SCAL> & state, SpatialParams<D> & sparam, Vec<1, SCAL> & res) {
    Vec<1, SCAL> bcstate;
    Mat<1, D, SCAL> flux;
    EvalBdryState(bcnr, state, sparam, bcstate);
    Advection<D>::EvalConvFlux(bcstate, sparam, flux);
    res = flux * sparam.normal;
  }


  // template <typename SCAL>
  // static void EvalBdryDiffFlux(int bcnr, Vec<1, SCAL> & state, Mat<1, D, SCAL> & grad, Vec<D> & pos, Vec<D> & normal, Vec<1, SCAL> & res) {

  //   // Vec<D> dir;
  //   // Advection<D>::GetDirection(pos, dir);
  //   // double dirn = 0.;
  //   // for (int dd = 0; dd < D; ++dd)
  //   //   dirn += dir(dd) * normal(dd);

  //   //    if (bcnr == 0 || bcnr == 1) { // Inflow
  //     Vec<1, SCAL> bcstate;
  //     EvalBdryState(bcnr, state, pos, normal, bcstate);
        
  //     Mat<1, D, SCAL> flux;
  //     Poisson<D>::EvalDiffFlux(state, grad, pos, flux);
  //     res = flux * normal;
  //     //    } else {
  //     //      res = 0.;
  //     //    }
  // }  

  static void EvalBdryFluxWeight(int bcnr, SpatialParams<D> & sparam, Mat<NumBdryFluxWeight, 1> & res) {

    res = cos(2. * M_PI * sparam.pos(0)) * cos(2. * M_PI * sparam.pos(1));

    // if (bcnr == 2 || bcnr == 3)
    //   res = 1.;
    // else
    //   res = 0.;
  }

  static void LoadParameters(PDE & apde) {
    Advection<D>::LoadParameters(apde);
    Poisson<D>::LoadParameters(apde);
  }
  
};


#endif
