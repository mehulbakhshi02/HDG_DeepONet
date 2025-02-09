#ifndef HDG_LAXFRIEDRICH_H
#define HDG_LAXFRIEDRICH_H

template <int D, int COMP, class Model>
template <typename SCAL>
void UnifyingFramework<D, COMP, Model>
  ::NumConvFlux(Vec<COMP, SCAL> & state1, Vec<COMP, SCAL> & state2, SpatialParams<D> & sparam, Vec<COMP, SCAL> & cflux) const {
    
  Mat<COMP, D, SCAL> fcl;
  Model::EvalConvFlux(state1, sparam, fcl);

  cflux = fcl * sparam.normal - stab_conv *  (state1 - state2);
}

#endif
