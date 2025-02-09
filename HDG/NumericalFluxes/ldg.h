#ifndef HDG_LDG_H
#define HDG_LDG_H

template <int D, int COMP, class Model>
template <typename SCAL>
void UnifyingFramework<D, COMP, Model>
  ::NumViscFlux(Vec<COMP, SCAL> & state1, Mat<COMP, D, SCAL> & grad1, Vec<COMP, SCAL> & state2, Mat<COMP, D, SCAL> & grad2, SpatialParams<D> & sparam, Vec<COMP, SCAL> & vflux) const {
    
  Mat<COMP, D, SCAL> fvl(0.);
  Model::EvalDiffFlux(state1, grad2, sparam, fvl);
  vflux = fvl * sparam.normal + stab_visc * (state1 - state2);
}

#endif
