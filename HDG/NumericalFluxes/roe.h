#ifndef HDG_ROE_H
#define HDG_ROE_H

template <int D, int COMP, class Model>
template <typename SCAL>
void UnifyingFramework<D, COMP, Model>
  ::NumConvFlux(Vec<COMP, SCAL> & state1, Vec<COMP, SCAL> & state2, SpatialParams<D> & sparam, Vec<COMP, SCAL> & cflux) const {
    
  Mat<COMP, D, SCAL> fcl(0.);
  Model::EvalConvFlux(state1, sparam, fcl);

  Vec<COMP, SCAL> eig(0.);
  Mat<COMP, COMP, SCAL> eigvr(0.), eigvl(0.);

  Model::EvalConvEigenSystem(state1, sparam, stab_conv, eig, eigvr, eigvl);

  for (int ll = 0; ll < COMP; ++ll)
    if (eig(ll) < 0.)
      eig(ll) *= -1.;

  Vec<COMP, SCAL> stab = eigvl * (state1 - state2);
  for (int ll = 0; ll < COMP; ++ll)
    stab(ll) *= eig(ll);

  cflux = fcl * sparam.normal - eigvr * stab;
}

#endif
