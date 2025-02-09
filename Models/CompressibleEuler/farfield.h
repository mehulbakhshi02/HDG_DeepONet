template <int D, int COMP>
template <typename SCAL>
void CompressibleEuler<D, COMP>
::EvalFarfield(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate) {

  Vec<COMP, SCAL> eig(0.);
  Mat<COMP, COMP, SCAL> eigvr(0.), eigvl(0.);
  EvalConvEigenSystem(state, sparam, 0.01, eig, eigvr, eigvl);

  Vec<COMP, SCAL> wc, wc_infty;
  wc_infty = eigvl * w_infty;
  wc       = eigvl * state;

  for (int ll = 0; ll < COMP; ++ll)
    if (eig(ll) < 0.)
      wc(ll) = wc_infty(ll);

  bcstate = eigvr * wc;
}

