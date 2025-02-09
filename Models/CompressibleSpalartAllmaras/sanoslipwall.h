template <int D, int COMP>
template <typename SCAL>
void CompressibleSpalartAllmaras<D, COMP>
::EvalNoslipwallAdiabatic(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

  CompressibleNavierStokes<D, COMP>::EvalNoslipwallAdiabatic(state, sparam, res);
  res(D+2)         = 0.;
}

template <int D, int COMP>
template <typename SCAL>
void CompressibleSpalartAllmaras<D, COMP>
::EvalNoslipwallIsothermal(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

  CompressibleNavierStokes<D, COMP>::EvalNoslipwallIsothermal(state, sparam, res);
  res(D+2)         = 0.;
}
