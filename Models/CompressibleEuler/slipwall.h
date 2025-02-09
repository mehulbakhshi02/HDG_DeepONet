template <int D, int COMP>
template <typename SCAL>
void CompressibleEuler<D, COMP>
::EvalSlipwall(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate) {

  bcstate = state;
  bcstate.Rows(1, D+1) = (Id<D>() - sparam.normal * Trans(sparam.normal)) * state.Rows(1, D+1);
}
