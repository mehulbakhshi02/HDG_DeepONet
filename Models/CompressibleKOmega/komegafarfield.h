
template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalFarfield(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & res) {

  // Evaluate characteristic farfield values for density, momentum, and energy using Euler BC
  Vec<D+2, SCAL> newstate = state.Rows(0, D+2);
  Vec<D+2, SCAL> newres;
  CompressibleNavierStokes<D>::EvalFarfield(newstate, sparam, newres);
  res.Rows(0, D+2) = newres;
  res(D+2) = rho_infty * k_infty;
  res(D+3) = rho_infty * w_infty;

  // Check normal velocity; if outgoing, take interior states for k and w
  Vec<D, SCAL> U = (1. / state(0)) * state.Rows(1, D+1);
  SCAL Un = 0.;
  for (int dd = 0; dd < D; ++dd)
    Un += U(dd) * sparam.normal(dd);

  if (Un > 0.) {
    res(D+2) = state(D+2);
    res(D+3) = state(D+3);
  }
}


