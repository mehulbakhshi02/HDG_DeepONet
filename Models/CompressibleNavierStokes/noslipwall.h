/*
Adiabatic no-slip wall:
- Density and total energy are taken from the interior
- Momentum is set to zero
 */
template <int D, int COMP>
template <typename SCAL>
void CompressibleNavierStokes<D, COMP>
::EvalNoslipwallAdiabatic(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate) {

  bcstate      = 0.;
  bcstate(0)   = state(0);
  bcstate(D+1) = state(D+1);

}

/*
Isothermal no-slip wall:
- Density is computed from free-stream temperature and interior total energy
- Momentum is set to zero
- Total energy is taken from the interior
 */
template <int D, int COMP>
template <typename SCAL>
void CompressibleNavierStokes<D, COMP>
::EvalNoslipwallIsothermal(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate) {

  double T_infty = gamma / (gamma - 1.) * CompressibleEuler<D>::p_infty / CompressibleEuler<D>::rho_infty;
  SCAL rhoE = state(D+1);

  bcstate      = 0.;
  bcstate(0)   = gamma / T_infty * rhoE;
  bcstate(D+1) = rhoE;

}
