template <int D>
template <typename SCAL>
SCAL CompressibleKOmega<D>
::EvalWallOmega(SCAL rho, SCAL p) {

  SCAL visc = CompressibleNavierStokes<D>::EvalPhysVisc(rho, p);
  return log(60. * visc / (rho * beta_w * min_y * min_y));
  
}

template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalNoslipwallAdiabatic(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate) {

  SCAL rho  = state(0);
  SCAL rhoE = state(D+1);
  SCAL p    = (gamma - 1.) * rhoE;
    
  bcstate      = 0.;
  bcstate(0)   = rho;
  bcstate(D+1) = rhoE;
  bcstate(D+3) = rho * EvalWallOmega(rho, p);
}

template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalNoslipwallIsothermal(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate) {

  double T_infty = gamma / (gamma - 1.) * p_infty / rho_infty;
  SCAL rhoE = state(D+1);
  SCAL p    = (gamma - 1.) * rhoE;
  
  bcstate      = 0.;
  bcstate(0)   = gamma / T_infty * rhoE;
  bcstate(D+1) = rhoE;
  bcstate(D+3) = bcstate(0) * EvalWallOmega(bcstate(0), p);  
}
