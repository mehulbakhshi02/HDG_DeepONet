template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalOutflow(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & res) {


  //  EvalFarfield(state, sparam, res);
  //  return;
  
  SCAL rho       = state(0);
  SCAL rhoinv    = 1. / rho;
  Vec<D, SCAL> U = rhoinv * state.Rows(1, D+1);
  SCAL rhoE      = state(D+1);
  SCAL k         = state(D+2) * rhoinv;
  SCAL what      = state(D+3) * rhoinv;

  double gm1     = gamma - 1.;
  double gm1inv  = 1. / gm1;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);

  // res = state;
  // res(D+1) = gm1inv * p_infty + 0.5 * rho * U2 + state(D+2);
  
  SCAL Un = 0.;
  for (int dd = 0; dd < D; ++dd)
    Un += U(dd) * sparam.normal(dd);

  SCAL p  = gm1 * (rhoE - 0.5 * rho * U2 - rho * k); // + 2. / 3. * rho * k;
  SCAL c2 = gamma * p * rhoinv;
  SCAL c  = sqrt(c2);

  SCAL rhob = rho * pow(p_infty/p, 1./gamma);

  // Do some iterations to find density  
  // SCAL f(0.), df(0.);
  // do {
  //   SCAL p_infty_eff = p_infty + 2. / 3. * rhob * k;
  //   f  = rho * pow(p_infty_eff / p, 1./gamma) - rhob;
  //   df = 2. / 3. * rho * k / (gamma * p) * pow(p_infty_eff / p, -gm1/gamma) - 1.;

  //   rhob -= f / df;
    
  // } while (f > 1.e-12);
  
  SCAL cb2  = gamma * p_infty / rhob;
  SCAL dUnb = 2. * gm1inv * (c - sqrt(cb2));
    
  Vec<D, SCAL> Ub = U + dUnb * sparam.normal;
  SCAL U2b = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2b += Ub(dd) * Ub(dd);
  
  res(0)           = rhob;
  res.Rows(1, D+1) = rhob * Ub;
  res(D+1)         = gm1inv * p_infty + 0.5 * rhob * U2b + rhob * k;
  res(D+2)         = rhob * k;
  res(D+3)         = rhob * what;
}
