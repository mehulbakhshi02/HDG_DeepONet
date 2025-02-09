template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalInflow(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & res) {

  //  EvalFarfield(state, sparam, res);
  //  return;

  
  SCAL rho       = state(0);
  SCAL rhoinv    = 1. / rho;
  Vec<D, SCAL> U = rhoinv * state.Rows(1, D+1);
  SCAL rhoE      = state(D+1);
  SCAL rhok      = state(D+2);

  double gm1     = gamma - 1.;
  double gm1inv  = 1. / gm1;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);
  SCAL Un = 0.;
  for (int dd = 0; dd < D; ++dd)
    Un += U(dd) * sparam.normal(dd);

  SCAL p  = gm1 * (rhoE - 0.5 * rho * U2 - rhok); // + 2. / 3. * rhok;
  SCAL c2 = gamma * p * rhoinv;
  SCAL c  = sqrt(c2);

  // Riemann invariant inflow BC
  SCAL rplus  = Un + 2. * c * gm1inv;
  SCAL rplus2 = rplus * rplus;

  Vec<D> dir;
  if (D == 2) {
    dir(0) = cos(angle(0));
    dir(1) = sin(angle(0));
  } else if (D == 3) {
    dir(0) = cos(angle(0)) * cos(angle(1));
    dir(1) = cos(angle(0)) * sin(angle(1));
    dir(2) = sin(angle(0));
  }
    
  double beta  = InnerProduct(dir, sparam.normal);
  double beta2 = beta * beta;
  SCAL aa = gm1 * (H_infty * beta2 - 0.5 * rplus2);
  SCAL bb = 4. * H_infty * beta;
  SCAL cc = 4. * H_infty * gm1inv - rplus2;

  SCAL disc = bb * bb - 4. * aa * cc;

  SCAL ma1 = 0.5 * (-bb - sqrt(disc)) / aa;
  SCAL ma2 = 0.5 * (-bb + sqrt(disc)) / aa;

  // Determine the smallest, positive solution
  SCAL mab;
  if ((ma1 < 0.) && (ma2 < 0.))
    exit(0);
  else if (ma1 < 0.)
    mab = ma2;
  else if (ma2 < 0.)
    mab = ma1;
  else {
    if (ma1 - ma2 < 0.)
      mab = ma1;
    else
      mab = ma2;
  }

  SCAL trat = 1. + 0.5 * mab * mab * gm1;
  SCAL cvt  = cvTtot / trat;
  SCAL pb   = ptot / pow(trat, gamma*gm1inv);
  SCAL rhob = gm1inv * pb / cvt;
  SCAL cb2  = gamma * pb / rhob;
  SCAL cb   = sqrt(cb2);
    
  res = 0.;
  res(0) = rhob;
  res.Rows(1, D+1) = rhob * mab * cb * dir;
  res(D+1) = pb * gm1inv + 0.5 * rhob * mab * mab * cb2 + rhob * k_infty;
  res(D+2) = rhob * k_infty;
  res(D+3) = rhob * w_infty;

}
