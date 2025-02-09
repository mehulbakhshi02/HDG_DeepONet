template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalSource(Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<D+4, SCAL> & res) {

  SCAL rho     = state(0);
  SCAL rhoE    = state(D+1);
  SCAL rhok    = state(D+2);
  SCAL rhowhat = state(D+3);

  SCAL rhoinv  = 1. / rho;
  SCAL rhoinv2 = rhoinv * rhoinv;

  // Compute primitive variables and their gradients
  Vec<D, SCAL> U = 1. / rho * state.Rows(1, D+1);

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);
  
  SCAL p = (gamma - 1.) * (rhoE - 0.5 * rho * U2 - rhok);

  Mat<D, D, SCAL> gradU = rhoinv * (grad.Rows(1, D+1) - U * Trans(grad.Row(0)));
    
  Vec<D, SCAL> gradk    = rhoinv2 * (rho * grad.Row(D+2) - rhok * grad.Row(0));
  Vec<D, SCAL> gradwhat = rhoinv2 * (rho * grad.Row(D+3) - rhowhat * grad.Row(0));

  Mat<D, D, SCAL> S = CompressibleNavierStokes<D>::EvalStrainRate(gradU);

  SCAL visc    = CompressibleNavierStokes<D>::EvalPhysVisc(rho, p);
  SCAL omega_r = EvalLimitedOmega(rho, rhok, rhowhat, visc, S);
  // Please note that 'rhok' is "post"-multiplied explicitly later
  // (this way it does not occur in the denominator in res(D+3))
  SCAL visc_t  = EvalEddyVisc(rho, rhok, omega_r, visc, false);

  SCAL gradkwhat = 0.;
  for (int dd = 0; dd < D; ++dd)
    gradkwhat += gradk(dd) * gradwhat(dd);

  if (gradkwhat > 0.)
    gradkwhat *= 0.125;
  else
    gradkwhat  = 0.;
  
  SCAL gradwhat2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    gradwhat2 += gradwhat(dd) * gradwhat(dd);

  Mat<D, D, SCAL> tau_R = visc_t * S - (2. / 3.) * Id<D>();

  SCAL tauU = 0.;
  for (int dd = 0; dd < D; ++dd)
    for (int dd2 = 0; dd2 < D; ++dd2)
      tauU += tau_R(dd, dd2) * gradU(dd, dd2);

  SCAL Pk = tauU;
  SCAL Dk = beta_k * omega_r;

  if (Pk - 10. * Dk > 0.)
    Pk = 10. * Dk;

  res = 0.;
  res(D+2) = rhok * (Pk - Dk);
  res(D+3) = alpha_w * rho * tauU - beta_w * rho * omega_r + (visc + sigma_w * rhok * visc_t) * gradwhat2 + rho * gradkwhat / omega_r;
}

