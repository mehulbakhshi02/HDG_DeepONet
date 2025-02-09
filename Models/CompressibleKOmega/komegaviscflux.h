template <int D>
template <typename SCAL>
SCAL CompressibleKOmega<D>
::EvalEddyVisc(SCAL & rho, SCAL & rhok, SCAL & omega_r, SCAL & visc, bool includek) {

  if (rhok < 0.) rhok = 0.;    // k should be non-negative

  SCAL visc_t;
  if (includek) { // for the viscous flux
    visc_t = rhok / omega_r;
    //    if (visc_t - 1.e6 * visc > 0.)
    //      visc_t = 1.e6 * visc;
  } else { // for the source term
    visc_t = 1. / omega_r;
    //    if (rhok * visc_t - 1.e6 * visc > 0. && rhok > 0.)
    //      visc_t = 1.e6 * visc / rhok;
  }

  return visc_t;
}

template <int D>
template <typename SCAL>
SCAL CompressibleKOmega<D>
::EvalLimitedOmega(SCAL & rho, SCAL & rhok, SCAL & rhowhat, SCAL & visc, Mat<D, D, SCAL> & S) {

  SCAL omega_r = exp(rhowhat / rho);

  SCAL amax = 0.;
  // Compute realizable omega
  for (int i = 0; i < D; ++i) {
    for (int j = i+1; j < D; ++j) {
      SCAL q1 = 0.5 * (S(i, i) + S(j, j));
      SCAL q2 = S(i, i) * S(j, j) - S(i, j) * S(i, j);

      // Solve quadratic equation for exp(omhat)/alpha*
      SCAL temp = pow(q1, 2.) - q2;
      SCAL a = 0.;
      if (temp > 0.)
        a = 1.5 * (q1 + sqrt(temp));
      else if (temp < 0.)
        cout << "OHOH1" << endl;
      else
        a = 1.5 * q1;
        
      if (a - amax > 0.) amax = a;
    }
  }

  // High-Reynolds;
  if (amax - omega_r > 0.) omega_r = amax;

  return omega_r;
}

template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalDiffFlux(Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<D+4, D, SCAL> & res) {

  SCAL rho       = state(0);
  Vec<D, SCAL> m = state.Rows(1, D+1);
  SCAL rhoE      = state(D+1);
  SCAL rhok      = state(D+2);
  SCAL rhowhat   = state(D+3);

  SCAL rhoinv  = 1. / rho;
  SCAL rhoinv2 = rhoinv * rhoinv;

  // Compute primitive variables and their gradients
  Vec<D, SCAL> U = 1. / rho * m;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);
  
  SCAL p = (gamma - 1.) * (rhoE - 0.5 * rho * U2 - rhok);

  Mat<D, D, SCAL> gradU = rhoinv * (grad.Rows(1, D+1) - U * Trans(grad.Row(0)));
  Vec<D, SCAL>    gradp = (gamma - 1.) * (grad.Row(D+1) - rho * Trans(gradU) * U - 0.5 * U2 * grad.Row(0) - grad.Row(D+2));
  Vec<D, SCAL>    gradT = rhoinv2 * gamma / (gamma - 1.) * (rho * gradp - p * grad.Row(0));
    
  Vec<D, SCAL> gradk    = rhoinv2 * (rho * grad.Row(D+2) - rhok * grad.Row(0));
  Vec<D, SCAL> gradwhat = rhoinv2 * (rho * grad.Row(D+3) - rhowhat * grad.Row(0));

  Mat<D, D, SCAL> S = CompressibleNavierStokes<D>::EvalStrainRate(gradU);

  SCAL visc    = CompressibleNavierStokes<D>::EvalPhysVisc(rho, p);
  SCAL omega_r = EvalLimitedOmega(rho, rhok, rhowhat, visc, S);
  SCAL visc_t  = EvalEddyVisc(rho, rhok, omega_r, visc, true);
    
  Mat<D, D, SCAL> tau = (visc + visc_t) * S;

  res = 0.;
  res.Rows(1, D+1) = tau;
  res.Row(D+1)     = tau * U + (visc / prandtl + visc_t / prandtl_t) * gradT + (visc + sigma_k * visc_t) * gradk;
  res.Row(D+2)     = (visc + sigma_k * visc_t) * gradk;
  res.Row(D+3)     = (visc + sigma_w * visc_t) * gradwhat;
}

template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::ApplyK(Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, Mat<D+4, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<D+4, D, SCAL> & res) {

  SCAL rho     = state(0);
  SCAL rhoE    = state(D+1);
  SCAL rhok    = state(D+2);
  SCAL rhowhat = state(D+3);

  SCAL rhoinv  = 1. / rho;

  SCAL k     = rhoinv * rhok;
  SCAL omhat = rhoinv * rhowhat;
    
  // Compute primitive variables and their gradients
  Vec<D, SCAL> U = 1. / rho * state.Rows(1, D+1);

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);
  
  SCAL p = (gamma - 1.) * (rhoE - 0.5 * rho * U2 - rhok);

  Mat<D, D, SCAL> gradU = rhoinv * (grad.Rows(1, D+1) - U * Trans(grad.Row(0)));
  Mat<D, D, SCAL> S     = CompressibleNavierStokes<D>::EvalStrainRate(gradU);

  SCAL visc    = CompressibleNavierStokes<D>::EvalPhysVisc(rho, p);
  SCAL omega_r = EvalLimitedOmega(rho, rhok, rhowhat, visc, S);
  SCAL visc_t  = EvalEddyVisc(rho, rhok, omega_r, visc, true);

  SCAL rgm1 = gamma * (visc / prandtl + visc_t / prandtl_t);
  SCAL visc_k = visc + sigma_k * visc_t;
  SCAL visc_w = visc + sigma_w * visc_t;
  visc = visc + visc_t;
  SCAL visc1 = 1. / visc;
  rgm1   *= visc1;
  visc_k *= visc1;
  visc_w *= visc1;
    
  if (D == 2) {
    double c1 = 4./3.,     c2 = 2./3.,     c3 = 1./3.;
    SCAL   u  = U(0),      v  = U(1);

    Mat<D+4, D+4, SCAL> B11(0.), B12(0.), B21(0.), B22(0.);

    B11(1, 0) = -c1 * u;
    B11(1, 1) = c1;
    B11(2, 0) = -v;
    B11(2, 2) = 1.;
    B11(3, 0) = -(c1 * u * u + v * v + rgm1 * (rhoE / rho - U2 - k) + visc_k * k);
    B11(3, 1) = (c1 - rgm1) * u;
    B11(3, 2) = (1. - rgm1) * v;
    B11(3, 3) = rgm1;
    B11(3, 4) = visc_k - rgm1;
    B11(4, 0) = -visc_k * k;     B11(4, 4) = visc_k;
    B11(5, 0) = -visc_w * omhat; B11(5, 5) = visc_w;

    B12(1, 0) = c2 * v;
    B12(1, 2) = -c2;
    B12(2, 0) = -u;
    B12(2, 1) = 1.;
    B12(3, 0) = -c3 * u * v;
    B12(3, 1) = v;
    B12(3, 2) = -c2 * u;

    B21(1, 0) = -v;
    B21(1, 2) = 1.;
    B21(2, 0) = c2 * u;
    B21(2, 1) = -c2;
    B21(3, 0) = -c3 * u * v;
    B21(3, 1) = -c2 * v;
    B21(3, 2) = u;

    B22(1, 0) = -u;
    B22(1, 1) = 1.;
    B22(2, 0) = -c1 * v;
    B22(2, 2) = c1;
    B22(3, 0) = -(u * u + c1 * v * v + rgm1 * (rhoE / rho - U2 - k) + visc_k * k);
    B22(3, 1) = (1. - rgm1) * u;
    B22(3, 2) = (c1 - rgm1) * v;
    B22(3, 3) = rgm1;
    B22(3, 4) = visc_k - rgm1;
    B22(4, 0) = -visc_k * k;     B22(4, 4) = visc_k;
    B22(5, 0) = -visc_w * omhat; B22(5, 5) = visc_w;

    res.Col(0) = B11 * rhs.Col(0) + B12 * rhs.Col(1);
    res.Col(1) = B21 * rhs.Col(0) + B22 * rhs.Col(1);

  } else if (D == 3) {
    cout << "Not implemented yet" << endl;
    exit(0);
  }

  res *= visc * rhoinv;  
}
