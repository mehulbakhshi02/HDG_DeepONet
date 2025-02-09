template <int D, int COMP>
template <typename SCAL>
void CompressibleSpalartAllmaras<D, COMP>
::EvalSource(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Vec<COMP, SCAL> & res) {

  SCAL rho(state(0));
  SCAL rhoinv(1. / rho);
  SCAL mu_t(state(D+2));

  // Primitive variables and their gradients
  SCAL p(0.), T(0.);
  Vec<D, SCAL> U(0.), gradp(0.), gradT(0.);
  Mat<D, D, SCAL> gradU(0.);
  CompressibleNavierStokes<D, COMP>::EvalPrim(state, grad, U, p, T, gradU, gradp, gradT);
  SCAL nu_t(mu_t * rhoinv);    
  Vec<D, SCAL> gradnu_t(rhoinv * (grad.Row(D+2) - nu_t * grad.Row(0)));

  // Viscosities
  SCAL visc(CompressibleNavierStokes<D, COMP>::EvalPhysVisc(rho, p));
  SCAL chi(amu * mu_t / visc);
  SCAL visc_sa(1. / sigma * (visc + EvalFn(chi) * mu_t));

  SCAL gradnut2(0.);
  for (int dd = 0; dd < D; ++dd)
    gradnut2 += gradnu_t(dd) * gradnu_t(dd);

  SCAL gradrhonut(0.);
  for (int dd = 0; dd < D; ++dd)
    gradrhonut += grad(0, dd) * gradnu_t(dd);

  // Vorticity
  Mat<D, D, SCAL> W(0.5 * (gradU - Trans(gradU)));
  SCAL omega(0.);
  for (int dd = 0; dd < D; ++dd)
    for (int dd2 = 0; dd2 < D; ++dd2)
      omega += pow(W(dd, dd2), 2.);
  if (omega > 0.)
    omega = sqrt(2. * omega);

  // Modified vorticity
  double kwd2 = pow(kappa * sparam.walldistance, 2.);  
  SCAL omegastar(amu * nu_t * EvalFv2(chi) / kwd2);
  SCAL omegatilde(0.);
  if (omegastar + cv2 * omega < 0.)
    omegatilde = omega * (1. + (pow(cv2, 2.) * omega + cv3 * omegastar) / ((cv3 - 2. * cv2) * omega - omegastar));
  else
    omegatilde = omega + omegastar;

  // Production
  SCAL rhoP(cb1 * mu_t);
  if (nu_t < 0.)
    rhoP *= (1. - ct3) * omega;
  else
    rhoP *= omegatilde;

  // Destruction
  SCAL rhoD(amu * cw1 * rho * pow(nu_t / sparam.walldistance, 2.));
  if (nu_t < 0.)
    rhoD *= -1.;
  else {
    SCAL r(amu * nu_t / (omegatilde * kwd2));
    if (r > rlim)
      r = rlim;
    // SCAL cw2lre = cw4 + cw5/(((chi/40.)+1.)*((chi/40.)+1.));
    SCAL g(r + cw2 * (pow(r, 6.) - r));
    rhoD *= EvalFw(g);
  }

  res = 0.;
  res(D+2) = rhoP - rhoD - rhoinv * visc_sa * gradrhonut + amu * rho / sigma * cb2 * gradnut2;  
}
