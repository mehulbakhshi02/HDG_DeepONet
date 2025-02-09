template <int D, int COMP>
template <typename SCAL>
void CompressibleSpalartAllmaras<D, COMP>
::EvalDiffFlux(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res) {

  SCAL rho(state(0));
  SCAL rhoinv(1. / rho);
  SCAL mu_t(state(D+2));
  
  // Primitive variables and their gradients
  SCAL p(0.), T(0.);
  Vec<D, SCAL> U(0.), gradp(0.), gradT(0.);
  Mat<D, D, SCAL> gradU(0.);
  CompressibleNavierStokes<D, COMP>::EvalPrim(state, grad, U, p, T, gradU, gradp, gradT);
  Vec<D, SCAL> gradnu_t(rhoinv * (grad.Row(D+2) - mu_t * rhoinv * grad.Row(0)));
  
  SCAL visc(CompressibleNavierStokes<D, COMP>::EvalPhysVisc(rho, p));
  SCAL chi(amu * mu_t / visc);
  SCAL visc_t(EvalEddyVisc(mu_t, chi));
  SCAL visc_sa(1. / sigma * (visc + EvalFn(chi) * mu_t));  

  Mat<D, D, SCAL> S(CompressibleNavierStokes<D, COMP>::EvalStrainRate(gradU));
  Mat<D, D, SCAL> tau((visc+visc_t) * S);

  res = 0.;
  res.Rows(1, D+1) = tau;
  res.Row(D+1)     = tau * U + (visc / prandtl + visc_t / prandtl_t) * gradT;
  res.Row(D+2)     = visc_sa * gradnu_t;
}


template <int D, int COMP>
template <typename SCAL>
void CompressibleSpalartAllmaras<D, COMP>
::ApplyK(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & Grad, Mat<COMP, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res) {

  SCAL rho(state(0));
  SCAL rhoinv(1. / rho);
  SCAL E(state(D+1));
  SCAL mu_t(state(D+2));

  // Primitive variables
  SCAL nu_t(mu_t * rhoinv);

  Vec<D, SCAL> U(rhoinv * state.Rows(1, D+1));

  SCAL U2(0.);
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);

  SCAL p((gamma - 1.) * (E - 0.5 * rho * U2));

  // Compute viscosities
  SCAL visc(CompressibleNavierStokes<D, COMP>::EvalPhysVisc(rho, p));
  SCAL chi(amu * mu_t / visc);
  SCAL visc_t(EvalEddyVisc(mu_t, chi));
  SCAL rgm1(gamma * (visc / prandtl + visc_t / prandtl_t));
  SCAL visc_sa(1. / sigma * (visc + EvalFn(chi) * mu_t));

  visc += visc_t;

  SCAL viscinv(1. / visc);
  rgm1    *= viscinv;
  visc_sa *= viscinv;

  SCAL temp(rgm1 * (E / rho - U2));

  const double c1 = 4./3., c2 = 2./3., c3 = 1 - c2;

  SCAL u(U(0)), v(U(1));
    
  if (D == 2) {

    Mat<COMP, COMP, SCAL> B11(0.), B12(0.), B21(0.), B22(0.);

    B11(1, 0) = -c1 * u;
    B11(1, 1) = c1;
    B11(2, 0) = -v;
    B11(2, 2) = 1.;
    B11(3, 0) = -(c1 * u * u + v * v + temp);
    B11(3, 1) = (c1 - rgm1) * u;
    B11(3, 2) = (1. - rgm1) * v;
    B11(3, 3) = rgm1;
    B11(4, 0) = -visc_sa * nu_t;
    B11(4, 4) = visc_sa;

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
    B22(3, 0) = -(u * u + c1 * v * v + temp);
    B22(3, 1) = (1. - rgm1) * u;
    B22(3, 2) = (c1 - rgm1) * v;
    B22(3, 3) = rgm1;
    B22(4, 0) = -visc_sa * nu_t;
    B22(4, 4) = visc_sa;

    res.Col(0) = B11 * rhs.Col(0) + B12 * rhs.Col(1);
    res.Col(1) = B21 * rhs.Col(0) + B22 * rhs.Col(1);

  } else if (D == 3) {
    cout << "CompressibleSpalartAllmaras<3>::ApplyK() not implemented yet" << endl;
    exit(0);
  }

  res *= visc * rhoinv;    
}
