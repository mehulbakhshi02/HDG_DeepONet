template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalConvFlux(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Mat<D+4, D, SCAL> & res) {

  SCAL rho       = state(0);
  Vec<D, SCAL> m = state.Rows(1, D+1);
  SCAL rhoE      = state(D+1);
  SCAL rhok      = state(D+2);

  Vec<D, SCAL> U = 1. / rho * m;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);

  SCAL p_eff = (gamma - 1.) * (rhoE - 0.5 * rho * U2 - rhok) + 2. / 3. * rhok;

  // Convective part
  res = state * Trans(U);

  // Pressure part
  res.Rows(1, D+1) += p_eff * Id<D>();
  res.Row(D+1)     += p_eff * U;
}

template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalMaxEigenvalue(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, SCAL & res) {

  SCAL rho       = state(0);
  Vec<D, SCAL> m = state.Rows(1, D+1);
  SCAL rhoE      = state(D+1);
  SCAL rhok      = state(D+2);

  Vec<D, SCAL> U = 1. / rho * m;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);

  SCAL p_eff = (gamma - 1.) * (rhoE - 0.5 * rho * U2 - rhok) + 2. / 3. * rhok;
  SCAL c2 = gamma * p_eff / rho;

  res = sqrt(U2) + sqrt(c2);
}

template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalConvEigenSystem(Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, double fix, Vec<D+4, SCAL> & eig, Mat<D+4, D+4, SCAL> & R, Mat<D+4, D+4, SCAL> & L) {

  SCAL rho       = state(0);
  Vec<D, SCAL> m = state.Rows(1, D+1);
  SCAL rhoE      = state(D+1);
  SCAL rhok      = state(D+2);
  SCAL rhoomhat  = state(D+3);

  Vec<D, SCAL> U = 1. / rho * m;

  SCAL k     = rhok / rho;
  SCAL omhat = rhoomhat / rho;

  double gm1    = gamma - 1.;
  double gm1inv = 1. / gm1;
  double gm53   = gm1 - 2. / 3.;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);
  SCAL Un = 0.;
  for (int dd = 0; dd < D; ++dd)
    Un += U(dd) * sparam.normal(dd);

  SCAL p_eff = gm1 * (rhoE - 0.5 * rho * U2 - rhok) + 2. / 3. * rhok;
  SCAL c2 = gamma * p_eff / rho;
  SCAL c  = sqrt(c2);
  SCAL c2inv = 1. / c2;
  SCAL cinv  = 1. / c;
  SCAL Ma2   = U2 * c2inv;

  eig = Un;
  eig(D+2) += c;
  eig(D+3) -= c;

  if (fix > 0.) {
    SCAL eps = fix * c;
    SCAL epsinv = 1. / eps;
    for (int ll = 0; ll < D+2; ++ll)
      if ((eps - eig(ll) > 0.) && (eig(ll) + eps > 0.))
        eig(ll) = 0.5 * (eps + eig(ll) * eig(ll) * epsinv);
  }

  SCAL H_eff = (rhoE + p_eff) / rho;

  double nx, ny, nz;
  nx = sparam.normal(0);
  ny = sparam.normal(1);
  if (D == 3)  nz = sparam.normal(2);

  R = L = 0.;
  
  // compute right eigenvectors
  Vec<D+4, SCAL> v0(0.), v1(0.), v2(0.), v3(0.);

  v0(0)           = 1.;
  v0.Rows(1, D+1) = U;
  v0(D+1)         = 0.5 * U2;
  
  if (D == 2) {
    v3(1) = ny;
    v3(2) = -nx;
    v3(3) = U(0) * ny - U(1) * nx;
  } else if (D == 3) {
    // To be done
  }

  if (D == 2) {
    R.Col(0) = v0;
    R.Col(1) = v3;
  } else if (D == 3) {
    // To be done
  }

  R(D+1, D)   = gm53;
  R(D+2, D)   = gm1;
  R(D+3, D+1) = 1.;

  Vec<D+4, SCAL> a1(0.), a2(0.);
  
  a1(0)           = 1.;
  a1.Rows(1, D+1) = U;
  a1(D+1)         = H_eff;
  a1(D+2)         = k;
  a1(D+3)         = omhat;

  a2.Rows(1, D+1) = sparam.normal;
  a2(D+1)         = Un;  

  R.Col(D+2) = a1 + c * a2;
  R.Col(D+3) = a1 - c * a2;

  // compute left eigenvectors
  SCAL u, v, w;
  u = U(0);
  v = U(1);
  if (D == 3) w = U(2);
  
  if (D == 2) {
    // The following can be simplified (just like the right eigenvectors)
    L(0, 0) = 1. - .5 * gm1 * Ma2;
    L(1, 0) = nx * v - ny * u;
    L(2, 0) = -.5 * Ma2 * k;
    L(3, 0) = -.5 * omhat * gm1 * Ma2;    
    L(4, 0) = -.5 * (Un * cinv - .5 * gm1 * Ma2);
    L(5, 0) =  .5 * (Un * cinv + .5 * gm1 * Ma2);

    L(0, 1) = gm1 * u * c2inv;
    L(1, 1) = ny;
    L(2, 1) = k * u * c2inv;
    L(3, 1) = omhat * gm1 * u * c2inv;
    L(4, 1) =  .5 * (nx * c - u * gm1) * c2inv;
    L(5, 1) = -.5 * (nx * c + u * gm1) * c2inv;

    L(0, 2) = gm1 * v * c2inv;
    L(1, 2) = -nx;
    L(2, 2) = k * v * c2inv;    
    L(3, 2) = omhat * gm1 * v * c2inv;
    L(4, 2) =  .5 * (ny * c - v * gm1) * c2inv;
    L(5, 2) = -.5 * (ny * c + v * gm1) * c2inv;

    L(0, 3) = -gm1 * c2inv;
    L(2, 3) = -k * c2inv;    
    L(3, 3) = -omhat * gm1 * c2inv;
    L(4, 3) =  .5 * gm1 * c2inv;
    L(5, 3) =  .5 * gm1 * c2inv;    

    L(0, 4) = gm53 * c2inv;
    L(2, 4) = (1. + gm53 * k * c2inv) * gm1inv;
    L(3, 4) = omhat * gm53 * c2inv;
    L(4, 4) = -.5 * gm53 * c2inv;
    L(5, 4) = -.5 * gm53 * c2inv;
  } else if (D == 3) {
    // To be done
  }

  L(D+1, D+3) = 1.;
}

template <int D>
void CompressibleKOmega<D>
::LoadParameters(PDE & apde) {

  CompressibleNavierStokes<D>::LoadParameters(apde);

  if (apde.ConstantUsed("Tu"))
    Tu  = apde.GetConstant("Tu");
  if (apde.ConstantUsed("eddy_visc_r"))
    Evr = apde.GetConstant("eddy_visc_r");

  double visc = CompressibleNavierStokes<D>::EvalPhysVisc(rho_infty, p_infty);
    
  k_infty = 1.5 * pow(mach * c_infty * Tu, 2.);
  w_infty = log(rho_infty * k_infty / (visc * Evr));  
}
