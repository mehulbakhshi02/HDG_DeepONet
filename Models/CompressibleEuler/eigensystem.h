template <int D, int COMP>
template <typename SCAL>
void CompressibleEuler<D, COMP>
::EvalMaxEigenvalue(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, SCAL & res) {

  SCAL rho       = state(0);
  Vec<D, SCAL> m = state.Rows(1, D+1);
  SCAL E         = state(D+1);

  double gm1 = gamma - 1.;

  Vec<D, SCAL> U = 1. / rho * m;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);

  SCAL p = gm1 * (E - 0.5 * rho * U2);
  SCAL c2 = gamma * p / rho;

  if (sparam.normal_set) {
    SCAL Un = 0.;
    for (int dd = 0; dd < D; ++dd)
      Un += U(dd) * sparam.normal(dd);
    if (Un < 0.)
      Un = -Un;
    res = Un + sqrt(c2);
  } else {
    res = sqrt(U2) + sqrt(c2);
  }
}

template <int D, int COMP>
template <typename SCAL>
void CompressibleEuler<D, COMP>
::EvalConvEigenSystem(Vec<COMP, SCAL> & state, SpatialParams<D> & sparam,
                      double fix,
                      Vec<COMP, SCAL> & eig,
                      Mat<COMP, COMP, SCAL> & eigvr,
                      Mat<COMP, COMP, SCAL> & eigvl) {

  SCAL rho       = state(0);
  Vec<D, SCAL> m = state.Rows(1, D+1);
  SCAL E         = state(D+1);
  
  double gm1 = gamma - 1.;

  Vec<D, SCAL> U = 1. / rho * m;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);
  SCAL Un = 0.;
  for (int dd = 0; dd < D; ++dd)
    Un += U(dd) * sparam.normal(dd);

  SCAL p = gm1 * (E - 0.5 * rho * U2);
  SCAL c2 = gamma * p / rho;
  SCAL c = sqrt(c2);
  SCAL c2inv = 1. / c2;
  SCAL cinv = 1. / c;

  eig = Un;
  eig(COMP-2) += c;
  eig(COMP-1) -= c;
  
  if (fix > 0.) {
    SCAL eps = fix * c;
    SCAL eps1 = 1. / eps;
    for (int dd = 0; dd < COMP; ++dd)
      if ((eps - eig(dd) > 0.) && (eig(dd) + eps > 0.))
        eig(dd) = 0.5 * (eps + eig(dd) * eig(dd) * eps1);
  }
  
  SCAL H = c2 / gm1 + 0.5 * U2;
  double nx = sparam.normal(0), ny = sparam.normal(1);

  // Eigenvectors of the normal flux jacobian
  // See A. Jameson: The Jacobian Matrix for the Conservative Variables (Notes, 2006)

  double nz = 0.0;
  if (D == 3) nz = sparam.normal(2);

  // right eigenvectors
  Vec<D+2, SCAL> a1, a2;
    
  a1(0)           = 1.;
  a1.Rows(1, D+1) = U;
  a1(D+1)         = H;
    
  a2(0)           = 0.;
  a2.Rows(1, D+1) = sparam.normal;
  a2(D+1)         = Un;
    
  Vec<D+2, SCAL> v0, v1, v2, v3;
    
  v0(0)           = 1.;
  v0.Rows(1, D+1) = U;
  v0(D+1)         = 0.5 * U2;

  if (D == 2) {
    v3(0) = 0.; 
    v3(1) = ny; v3(2) = -nx;
    v3(3) = U(0) * ny - U(1) * nx;
  } else if (D == 3) {
    v1(0) = 0.; 
    v1(1) = 0.; v1(2) = nz; v1(3) = -ny;
    v1(4) = U(1) * nz - U(2) * ny;
      
    v2(0) = 0.; 
    v2(1) = -nz; v2(2) = 0.; v2(3) = nx;
    v2(4) = U(2) * nx - U(0) * nz;
      
    v3(0) = 0.; 
    v3(1) = ny; v3(2) = -nx; v3(3) = 0.;
    v3(4) = U(0) * ny - U(1) * nx;
  }

  if (D == 2) {
    eigvr.Col(0) = v0;
    eigvr.Col(1) = v3;
    eigvr.Col(2) = a1 + c * a2;
    eigvr.Col(3) = a1 - c * a2;
  } else if (D == 3) {
    eigvr.Col(0) = nx * v0 + c * v1;
    eigvr.Col(1) = ny * v0 + c * v2;
    eigvr.Col(2) = nz * v0 + c * v3;
    eigvr.Col(3) = a1 + c * a2;
    eigvr.Col(4) = a1 - c * a2;
  }
    
  // left eigenvectors
  Vec<D+2, SCAL> b1, b2;
    
  b1(0)           = -Un;
  b1.Rows(1, D+1) = sparam.normal;
  b1(D+1)         = 0.;
    
  b2(0)           = .5 * gm1 * U2;
  b2.Rows(1, D+1) = -gm1 * U;
  b2(D+1)         = gm1;

  Vec<D+2, SCAL> p0, p1, p2, p3;
    
  SCAL gm1c2inv = gm1 * c2inv;

  p0(0)           = gm1c2inv * (H - U2);
  p0.Rows(1, D+1) = gm1c2inv * U;
  p0(D+1)         = -gm1c2inv;

  if (D == 2) {
    p3(0) = U(0) * ny - U(1) * nx;
    p3(1) = -ny; p3(2) = nx;
    p3(3) = 0.;
  } else if (D == 3) {
    p1(0) = U(1) * nz - U(2) * ny;
    p1(1) = 0.; p1(2) = -nz; p1(3) = ny;
    p1(4) = 0.;
      
    p2(0) = U(2) * nx - U(0) * nz;
    p2(1) = nz; p2(2) = 0.; p2(3) = -nx;
    p2(4) = 0.;
      
    p3(0) = U(0) * ny - U(1) * nx;
    p3(1) = -ny; p3(2) = nx; p3(3) = 0.;
    p3(4) = 0.;
  }

  if (D == 2) {
    eigvl.Row(0) = p0;
    eigvl.Row(1) = -p3;
    eigvl.Row(2) = 0.5 * c2inv * (b2 + c * b1);
    eigvl.Row(3) = 0.5 * c2inv * (b2 - c * b1);
  } else if (D == 3) {
    eigvl.Row(0) = nx * p0 - cinv * p1;
    eigvl.Row(1) = ny * p0 - cinv * p2;
    eigvl.Row(2) = nz * p0 - cinv * p3;
    eigvl.Row(3) = 0.5 * c2inv * (b2 + c * b1);
    eigvl.Row(4) = 0.5 * c2inv * (b2 - c * b1);
  }

  // HACK FOR NOW (must be generalized for n passive scalars)
  if (COMP > D+2) {

    Mat<COMP, COMP, SCAL> L(eigvl), R(eigvr);
    eigvl = 0.; eigvr = 0.;

    for (int i = 0; i < D+2; ++i)
      for (int j = 0; j < D; ++j)
        eigvr(i, j) = R(i, j);
    for (int i = 0; i < D+2; ++i)
      for (int j = D; j < D+2; ++j)
        eigvr(i, j+1) = R(i, j);

    for (int i = 0; i < D+2; ++i)
      for (int j = 0; j < D; ++j)
        eigvl(j, i) = L(j, i);
    for (int i = 0; i < D+2; ++i)
      for (int j = D; j < D+2; ++j)
        eigvl(j+1, i) = L(j, i);
    
    // eigvr.Rows(0, D+2).Cols(0, D)     = R.Rows(0, D+2).Cols(0, D);
    // eigvr.Rows(0, D+2).Cols(D+1, D+3) = R.Rows(0, D+2).Cols(D, D+2);

    // eigvl.Cols(0, D+2).Rows(0, D)     = L.Cols(0, D+2).Rows(0, D);
    // eigvl.Cols(0, D+2).Rows(D+1, D+3) = L.Cols(0, D+2).Rows(D, D+2);

    SCAL nu_t = state(D+2) / state(0);
  
    eigvr(D+2, D) = 1.;
    eigvr(D+2, D+1) = nu_t;
    eigvr(D+2, D+2) = nu_t;

    eigvl.Row(D)  = nu_t * eigvl.Row(0);
    eigvl(D, 0)   = eigvl(D, 0) - nu_t;
    eigvl(D, D+2) = 1.;
  }
  
}
