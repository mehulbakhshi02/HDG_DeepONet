template <int D, int COMP>
template <typename SCAL>
void CompressibleNavierStokes<D, COMP>
::ApplyK(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, Mat<COMP, D, SCAL> & rhs, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & res) {

    SCAL rho = state(0);
    SCAL E   = state(D+1);

    Vec<D, SCAL> U = (1. / rho) * state.Rows(1, D+1);    

    SCAL U2 = 0.;
    for (int dd = 0; dd < D; ++dd)
      U2 += U(dd) * U(dd);

    SCAL p = (gamma - 1.) * (E - 0.5 * rho * U2);    
 
    SCAL visc = EvalPhysVisc(rho, p);

    double rgm1 = gamma / prandtl;

    SCAL temp = rgm1 * (E / rho - U2);

    double c1 = 4./3., c2 = 2./3., c3 = 1 - c2;

    SCAL u = U(0), v = U(1);
    
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

      res.Col(0) = B11 * rhs.Col(0) + B12 * rhs.Col(1);
      res.Col(1) = B21 * rhs.Col(0) + B22 * rhs.Col(1);

    } else if (D == 3) {
      
      SCAL w = U(2);

      Mat<COMP, COMP, SCAL> B11(0.), B12(0.), B13(0.), B21(0.), B22(0.), B23(0.), B31(0.), B32(0.), B33(0.);

      B11(1, 0) = -c1 * u;
      B11(1, 1) = c1;
      B11(2, 0) = -v;
      B11(2, 2) = 1.;
      B11(3, 0) = -w;
      B11(3, 3) = 1.;
      B11(4, 0) = -(c1 * u * u + v * v + w * w + temp);
      B11(4, 1) = (c1 - rgm1) * u;
      B11(4, 2) = (1. - rgm1) * v;
      B11(4, 3) = (1. - rgm1) * w;
      B11(4, 4) = rgm1;

      B12(1, 0) = c2 * v;
      B12(1, 2) = -c2;
      B12(2, 0) = -u;
      B12(2, 1) = 1.;
      B12(4, 0) = -c3 * v * u;
      B12(4, 1) = v;
      B12(4, 2) = -c2 * u;

      B13(1, 0) = c2 * w;
      B13(1, 3) = -c2;
      B13(3, 0) = -u;
      B13(3, 1) = 1.;
      B13(4, 0) = -c3 * w * u;
      B13(4, 1) = w;
      B13(4, 3) = -c2 * u;

      B21(1, 0) = -v;
      B21(1, 2) = 1.;
      B21(2, 0) = c2 * u;
      B21(2, 1) = -c2;
      B21(4, 0) = -c3 * u * v;
      B21(4, 1) = -c2 * v;
      B21(4, 2) = u;

      B22(1, 0) = -u;
      B22(1, 1) = 1.;
      B22(2, 0) = -c1 * v;
      B22(2, 2) = c1;
      B22(3, 0) = -w;
      B22(3, 3) = 1.;
      B22(4, 0) = -(u * u + c1 * v * v + w * w + temp);
      B22(4, 1) = (1. - rgm1) * u;
      B22(4, 2) = (c1 - rgm1) * v;
      B22(4, 3) = (1. - rgm1) * w;
      B22(4, 4) = rgm1;

      B23(2, 0) = c2 * w;
      B23(2, 3) = -c2;
      B23(3, 0) = -v;
      B23(3, 2) = 1.;
      B23(4, 0) = -c3 * v * w;
      B23(4, 2) = w;
      B23(4, 3) = -c2 * v;

      B31(1, 0) = -w;
      B31(1, 3) = 1.;
      B31(3, 0) = c2 * u;
      B31(3, 1) = -c2;
      B31(4, 0) = -c3 * u * w;
      B31(4, 1) = -c2 * w;
      B31(4, 3) = u;

      B32(2, 0) = -w;
      B32(2, 3) = 1.;
      B32(3, 0) = c2 * v;
      B32(3, 2) = -c2;
      B32(4, 0) = -c3 * v * w;
      B32(4, 2) = -c2 * w;
      B32(4, 3) = v;
      
      B33(1, 0) = -u;
      B33(1, 1) = 1.;
      B33(2, 0) = -v;
      B33(2, 2) = 1.;
      B33(3, 0) = -c1 * w;
      B33(3, 3) = c1;
      B33(4, 0) = -(u * u + v * v + c1 * w * w + temp);
      B33(4, 1) = (1. - rgm1) * u;
      B33(4, 2) = (1. - rgm1) * v;
      B33(4, 3) = (c1 - rgm1) * w;
      B33(4, 4) = rgm1;

      res.Col(0) = B11 * rhs.Col(0) + B12 * rhs.Col(1) + B13 * rhs.Col(2);
      res.Col(1) = B21 * rhs.Col(0) + B22 * rhs.Col(1) + B23 * rhs.Col(2);
      res.Col(2) = B31 * rhs.Col(0) + B32 * rhs.Col(1) + B33 * rhs.Col(2);      
    }

    res *= visc / rho;    
  }
