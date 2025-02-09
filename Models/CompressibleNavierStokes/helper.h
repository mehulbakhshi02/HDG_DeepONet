template <int D, int COMP>
template <typename SCAL>
Mat<D, D, SCAL> CompressibleNavierStokes<D, COMP>
  ::EvalStrainRate(Mat<D, D, SCAL> & gradU) {

  SCAL divU = 0.;
  for (int dd = 0; dd < D; ++dd)
    divU += gradU(dd, dd);
  return gradU + Trans(gradU) - 2. / 3. * divU * Id<D>();
  
}

template <int D, int COMP>
template <typename SCAL>
SCAL  CompressibleNavierStokes<D, COMP>
::EvalPhysVisc(SCAL & rho, SCAL & p) {

  // Sutherland's law
  SCAL TT   = rr * p / rho;
  SCAL OPF  = SCAL(1.5);
  SCAL rmu  = suth_a1 * pow(TT, OPF) / (TT + suth_a2);
  // Some scaling due to non-dimensionalization
  return amu * rmu / mu0;  

}

template <int D, int COMP>
template <typename SCAL>
void  CompressibleNavierStokes<D, COMP>
::EvalPrim(Vec<COMP, SCAL> & state, Vec<D, SCAL> & U, SCAL & p, SCAL & T) {

  SCAL rho       = state(0);
  Vec<D, SCAL> m = state.Rows(1, D+1);
  SCAL E         = state(D+1);

  SCAL rhoinv  = 1. / rho;
  SCAL rhoinv2 = rhoinv * rhoinv;

  U = rhoinv * m;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);

  p = (gamma - 1.) * (E - 0.5 * rho * U2);
  T = gamma / (gamma - 1.0) * p * rhoinv;  

}

template <int D, int COMP>
template <typename SCAL>
void  CompressibleNavierStokes<D, COMP>
::EvalPrim(Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, Vec<D, SCAL> & U, SCAL & p, SCAL & T, Mat<D, D, SCAL> & gradU, Vec<D, SCAL> & gradp, Vec<D, SCAL> & gradT) {

  SCAL rho       = state(0);
  Vec<D, SCAL> m = state.Rows(1, D+1);
  SCAL E         = state(D+1);

  SCAL rhoinv  = 1. / rho;
  SCAL rhoinv2 = rhoinv * rhoinv;

  U = rhoinv * m;

  SCAL U2 = 0.;
  for (int dd = 0; dd < D; ++dd)
    U2 += U(dd) * U(dd);

  p = (gamma - 1.) * (E - 0.5 * rho * U2);
  T = gamma / (gamma - 1.0) * p * rhoinv;

  gradU = rhoinv * (grad.Rows(1, D+1) - U * Trans(grad.Row(0)));
  gradp = (gamma - 1.) * (grad.Row(D+1) - rho * Trans(gradU) * U - 0.5 * U2 * grad.Row(0));
  gradT = gamma / (gamma - 1.) * rhoinv2 * (rho * gradp - p * grad.Row(0));  
  
}
