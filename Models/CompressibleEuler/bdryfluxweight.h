
template <int D, int COMP>
template <typename SCAL>
void CompressibleEuler<D, COMP>
::EvalBdryCoefficients(int bcnr, Vec<COMP, SCAL> & fcn, Vec<COMP, SCAL> & fvn, SpatialParams<D> & sparam, Vec<NumBdryCoeffs, SCAL> & res) {

  SCAL p(0.);
  for (int dd = 0; dd < D; ++dd)
    p += fcn(dd+1) * sparam.normal(dd);

  res(0) = 2. * (p - p_infty) / (rho_infty * pow(mach * c_infty, 2.));
  
}

template <int D, int COMP>
void CompressibleEuler<D, COMP>
::EvalBdryFluxWeight(int bcnr, SpatialParams<D> & sparam, Mat<NumBdryFluxWeight, COMP> & res) {

  res = 0.;
  if (bcnr != 1 && bcnr != 5) return;

  if (D == 2) {
    // Drag
    res(0, 1) = cos(angle(0));
    res(0, 2) = sin(angle(0));

    // // Lift
    // res(1, 1) = -sin(angle(0));
    // res(1, 2) = cos(angle(0));

  } else if (D == 3) {
    // Drag
    res(0, 1) = cos(angle(0)) * cos(angle(1));
    res(0, 2) = cos(angle(0)) * sin(angle(1));
    res(0, 3) = sin(angle(0));

    // Lift
    res(1, 1) = -sin(angle(0)) * cos(angle(1));
    res(1, 2) = -sin(angle(0)) * sin(angle(1));
    res(1, 3) = cos(angle(0));
  }

  res /= C_infty;
}
