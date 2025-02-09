template <int D, int COMP>
void CompressibleEuler<D, COMP>
::LoadParameters(shared_ptr<PDE> apde) {

  if (apde->ConstantUsed("mach"))
    mach = apde->GetConstant("mach");
  else {
    cout << "Please provide a Mach number as input" << endl;
    exit(0);
  }

  if (apde->ConstantUsed("alpha"))
    angle(0) = apde->GetConstant("alpha") * M_PI / 180.;
  if (apde->ConstantUsed("beta") && D == 3)
    angle(1) = apde->GetConstant("beta") * M_PI / 180.;

  if (apde->ConstantUsed("gamma"))
    gamma = apde->GetConstant("gamma");

  if (apde->ConstantUsed("chord"))
    chord = apde->GetConstant("chord");

  if (apde->ConstantUsed("rho_infty"))
    rho_infty = apde->GetConstant("rho_infty");

  // Compute dependent variables
  c_infty   = sqrt(gamma * p_infty / rho_infty);

  Vec<D> dir;
  if (D == 2) {
    dir(0) = cos(angle(0));
    dir(1) = sin(angle(0));
  } else if (D == 3) {
    dir(0) = cos(angle(0)) * cos(angle(1));
    dir(1) = cos(angle(0)) * sin(angle(1));
    dir(2) = sin(angle(0));
  }

  v_infty = mach * c_infty * dir;

  E_infty = p_infty / (gamma - 1.) + 0.5 * rho_infty * pow(mach * c_infty, 2.);
  C_infty = 0.5 * gamma * mach * mach * p_infty * chord;		
  H_infty = gamma/(gamma-1.) * p_infty / rho_infty + 0.5 * pow(mach * c_infty, 2.);
  cvTtot  = (1. + 0.5 * (gamma-1.) * pow(mach, 2.)) * p_infty / (rho_infty * (gamma-1.));
  ptot    = p_infty * pow(1. + 0.5 * (gamma-1.) * pow(mach, 2.), gamma/(gamma-1.));

  w_infty(0)           = rho_infty;
  w_infty.Rows(1, D+1) = rho_infty * v_infty;
  w_infty(D+1)         = E_infty;
}  
