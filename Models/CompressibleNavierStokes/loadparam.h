template <int D, int COMP>
void CompressibleNavierStokes<D, COMP>
::LoadParameters(shared_ptr<PDE> apde) {

  CompressibleEuler<D, COMP>::LoadParameters(apde);

  if (apde->ConstantUsed("reynolds"))
    reynolds = apde->GetConstant("reynolds");
  else {
    cout << "Please provide a Reynolds number as input" << endl;
    exit(0);
  }

  if (apde->ConstantUsed("prandtl"))
    prandtl = apde->GetConstant("prandtl");

  if (apde->ConstantUsed("ref_temp"))
    T0 = apde->GetConstant("ref_temp");

  if (apde->ConstantUsed("stab_visc"))
    stab_visc = apde->GetConstant("stab_visc");

  mu0      = suth_a1 * pow(T0, 1.5) / (T0 + suth_a2);
  rr       = T0;
  amu      = sqrt(gamma) * mach * chord / reynolds;

}
