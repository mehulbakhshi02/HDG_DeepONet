template <int D, int COMP>
template <typename SCAL>
void CompressibleNavierStokes<D, COMP>
::EvalBdryState(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate) {

  switch (bcnr) {
    case 0: // Far field
      EvalFarfield(state, sparam, bcstate);
      break;
    case 1: // No-slip wall (adiabatic)
      EvalNoslipwallAdiabatic(state, sparam, bcstate);
      break;
    case 2: // Symmetry
      EvalSlipwall(state, sparam, bcstate);
      break;
    case 3: // Inflow
      EvalInflow(state, sparam, bcstate);
      break;
    case 4: // Outflow
      EvalOutflow(state, sparam, bcstate);
      break;
    case 5: // No-slip wall (isothermal)
      EvalNoslipwallIsothermal(state, sparam, bcstate);
      break;
    default:
      cout << endl << "[CompressibleNavierStokes::EvalBdryState] Unknown boundary condition: " << bcnr << endl;
      exit(0);
  }
           
}

template <int D, int COMP>
template <typename SCAL>
void CompressibleNavierStokes<D, COMP>
::EvalBdryGradient(int bcnr, Vec<COMP, SCAL> & state, Mat<COMP, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<COMP, D, SCAL> & bcgrad) {

  bcgrad = grad;

  switch (bcnr) {
    case 0: // Far field
    case 1: // No-slip wall (adiabatic)
      break;
    case 2: { // symmetry

      Mat<D, D> N = sparam.normal * Trans(sparam.normal);
      Mat<D, D> A = Id<D>() - N;
    
      bcgrad.Row(0)          = A * grad.Row(0); 
      bcgrad.Rows(1, D+1)    = A * grad.Rows(1, D+1) * A + N * grad.Rows(1, D+1) * N;
      for (int i = D+1; i < COMP; ++i)
        bcgrad.Row(i) = A * grad.Row(i);
      
      break; }
    case 3: // Inflow
    case 4: // Outflow
    case 5: // No-slip wall (isothermal)
      break;
    default:
      cout << endl << "[CompressibleNavierStokes::EvalBdryGradient] Unknown boundary condition: " << bcnr << endl;
      exit(0);
  }
           
}
template <int D, int COMP>
void CompressibleNavierStokes<D, COMP>
::GetBCName(int bcnr, stringstream & oss) {
    switch (bcnr) {
    case 0: // Far field
      oss << "Far field";
      break;
    case 1: // No-slip wall (adiabatic)
      oss << "Adiabatic wall";
      break;
    case 2: // Symmetry
      oss << "Symmetry";
      break;
    case 3: // Inflow
      oss << "Inflow";
      break;
    case 4: // Outflow
      oss << "Outflow";
      break;
    case 5: // No-slip wall (isothermal)
      oss << "Isothermal wall";
      break;
    default:
      cout << endl << "[CompressibleNavierStokes::EvalBdryState] Unknown boundary condition: " << bcnr << endl;
      exit(0);
  }
}
