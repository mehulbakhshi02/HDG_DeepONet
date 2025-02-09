template <int D, int COMP>
template <typename SCAL>
void CompressibleEuler<D, COMP>
::EvalBdryState(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam,
                          Vec<COMP, SCAL> & bcstate) {

  switch (bcnr) {
    case 0: // far field
      EvalFarfield(state, sparam, bcstate);
      break;
    case 1: // slip wall
      EvalSlipwall(state, sparam, bcstate);
      break;
    case 2: // symmetry
      EvalSlipwall(state, sparam, bcstate);
      break;
    case 3:
      EvalInflow(state, sparam, bcstate);
      break;
    case 4:
      EvalOutflow(state, sparam, bcstate);
      break;
    default:
      cout << endl << "[CompressibleEuler::EvalBdryState] Unknown boundary condition: " << bcnr << endl;
      exit(0);
  }
           
}
template <int D, int COMP>
void CompressibleEuler<D, COMP>
::GetBCName(int bcnr, stringstream & oss) {
    switch (bcnr) {
    case 0: // Far field
      oss << "Far field";
      break;
    case 1: // No-slip wall (adiabatic)
      oss << "Slip wall";
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
    default:
      cout << endl << "[CompressibleEuler::EvalBdryState] Unknown boundary condition: " << bcnr << endl;
      exit(0);
  }
}
