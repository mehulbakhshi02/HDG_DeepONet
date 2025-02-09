template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalBdryState(int bcnr, Vec<D+4, SCAL> & state, SpatialParams<D> & sparam, Vec<D+4, SCAL> & bcstate) {

  switch (bcnr) {
    case 0: // Far field
      EvalFarfield(state, sparam, bcstate);
      break;
    case 1: // No-slip wall (adiabatic)
      EvalNoslipwallAdiabatic(state, sparam, bcstate);
      break;
    case 2: // Symmetry
      bcstate = state;
      bcstate.Rows(1, D+1) = (Id<D>() - sparam.normal * Trans(sparam.normal)) * state.Rows(1,D+1);
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
      cout << endl << "[CompressibleKOmega::EvalBdryState] Unknown boundary condition: " << bcnr << endl;
      exit(0);
  }
           
}

template <int D>
template <typename SCAL>
void CompressibleKOmega<D>
::EvalBdryGradient(int bcnr, Vec<D+4, SCAL> & state, Mat<D+4, D, SCAL> & grad, SpatialParams<D> & sparam, Mat<D+4, D, SCAL> & bcgrad) {

  bcgrad = grad;

  switch (bcnr) {
    case 0: // Far field
    case 1: // No-slip wall (adiabatic)
      break;
    case 2: { // symmetry
      Mat<D, D> N = sparam.normal * Trans(sparam.normal);
      Mat<D, D> A = Id<D>() - N;
    
      bcgrad.Row(0)         = A * grad.Row(0); 
      bcgrad.Rows(1, D+1)   = A * grad.Rows(1, D+1) * A + N * grad.Rows(1, D+1) * N;
      bcgrad.Rows(D+1, D+4) = grad.Rows(D+1, D+4) * A;
      
      break; }
    case 3: // Inflow
    case 4: // Outflow
    case 5: // No-slip wall (isothermal)
      break;
    default:
      cout << endl << "[CompressibleKOmega::EvalBdryGradient] Unknown boundary condition: " << bcnr << endl;
      exit(0);
  }
           
}
