template <int D, int COMP>
template <typename SCAL>
void CompressibleSpalartAllmaras<D, COMP>
::EvalBdryState(int bcnr, Vec<COMP, SCAL> & state, SpatialParams<D> & sparam, Vec<COMP, SCAL> & bcstate) {

  switch (bcnr) {
    case 0: // Far field
      EvalFarfield(state, sparam, bcstate);
      break;
    case 1: // No-slip wall (adiabatic)
      EvalNoslipwallAdiabatic(state, sparam, bcstate);
      break;
    case 2: // Symmetry
      bcstate = state;
      bcstate.Rows(1, D+1) = (Id<D>() - sparam.normal * Trans(sparam.normal)) * state.Rows(1, D+1);
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
      cout << endl << "[CompressibleSpalartAllmaras::EvalBdryState] Unknown boundary condition: " << bcnr << endl;
      exit(0);
  }
}
