// **********************************************************************
// Reconstruct the coefficient vector of Lambda and those of w and Sigma
// using the local solves
// **********************************************************************

template<int D, int COMP, class Model>
double UnifyingFramework<D, COMP, Model>
::UpdateSolution(Solution & delta, Solution & sol) {

  assemblyMode = AM_BackSolvePrimal;

  vector<double> LocalLambda(COMP * nf_max * ndof_l_max);
  
  Residual res;
  res.resQ = 0.; res.resW = 0.;
  
  // Compute \delta w and \delta q using the local solves
  for (int i = 0; i < ne; ++i) {
    ElementData<D,COMP> & ed = *eldata[i];    
    
    // Obtain the local solves
    int ndof_w = ed.ndof_w;
    int ndof_q = ed.ndof_q;
    int ndof_lt = ed.ndof_lt;
    
    SolveLocalSystem(i, matQ, matW, res);

    copy(&matW[0], &matW[0] + COMP * ndof_w, &delta.vecW[COMP * ed.offset_w]);
    if (Model::Diffusion || Model::Source)
      copy(&matQ[0], &matQ[0] + COMP * ndof_q, &delta.vecQ[COMP * ed.offset_q]);

    // Collect lambda from the element faces
    int index = 0;
    for (int ff = 0; ff < ed.nf; ++ff) {
      int facet_number = ed.faces[ff];
      FacetData<D,COMP> & fd = *fadata[facet_number];
      int ndof_l = fd.ndof_l;
      int offset = COMP * fd.offset_l;

      // No boundary faces
      if (fd.ndof_l == 0)
        continue;

      copy(&delta.vecL[offset], &delta.vecL[offset] + COMP * ndof_l, &LocalLambda[index]);

      index += COMP * ndof_l;
    }
        
    mygemv('n', COMP * ndof_w, COMP * ndof_lt, 1., &matW[COMP * ndof_w], &LocalLambda[0], 1., &delta.vecW[COMP*ed.offset_w]);
    if (Model::Diffusion || Model::Source)
      mygemv('n', COMP * ndof_q, COMP * ndof_lt, 1., &matQ[COMP * ndof_q], &LocalLambda[0], 1., &delta.vecQ[COMP*ed.offset_q]);
  }

  res.resW = sqrt(res.resW);

  double alpha = 1.;

  if (SolverData::limit_update)
    LimitUpdate(alpha, delta, sol);

  if (SolverData::check_physics)
    CheckUpdate(alpha, res, delta, sol);

  if (SolverData::limit_update || SolverData::check_physics || SolverData::line_search) {
    cout.precision(2); cout.setf(ios::fixed, ios::floatfield); 
    cout << setw(8) << alpha;
    cout.precision(8); cout.setf(ios::scientific, ios::floatfield); 
  }

  if (alpha <= SolverData::min_update) {
    if (pcfl == 1.)
      sol = sol_save;
    else
      pcfl = max(SolverData::pcfl_reduce * pcfl, 1.);
  } else {

    // In case of full update
    if (alpha == 1.) {

      // If we are in the Newton regime
      if (res.resW < SolverData::full_newton)
        pcfl *= max(SolverData::pcfl_newton, SolverData::pcfl_beta);
      else
        pcfl *= SolverData::pcfl_beta;

      pcfl = min(pcfl, SolverData::pcfl_max);
      
      sol_save = sol;
    }
    
    // Compute the new coefficient vector of lambda
    int size = COMP * ndof_l_total;
    int ione = 1;
    daxpy_(&size, &alpha, &delta.vecL[0], &ione, &sol.vecL[0], &ione);

    // Compute the new coefficient vector of w and q
    int size_w = COMP * ndof_w_total;
    int size_q = COMP * ndof_q_total;
    daxpy_(&size_w, &alpha, &delta.vecW[0], &ione, &sol.vecW[0], &ione);
    if (Model::Diffusion || Model::Source)
      daxpy_(&size_q, &alpha, &delta.vecQ[0], &ione, &sol.vecQ[0], &ione);
  }

  return alpha;
}
