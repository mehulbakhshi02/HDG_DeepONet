template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::SolveLinearizedSystem(Solution & delta, Residual & res, int & its) {

  petsc_matrix_reset();
  petsc_rhs_reset();

  res.resL = 0.; res.resW = 0.; res.resQ = 0.;

  assemblyMode = AM_AssemblyPrimal;

  for (int i = 0; i < ne; ++i) {
    SolveLocalSystem(i, matQ, matW, res);
    AssembleHybridSystemLocal(i, matQ, matW, matL, matM);
  }

  assemblyMode = AM_None;
  
  petsc_begin_mat_alloc(); petsc_end_mat_alloc();
  petsc_begin_vec_alloc(); petsc_end_vec_alloc();

  // Convergence Check
  petsc_get_res_norm(res.resL);

  if (res.resL > SolverData::newton_res)
    petsc_solve_it(delta.vecL, its);

  res.resW = sqrt(res.resW);
  res.resQ = sqrt(res.resQ);

  cout << setw(8) << its << setw(16) << res.resL << setw(16) << res.resW;
  if (Model::Diffusion || Model::Source)
      cout << setw(16) << res.resQ;
  cout << flush;

}
