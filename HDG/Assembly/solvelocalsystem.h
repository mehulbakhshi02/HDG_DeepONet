template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::SolveLocalSystem(const int i, vector<double> & vecQ, vector<double> & vecW, Residual & res) {

  ElementData<D, COMP> & ed = *eldata[i];

  int ndof_q  = ed.ndof_q;
  int ndof_w  = ed.ndof_w;
  int ndof_lt = ed.ndof_lt; // Contains the accumulated number of hybrid DOFs of all element faces

  fill(&vecQ[0], &vecQ[0] + COMP * ndof_q * (1 + COMP * ndof_lt), 0.0);
  fill(&vecW[0], &vecW[0] + COMP * ndof_w * (1 + COMP * ndof_lt), 0.0);

  fill(&matB[0], &matB[0] + COMP * ndof_q * COMP * ndof_w, 0.0);
  fill(&matC[0], &matC[0] + COMP * ndof_w * COMP * ndof_q, 0.0);
  fill(&matD[0], &matD[0] + COMP * ndof_w * COMP * ndof_w, 0.0);

  if (assemblyMode == AM_AssemblyPrimal || assemblyMode == AM_AssemblyAdjoint) {
    fill(&matL[0], &matL[0] + COMP * ndof_lt * COMP * ndof_q, 0.0);
    fill(&matM[0], &matM[0] + COMP * ndof_lt * COMP * ndof_w, 0.0);
  }

#ifdef BR2
  AssembleBR2(i, true, 0);
#endif

  AssembleLocalResidual(i, vecQ, vecW);
  AssembleLocalMatrix(i, matB, matC, matD, vecQ, vecW, matL, matM);
  
  // Compute the L2 norm of the residual
  for (int ll = 0, index = 0; ll < COMP; ++ll)
    for (int pp = 0; pp < ndof_q; ++pp, ++index)
      res.resQ += pow(residual_scale[ll] * vecQ[index], 2.0);

  for (int ll = 0, index = 0; ll < COMP; ++ll) {
    double temp = 0.;
    for (int pp = 0; pp < ndof_w; ++pp, ++index)
      temp += pow(residual_scale[ll] * vecW[index], 2.);
    res.resW += temp;
    residual[ll] += temp;
  }

  // Compute Schur complement (eliminate the gradient locally)
  if (Model::Diffusion || Model::Source) {
    // A^{-1} * B
    mygemm('n', 'n', ndof_w, COMP * D * ndof_w * COMP, ndof_w, 1., &ed.inv_mass[0], &matB[0], 0., &temp[0]);

    // D - C * A^{-1} * B
    mygemm('n', 'n', COMP * ndof_w, COMP * ndof_w, COMP * ndof_q, -1., &matC[0], &temp[0], 1., &matD[0]);    

    // G - S - C * A^{-1} * (F - R)
    mygemm('n', 'n', ndof_w, COMP * D * (1 + COMP * ndof_lt), ndof_w, 1., &ed.inv_mass[0], &vecQ[0], 0., &temp[0]);
    mygemm('n', 'n', COMP * ndof_w, 1 + COMP * ndof_lt, COMP * ndof_q, -1., &matC[0], &temp[0], 1., &vecW[0]);
  }

  // Solve (D - C * A^{-1} * B) * W = G - S - C * A^{-1} * (F - R)
  int info;
  int size = COMP * ndof_w;
  int nrhs = 1 + COMP * ndof_lt;
  vector<int> piv(size);
  dgesv_(&size, &nrhs, &matD[0], &size, &piv[0], &vecW[0], &size, &info);

  // Backward substitution
  if (Model::Diffusion || Model::Source) {
    // F - R - B * W
    mygemm('n', 'n', COMP * ndof_q, 1 + COMP * ndof_lt, COMP * ndof_w, -1., &matB[0], &vecW[0], 1., &vecQ[0]);

    // A^{-1} * (F - R - B * W)
    mygemm('n', 'n', ndof_w, COMP * D * (1 + COMP * ndof_lt), ndof_w, 1., &ed.inv_mass[0], &vecQ[0], 0., &temp[0]);

    copy(&temp[0], &temp[0] + COMP * ndof_q * (1 + COMP * ndof_lt), &vecQ[0]);
  }
}
