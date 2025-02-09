template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::SolveLocalAdjointSystem(const int i, vector<double> & vecQ, vector<double> & vecW) {

  ElementData<D, COMP> & ed = *eldata[i];

  int ndof_q  = ed.ndof_q;
  int ndof_w  = ed.ndof_w;
  int ndof_lt = ed.ndof_lt; // Contains the accumulated number of hybrid DOFs of all element faces

  fill(&vecQ[0], &vecQ[0] + COMP * ndof_q * (1 + COMP * ndof_lt), 0.0);
  fill(&vecW[0], &vecW[0] + COMP * ndof_w * (1 + COMP * ndof_lt), 0.0);

  fill(&matB[0], &matB[0] + COMP * ndof_q * COMP * ndof_w, 0.0);
  fill(&matC[0], &matC[0] + COMP * ndof_w * COMP * ndof_q, 0.0);
  fill(&matD[0], &matD[0] + COMP * ndof_w * COMP * ndof_w, 0.0);

  fill(&matL[0], &matL[0] + COMP * ndof_lt * COMP * ndof_q, 0.0);
  fill(&matM[0], &matM[0] + COMP * ndof_lt * COMP * ndof_w, 0.0);

#ifdef BR2
  AssembleBR2(i, true, 0);
#endif
  
  AssembleLocalAdjointRhs(i, vecQ, vecW);
  AssembleLocalMatrix(i, matB, matC, matD, vecQ, vecW, matL, matM);

  // Write [L^T, M^T] into [R, S]

  int offset_lq = 0;
  int offset_lw = 0;

  for (int ff = 0; ff < ed.nf; ++ff) {

    FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];
    int ndof_l = fd.ndof_l;
    
    if (ndof_l == 0) continue;

    if (Model::Diffusion || Model::Source)
      for (int qq = 0; qq < COMP * ndof_l; ++qq)
        for (int pp = 0; pp < COMP * ndof_q; ++pp)
          vecQ[offset_lq + COMP * ndof_q * (1 + qq) + pp] = -matL[offset_lq + COMP * ndof_l * pp + qq];

    for (int qq = 0; qq < COMP * ndof_l; ++qq)
      for (int pp = 0; pp < COMP * ndof_w; ++pp)
        vecW[offset_lw + COMP * ndof_w * (1 + qq) + pp] = -matM[offset_lw + COMP * ndof_l * pp + qq];

    offset_lw += COMP * ndof_l * COMP * ndof_w;
    offset_lq += COMP * ndof_l * COMP * ndof_q;
  }

  // Compute Schur complement (eliminate the gradient locally)
  if (Model::Diffusion || Model::Source) {
    // A^{-1} * B
    mygemm('n', 'n', ndof_w, COMP * D * ndof_w * COMP, ndof_w, 1., &ed.inv_mass[0], &matB[0], 0., &temp[0]);    

    // D - C * A^{-1} * B
    mygemm('n', 'n', COMP * ndof_w, COMP * ndof_w, COMP * ndof_q, -1., &matC[0], &temp[0], 1., &matD[0]);

    // G - M^T - (A^{-1} * B)^T * (F - L^T)
    mygemm('t', 'n', COMP * ndof_w, 1 + COMP * ndof_lt, COMP * ndof_q, -1., &temp[0], &vecQ[0], 1., &vecW[0]);
  }

  // Solve (D - C * A^{-1} * B)^T * W = G - M^T - (A^{-1} * B)^T * (F - L^T)
  int info;
  int size = COMP * ndof_w;
  char trans = 't';
  vector<int> piv(size);
  int nrhs = 1 + COMP * ndof_lt;
  dgetrf_(&size, &size, &matD[0], &size, &piv[0], &info);
  dgetrs_(&trans, &size, &nrhs, &matD[0], &size, &piv[0], &vecW[0], &size, &info);

  // Backward substitution
  if (Model::Diffusion || Model::Source) {
    // F - L - C^T * W
    mygemm('t', 'n', COMP * ndof_q, 1 + COMP * ndof_lt, COMP * ndof_w, -1., &matC[0], &vecW[0], 1., &vecQ[0]);

    // A^{-T} * (F - L - C^T * W)
    mygemm('n', 'n', ndof_w, COMP * D * (1 + COMP * ndof_lt), ndof_w, 1., &ed.inv_mass[0], &vecQ[0], 0., &temp[0]);    

    copy(&temp[0], &temp[0] + COMP * ndof_q * (1 + COMP * ndof_lt), &vecQ[0]);
  }
}
