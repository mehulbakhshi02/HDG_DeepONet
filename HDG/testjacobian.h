template <int D, int COMP, class Model>
double UnifyingFramework<D, COMP, Model>
::TestJacobian(Solution & sol, double eps) {

  Solution solL = sol;
  Solution solR = sol;
  Solution del  = sol;
  Solution rhs = sol;  
  
  double * vecWL = &solL.vecW[0];
  double * vecQL = &solL.vecQ[0];
  double * vecLL = &solL.vecL[0];

  double * vecWR = &solR.vecW[0];
  double * vecQR = &solR.vecQ[0];
  double * vecLR = &solR.vecL[0];
  
  double * delW  = &del.vecW[0];
  double * delQ  = &del.vecQ[0];
  double * delL  = &del.vecL[0];
  
  for (int pp = 0; pp < ndof_w_total * COMP; ++pp) {
    double temp = double(rand()) / RAND_MAX;
    vecWR[pp] += eps * temp;
    vecWL[pp] -= eps * temp;
    delW[pp]   = temp;
  }

  for (int pp = 0; pp < ndof_q_total * COMP; ++pp) {
    double temp = double(rand()) / RAND_MAX;
    vecQR[pp] += eps * temp;
    vecQL[pp] -= eps * temp;
    delQ[pp]   = temp;
  }

  for (int pp = 0; pp < ndof_l_total * COMP; ++pp) {
    double temp = double(rand()) / RAND_MAX;
    vecLR[pp] += eps * temp;
    vecLL[pp] -= eps * temp;
    delL[pp]   = temp;
  }  

  petsc_matrix_reset(); petsc_rhs_reset();

  assemblyMode = AM_AssemblyPrimal;

  ReconstructSolution(solR);

  for (int i = 0; i < ne; ++i) {
    ElementData<D,COMP> & ed = *eldata[i];
  
    int ndof_q = ed.ndof_q;
    int ndof_w = ed.ndof_w;

    int index_w = ed.offset_w * COMP;
    int index_q = ed.offset_q * COMP;
    
    fill(&matQ[0], &matQ[0] + COMP * ndof_q, 0.0);
    fill(&matW[0], &matW[0] + COMP * ndof_w, 0.0);

#ifdef BR2
    AssembleBR2(i, false, 0);
#endif
    
    AssembleLocalResidual(i, matQ, matW);

    for (int p = 0; p < COMP * ndof_w; ++p)
      rhs.vecW[index_w+p] = matW[p];
    for (int p = 0; p < COMP * ndof_q; ++p)
      rhs.vecQ[index_q+p] = matQ[p];

  }

  petsc_rhs_scale(-1.);

  ReconstructSolution(solL);

  for (int i = 0; i < ne; ++i) {
    ElementData<D,COMP> & ed = *eldata[i];
  
    int ndof_q = ed.ndof_q;
    int ndof_w = ed.ndof_w;

    int index_w = ed.offset_w * COMP;
    int index_q = ed.offset_q * COMP;
    
    fill(&matQ[0], &matQ[0] + COMP * ndof_q, 0.0);
    fill(&matW[0], &matW[0] + COMP * ndof_w, 0.0);

#ifdef BR2
    AssembleBR2(i, false, 0);
#endif
    
    AssembleLocalResidual(i, matQ, matW);

    for (int pp = 0; pp < ndof_w * COMP; ++pp)
      rhs.vecW[index_w+pp] = -(rhs.vecW[index_w + pp]-matW[pp]);
    for (int pp = 0; pp < ndof_q * COMP; ++pp)
      rhs.vecQ[index_q+pp] = -(rhs.vecQ[index_q + pp]-matQ[pp]);
  }  

  ReconstructSolution(sol);

  for (int i = 0; i < ne; ++i) {

    ElementData<D, COMP> & ed = *eldata[i];

    int ndof_q = ed.ndof_q;
    int ndof_w = ed.ndof_w;
    int ndof_lt = ed.ndof_lt; // Contains the accumulated number of hybrid DOFs of all element faces

    int index_w = ed.offset_w * COMP;
    int index_q = ed.offset_q * COMP;

    fill(&matQ[0], &matQ[0] + COMP * ndof_q * (1 + COMP * ndof_lt), 0.0);
    fill(&matW[0], &matW[0] + COMP * ndof_w * (1 + COMP * ndof_lt), 0.0);

    //    fill(&matA[0], &matA[0] + COMP * ndof_q * COMP * ndof_q, 0.0);
    fill(&matB[0], &matB[0] + COMP * ndof_q * COMP * ndof_w, 0.0);
    fill(&matC[0], &matC[0] + COMP * ndof_w * COMP * ndof_q, 0.0);
    fill(&matD[0], &matD[0] + COMP * ndof_w * COMP * ndof_w, 0.0);
    
    fill(&matL[0], &matL[0] + COMP * ndof_lt * COMP * ndof_q, 0.0);
    fill(&matM[0], &matM[0] + COMP * ndof_lt * COMP * ndof_w, 0.0);    

#ifdef BR2
    AssembleBR2(i, true, 0);
#endif
    
    //    AssembleLocalResidual(i, matQ, matW);
    AssembleLocalMatrix(i, matB, matC, matD, matQ, matW, matL, matM);

    for (int pp = 0; pp < ndof_w * COMP; ++pp)
      matW[pp] = rhs.vecW[index_w+pp];
    for (int pp = 0; pp < ndof_q * COMP; ++pp)
      matQ[pp] = rhs.vecQ[index_q+pp];
    
    // Compute Schur complement (eliminate the gradient locally)
    if (Model::Diffusion || Model::Source) {
      // for (int dd = 0; dd < D; ++dd)
      //   for (int ll = 0; ll < COMP; ++ll)
      //     for (int pp = 0; pp < ndof_w; ++pp)
      //       for (int qq = 0; qq < ndof_w; ++qq)
      //         matA[qq+ndof_w*(dd+D*(ll+COMP*(pp+ndof_w*(dd+D*ll))))] = ed.inv_mass[pp*ndof_w+qq];

      // A^{-1} * B
      mygemm('n', 'n', ndof_w, COMP * D * ndof_w * COMP, ndof_w, 1., &ed.inv_mass[0], &matB[0], 0., &temp[0]);

      // D - C * A^{-1} * B
      mygemm('n', 'n', COMP * ndof_w, COMP * ndof_w, COMP * ndof_q, -1., &matC[0], &temp[0], 1., &matD[0]);    

      // G - S - C * A^{-1} * (F - R)
      mygemm('n', 'n', ndof_w, COMP * D * (1 + COMP * ndof_lt), ndof_w, 1., &ed.inv_mass[0], &matQ[0], 0., &temp[0]);
      mygemm('n', 'n', COMP * ndof_w, 1 + COMP * ndof_lt, COMP * ndof_q, -1., &matC[0], &temp[0], 1., &matW[0]);

      // C * A^{-1}
      //      mygemm('n', 'n', COMP * ndof_w, COMP * ndof_q, COMP * ndof_q, 1., &matC[0], &matA[0], 0., &temp[0]);

      // D - C * A^{-1} * B
      //      mygemm('n', 'n', COMP * ndof_w, COMP * ndof_w, COMP * ndof_q, -1., &temp[0], &matB[0], 1., &matD[0]);

      // G - S - C  * A^{-1} * (F - R)
      //      mygemm('n', 'n', COMP * ndof_w, 1 + COMP * ndof_lt, COMP * ndof_q, -1., &temp[0], &matQ[0], 1., &matW[0]);
    }
    
    int size   = COMP * ndof_w;
    int nrhs   = 1 + ndof_lt * COMP;
    int info;
    vector<int> piv(size);
    dgesv_(&size, &nrhs, &matD[0], &size, &piv[0], &matW[0], &size, &info);

    // Backward substitution
    if (Model::Diffusion || Model::Source) {
      // F - R - B * W
      mygemm('n', 'n', COMP * ndof_q, 1 + COMP * ndof_lt, COMP * ndof_w, -1., &matB[0], &matW[0], 1., &matQ[0]);

      // A^{-1} * (F - R - B * W)
      //      mygemm('n', 'n', COMP * ndof_q, 1 + COMP * ndof_lt, COMP * ndof_q, 1., &matA[0], &matQ[0], 0., &temp[0]);
      mygemm('n', 'n', ndof_w, COMP * D * (1 + COMP * ndof_lt), ndof_w, 1., &ed.inv_mass[0], &matQ[0], 0., &temp[0]);      

      copy(&temp[0], &temp[0] + COMP * ndof_q * (1 + COMP * ndof_lt), &matQ[0]);
    }    

    AssembleHybridSystemLocal(i, matQ, matW, matL, matM);  
  }

  petsc_rhs_scale(-1./(2. * eps));

  petsc_set_startvector(&delL[0]);

  petsc_begin_mat_alloc(); petsc_end_mat_alloc();
  petsc_begin_vec_alloc(); petsc_end_vec_alloc();

  //petsc_show_matrix();
  //petsc_show_rhs();

  assemblyMode = AM_None;
  
  // Compute N*dL-H
  return petsc_test_solution();
  
  
}
