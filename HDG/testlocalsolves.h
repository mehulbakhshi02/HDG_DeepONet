template <int D, int COMP, class Model>
double UnifyingFramework<D, COMP, Model>
::TestLocalSolves(Solution & sol, double eps) {

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

  assemblyMode = AM_None;

  // Compute the local residuals for solR
  ReconstructSolution(solR);  

  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];

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
      rhs.vecW[index_w+pp] = matW[pp];
    for (int pp = 0; pp < ndof_q * COMP; ++pp)
      rhs.vecQ[index_q+pp] = matQ[pp];
  }

  ReconstructSolution(solL);

  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];

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
      rhs.vecW[index_w+pp] = (rhs.vecW[index_w+pp] - matW[pp]) / (2. * eps);
    for (int pp = 0; pp < ndof_q * COMP; ++pp)
      rhs.vecQ[index_q+pp] = (rhs.vecQ[index_q+pp] - matQ[pp]) / (2. * eps);
  }  

  // Compute the local residual and jacobian for sol
  ReconstructSolution(sol);

  vector<double> LocalLambda(COMP * nf_max * ndof_l_max);  
  
  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];

    int ndof_q = ed.ndof_q;
    int ndof_w = ed.ndof_w;
    int ndof_lt = ed.ndof_lt;

    int index_q = ed.offset_q * COMP;
    int index_w = ed.offset_w * COMP;

    fill(&matQ[0], &matQ[0] + COMP * ndof_q * (1 + COMP * ndof_lt), 0.0);
    fill(&matW[0], &matW[0] + COMP * ndof_w * (1 + COMP * ndof_lt), 0.0);

    //    fill(&matA[0], &matA[0] + COMP * ndof_q * COMP * ndof_q, 0.0);
    fill(&matB[0], &matB[0] + COMP * ndof_q * COMP * ndof_w, 0.0);
    fill(&matC[0], &matC[0] + COMP * ndof_w * COMP * ndof_q, 0.0);
    fill(&matD[0], &matD[0] + COMP * ndof_w * COMP * ndof_w, 0.0);

#ifdef BR2
    AssembleBR2(i, true, 0);
#endif
    
    AssembleLocalResidual(i, matQ, matW);
    AssembleLocalMatrix(i, matB, matC, matD, matQ, matW, matL, matM);

    if (Model::Diffusion || Model::Source) {
      // for (int ll = 0; ll < COMP; ++ll)
      //   for (int dd = 0; dd < D; ++dd)
      //     for (int pp = 0; pp < ndof_w; ++pp)
      //       for (int qq = 0; qq < ndof_w; ++qq)
      //         matA[qq+ndof_w*(dd+D*(ll+COMP*(pp+ndof_w*(dd+D*ll))))] = ed.mass[pp*ndof_w+qq];

      mygemm('n', 'n', ndof_w, COMP * D, ndof_w, 1., &ed.mass[0], &del.vecQ[index_q], 1., &rhs.vecQ[index_q]);
      
      //      mygemv('n', COMP * ndof_q, COMP * ndof_q, 1., &matA[0], &del.vecQ[index_q], 1., &rhs.vecQ[index_q]);
      mygemv('n', COMP * ndof_q, COMP * ndof_w, 1., &matB[0], &del.vecW[index_w], 1., &rhs.vecQ[index_q]);
      mygemv('n', COMP * ndof_w, COMP * ndof_q, 1., &matC[0], &del.vecQ[index_q], 1., &rhs.vecW[index_w]);
    }

    mygemv('n', COMP * ndof_w, COMP * ndof_w, 1., &matD[0], &del.vecW[index_w], 1., &rhs.vecW[index_w]);

    int nrhs = 0;
    for (int ff = 0; ff < ed.nf; ++ff) {
      int facet_number = ed.faces[ff];
      FacetData<D,COMP> & fd = *fadata[facet_number];
      int ndof_l = fd.ndof_l;
    
      if (fd.ndof_l == 0)
        continue;
    
      for (int ll = 0; ll < COMP; ++ll)      
        for (int qq = 0; qq < ndof_l; ++qq, ++nrhs)
          LocalLambda[nrhs] = -del.vecL[COMP*fd.offset_l+qq+ndof_l*ll];
    }

    mygemv('n', COMP * ndof_w, nrhs, 1., &matW[COMP * ndof_w], &LocalLambda[0], 1., &rhs.vecW[index_w]);

    if (Model::Diffusion || Model::Source)
      mygemv('n', COMP * ndof_q, nrhs, 1., &matQ[COMP * ndof_q], &LocalLambda[0], 1., &rhs.vecQ[index_q]);    
  }

  double err_norm = 0.;
  for (int pp = 0; pp < ndof_w_total * COMP; ++pp)
    err_norm += pow(rhs.vecW[pp], 2.);
  for (int pp = 0; pp < ndof_q_total * COMP; ++pp)
    err_norm += pow(rhs.vecQ[pp], 2.);
  err_norm = sqrt(err_norm);

  return err_norm;
}
