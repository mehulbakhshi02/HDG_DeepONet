template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::SolveAdjointSystem(const Solution & sol_old, Solution & adj, vector<double> & error, int & its, vector<double> &adj_coordinates, vector<double> &adj_sol, LocalHeap & lh) {
  
  cout << string(32, '*') << endl;
  cout << "Compute Adjoint" << endl;
  cout << string(32, '*') << endl;

  order++;
  max_order++;

  // Increase local element orders by 1
  for (int i = 0; i < ne; ++i)
    order_array[i] += 1;  

  GetElementInformation(fspacedual, fadata, eldata, ma, order_array, lh);

  adj.vecQ.assign(ndof_q_total * COMP, 0.);
  adj.vecW.assign(ndof_w_total * COMP, 0.);
  adj.vecL.assign(ndof_l_total * COMP, 0.);

  // Load adjoint solution
  cout << "Loading adjoint solution" << endl;
  bool loaded = false;
  stringstream ossadj(" ");
  ossadj << "solution-adjoint-" << ne << "-" << max_order;
  Model::GetFilename(ossadj);

  loaded = LoadSolution(ossadj.str(), adj);

  Solution sol = adj;

  ProlongateOrder(sol_old, sol, 1);
  ReconstructSolution(sol);

  // Allocate some helper arrays
  AllocateTempMemory();
  
  // Determine the nonzero entries in the system matrix
  vector<int> d_nnz(ndof_l_total * COMP);
  
  int index = 0;
  int sum   = 0;
  for (int i = 0; i < nf; ++i) {
    if (!fadata[i]) continue;      
    FacetData<D,COMP> & fd = *fadata[i];
    
    if (fd.ndof_l != 0) {
      ElementData<D,COMP> & el1 = *eldata[fd.elnr1];      
      int ndof_l = el1.ndof_lt;
      
      ElementData<D,COMP> & el2 = *eldata[fd.elnr2];
      ndof_l += el2.ndof_lt - fd.ndof_l;
      
      for (int ll = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < fd.ndof_l; ++pp, ++index)
          d_nnz[index] = COMP * ndof_l;
      
      sum += COMP * fd.ndof_l * COMP * ndof_l;
    }
  }

  petsc_initialize(ndof_l_total * COMP, ndof_l_max * COMP, &d_nnz[0], show_monitor, AdjointData::ksp_restart, pc_factor_levels, false, AdjointData::lin_tol, AdjointData::nit_lin);

  assemblyMode = AM_AssemblyAdjoint;
  
  ComputeWallDistance();

  // Assemble matrix and adjoint rhs
  vector<double> vecF(COMP * ndof_q_max, 0.);
  vector<double> vecG(COMP * ndof_w_max, 0.);

  Residual res;

  // Calls to PetSc
  pcfl = AdjointData::pcfl_min;
  double adj_res = 1., res_lin;

  cout << "\t" << string(8*8, '*') << endl;
  cout << "\t" << setw(16) << "WallTime" << setw(8) << "#Newton" << setw(16) << "CFL" << setw(8) << "#GMRES" << setw(16) << "resL2" << endl;
  cout << "\t" << string(8*8, '*') << endl;

  for (int t = 0; t < max(2, AdjointData::nit) && adj_res > AdjointData::nonlin_tol; ++t) {

    petsc_matrix_reset();
    petsc_rhs_reset();
    
    for (int i = 0; i < ne; ++i) {
      SolveLocalSystem(i, matQ, matW, res);
      AssembleHybridSystemLocal(i, matQ, matW, matL, matM);

      AssembleLocalAdjointRhs(i, vecF, vecG);
      AssembleHybridAdjointRhs(i, matQ, matW, vecF, vecG);
    }

    petsc_begin_mat_alloc(); petsc_end_mat_alloc();
    petsc_begin_vec_alloc(); petsc_end_vec_alloc();

    //petsc_matrix_transpose();
    
    petsc_assemble_adj_rhs(adj.vecL, adj_res);

    its = 0;
    if (adj_res > AdjointData::nonlin_tol)
      petsc_solve_adjoint(adj.vecL, its, res_lin);

    double now = get_time();
    double tcomp = now - starttime;

    cout << "\t" << setw(16) << tcomp << setw(8) << t << setw(16) << pcfl << setw(8) << its << setw(16) << adj_res << endl;    

    pcfl *= AdjointData::pcfl_beta;
  }

  vector<double> LocalLambda(COMP * nf_max * ndof_l_max);

  assemblyMode = AM_BackSolveAdjoint;
  
  // Compute \delta w and \delta q using the local solves
  for (int i = 0; i < ne; ++i) {
    ElementData<D,COMP> & ed = *eldata[i];    
    
    // Obtain the local solves
    int ndof_w  = ed.ndof_w;
    int ndof_q  = ed.ndof_q;
    int ndof_lt = ed.ndof_lt;
    
    SolveLocalAdjointSystem(i, matQ, matW);

    copy(&matW[0], &matW[0] + COMP * ndof_w, &adj.vecW[COMP * ed.offset_w]);
    if (Model::Diffusion || Model::Source)
      copy(&matQ[0], &matQ[0] + COMP * ndof_q, &adj.vecQ[COMP * ed.offset_q]);
    
    int index = 0;
    for (int ff = 0; ff < ed.nf; ++ff) {
      FacetData<D,COMP> & fd = *fadata[ed.faces[ff]];
      int ndof_l = fd.ndof_l;
      int offset = COMP * fd.offset_l;

      // No boundary faces
      if (fd.ndof_l == 0) continue;

      copy(&adj.vecL[offset], &adj.vecL[offset] + COMP * ndof_l, &LocalLambda[index]);

      index += COMP * ndof_l;
    }  

    mygemv('n', COMP * ndof_w, COMP * ndof_lt, 1., &matW[COMP * ndof_w], &LocalLambda[0], 1., &adj.vecW[COMP*ed.offset_w]);

    if (Model::Diffusion || Model::Source)
      mygemv('n', COMP * ndof_q, COMP * ndof_lt, 1., &matQ[COMP * ndof_q], &LocalLambda[0], 1., &adj.vecQ[COMP*ed.offset_q]);
  }

  if (visualize_adjoint)
    TransferCoefficients(adj, gfuncdual);
  Ng_Redraw();

  // Save the solution
  stringstream oss(" ");
  oss << "solution-adjoint-" << ne << "-" << max_order;
  Model::GetFilename(oss);

  string filename(oss.str());
  
  SaveSolution(filename, adj, fspacedual, adj_coordinates, adj_sol, lh);

  // Compute error estimate
  assemblyMode = AM_ErrorEstimation;
  
  sum_error     = 0.;
  sum_abs_error = 0.;

  // adj_rhs_l contains the pure hybrid residual
  adj_rhs_l.assign(COMP * ndof_l_total, 0.);

  // VVector<double> & errvec = dynamic_cast<VVector<double> &> (gf_err -> GetVector());          
  FlatVector<Vec<1>> errvec = gf_err->GetVector().FV<Vec<1>>();
  // Element contribution
  for (int i = 0; i < ne; ++i) {
    ElementData<D,COMP> & ed = *eldata[i];    
    
    // Obtain the local solves
    int ndof_w = ed.ndof_w;
    int ndof_q = ed.ndof_q;

    fill(&matQ[0], &matQ[0] + COMP * ndof_q, 0.0);
    fill(&matW[0], &matW[0] + COMP * ndof_w, 0.0);    
    
#ifdef BR2
    // Order for the lifting operator has to be decreased by one
    AssembleBR2(i, false, 1);
#endif

    AssembleLocalResidual(i, matQ, matW);

    double err = 0.;
    for (int ll = 0; ll < COMP; ++ll)
      for (int pp = 0; pp < ndof_w; ++pp)
        err += matW[pp+ndof_w*ll] * adj.vecW[COMP*ed.offset_w+pp+ndof_w*ll];

    if (Model::Diffusion || Model::Source)
      for (int ll = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof_q; ++pp)
          {
            err += matQ[pp+ndof_q*ll] * adj.vecQ[COMP*ed.offset_q+pp+ndof_q*ll];
          }

    sum_error += err;
    sum_abs_error += abs(err);

    error[i] = abs(err);
    errvec(i) = log10(abs(err));

    // // Saving ML output (Mehul)

    // for (int pp = 0; pp < ed.nip; ++pp)
    //   {
    //     for (int ll=0; ll < COMP; ++ll)
    //     {
    //       //adjw[ll+pp] = ed.w[ll+pp];
    //       adjw.push_back(ed.w[ll+pp]);
    //     }

    //     // adjqp[pp*D] = ed.qp[pp*D];
    //     // adjqp[pp*D+1] = ed.qp[pp*D+1];

    //     adjqp.push_back(ed.qp[pp*D]);
    //     adjqp.push_back(ed.qp[pp*D+1]);

    //   }   
  
  }


  Ng_Redraw();
  
  // Facet contribution
  for (int i = 0; i < nf; ++i) {
    if (!fadata[i]) continue;
    FacetData<D, COMP> & fd = *fadata[i];
    int ndof_l = fd.ndof_l;
    
    if (ndof_l == 0) continue;
    
    double err = 0.;
    for (int ll = 0; ll < COMP; ++ll)
      for (int pp = 0; pp < ndof_l; ++pp)
        err += adj.vecL[COMP*fd.offset_l+pp+ndof_l*ll] * adj_rhs_l[COMP*fd.offset_l+pp+ndof_l*ll];

    sum_error += err;
    sum_abs_error += abs(err);        
  }  

  cout << "The global estimated error is " << sum_error << endl;
   
  order--;
  max_order--;
  // Decrease local element orders again
  for (int i = 0; i < ne; ++i)
    order_array[i] -= 1;    

  assemblyMode = AM_None;

  if (AdjointData::write_error) {
    stringstream oss(" ");
    oss << "error-" << ne << "-" << max_order;
    Model::GetFilename(oss);
    oss << ".txt";

    string filename(oss.str());

    ofstream errfile(filename.c_str());
    errfile.precision(16);
    errfile.setf(ios::scientific, ios::floatfield);
    
    for (int i = 0; i < ne; ++i)
      errfile << error[i] << endl;
    
    errfile.close();
    if(AnisotropyData::sol_write_error)// && AnisotropyData::adjoint_based
    {
      err_monitors.push_back(sum_error);
    }
  }
  
}
