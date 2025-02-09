template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::Do(LocalHeap & lh) {

  if (ma->GetNE() == 0) {
    cout << "Empty mesh!!!" << endl;
    return;
  }

// C++ code

  starttime = get_time();
  // HACK
  for (int ll = 0; ll < COMP; ++ll)
    residual_scale[ll] = 1.;

  if (COMP == D+4) {
    residual_scale[D+2] = 1. / 0.01;
    residual_scale[D+3] = 1. / 100.;
  } else if (COMP == D+3) {
    residual_scale[D+2] = 1. / 1000.;
  }
  // END HACK
  vector<int> eleorder(ma->GetNE());
  if (AnisotropyData::adaptation && AnisotropyData::read_bamg_solution)
    min_order = max_order;
  if (AnisotropyData::hp && AnisotropyData::read_order)//AnisotropyData::adaptation && 
  {
    ReadOrder("nodeorder_new.bb", eleorder);
    min_order = max_order;
  }
  
  Solution sol;

  vector<double> u_coordinates;
  vector<double> u_sol;

  for (int p = min_order; p <= max_order; ++p) {
    cout << string(32, '*') << endl;
    cout << "Order " << p << endl;
    cout << string(32, '*') << endl;
    if(!AnisotropyData::read_order)
    {    
      order = p;
      order_array.assign(ma->GetNE(), p);
    }
    else
    {
      order_array.resize(ma->GetNE());
      for(int i = 0; i < ma->GetNE(); ++i)
      {
        order_array[i] = eleorder[i];
      }
    }    
 
    GetElementInformation(fspace, fadata, eldata, ma, order_array, lh);

    ComputeWallDistance();
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

    cout << "NDOF: " << ndof_l_total * COMP << endl;
    cout << "NNZ:  " << sum << endl;
    
    petsc_initialize(ndof_l_total * COMP, ndof_l_max * COMP, &d_nnz[0], show_monitor, ksp_restart, pc_factor_levels, false, rel_tol, max_steps);

    
    // Try to load a previous solution file
    stringstream oss(" ");
    oss << "solution-" << ne << "-" << p;
    Model::GetFilename(oss);

    bool loaded = false;
    if (AnisotropyData::read_bamg_solution){
      
      loaded = LoadSolutionBamg("nodesolution_new.bb", sol, lh);
      }
    else{
      
      loaded = LoadSolution(oss.str(), sol);
    }

    // If there is none, use the initial solution or a lower order solution
    if (!loaded) {
      if (p == min_order)
        Initial(sol);
      else
        ProlongateOrder(sol_old, sol, 1);
    }    

    TransferCoefficients(sol, gfunc);
    Ng_Redraw();
    // Allocate some helper arrays
    AllocateTempMemory();

    if (testing) {
      Test(sol);
      return;
    }
    // Reconstructng the initial condition
    ReconstructSolution(sol);

    Newton(sol);
    // Save the solution
    string filename(oss.str());
    
    SaveSolution(filename, sol, fspace, u_coordinates, u_sol, lh);
    sol_old = sol;
    // Setting the solve ndof to the total ndof.
    // The total ndof's change as we go to a richer space for adjoints and so on
    ndof_w_solve = ndof_w_total;
    ndof_q_solve = ndof_q_total;
    ndof_l_solve = ndof_l_total;
    if(p==max_order)
      AnalyzeSolution(sol, sol, lh);

    WriteLog();

  }
  solve = get_time();

  // Generate the data required for ML
  GenerateMLData(sol_old, lh, sol, u_coordinates, u_sol);
  printf("GenerateMLData: OK\n");

  if (AnisotropyData::adaptation)
  {  
    AnisotropicAdaptation(sol_old, lh);
  }

  cout<< "AnisotropicAdaptation done" << endl;
  
  adapt = get_time();

  WriteErrorTimeLog();
  petsc_finalize();
  
  cout<< "Main_do.h EOF done" << endl;
  fcloseall();
  return;
}
