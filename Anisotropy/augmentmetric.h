  /** @file
 * @brief This function takes the anisotropy of the solution and the metric sizes to finally form
 * the metric
 * @param[in] - aniso_sol - Contains the anisotropy information. Size = nexD*(D+1)/2 double
 * @param[in] - metric_size - Contains the size information based on some optimization. Size = ne double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::AugmentMetric(const SolAniso<D> aniso_sol, const vector<double> & metric_size, LocalHeap & lh) {

  cout << string(32, '*') << endl;
  cout << "Augment solution metric... " << endl;
  cout << string(32, '*') << endl;

  MeshMetric<D> metric(ne);
  ComputeSolutionMetric(aniso_sol, metric_size, metric);
  if(AnisotropyData::hp)
    WriteOrder("nodeorder.bb", order_new_array);

  string var = "mesh";
  SaveTecplotCellWise(order, var, metric, fspace, lh);

  // Scaling metric for appropriate mesh generator and optimization strategy
  double metric_scaling = 1.0;
  metric_scaling = metric.ScalingFactor(AnisotropyData::opt_strategy, AnisotropyData::mesh_generator);
  metric.Scale(metric_scaling);

  // Scaling the metric such that we get the target DOFs
      
  if (AnisotropyData::dof_control && AnisotropyData::opt_strategy == AnisotropyData::mesh_fraction){
    double dof_met=0.;
    for (int i = 0; i < ne; ++i) {
      ElementData<D,COMP> & ed = *eldata[i];
      // Hack for 2D
      double p = (double)order_array[i];// Contains the original polynomial order
      double dof_p = 2. * (p+1.0) * (p+2.0) / sqrt(3.);
      // Hack for 2D ends
      double det = 0.0;
      det = metric.MetricDet(i);
      dof_met += sqrt(abs(det)) * ed.vol * dof_p;
    } 
    double p_avg = 2.0;
    double dof_p_avg = (p_avg+1.0)*(p_avg+2.0)/2.0;
    double dof_t = AnisotropyData::dof_target * dof_p_avg;
    metric.Scale(double(dof_t)/dof_met);
  }

  if (AnisotropyData::save_adj_metric) 
  {
    string fname("adj_metric.mtr");
    SaveMetric(fname, metric);
    
    if(AnisotropyData::mesh_generator == AnisotropyData::bamg && D==2)
    {

    // bamg system calls
      system("cp bamg.mesh bamg_prev.mesh");
      system("cp netgen.vol netgen_prev.vol");
      
      if (AnisotropyData::save_bamg_solution && !AnisotropyData::hp)
        system("bamg -b bamg.mesh -M adj_metric.mtr -rbb nodesolution.bb -v 3 -o bamg_adapted.mesh -wbb nodesolution_new.bb -nbv 200000");
      else if(!AnisotropyData::save_bamg_solution && AnisotropyData::hp)
        system("bamg -b bamg.mesh -M adj_metric.mtr -rbb nodeorder.bb -v 3 -o bamg_adapted.mesh -wbb nodeorder_new.bb -nbv 200000");
      else
        system("bamg -b bamg.mesh -M adj_metric.mtr -v 3 -o bamg_adapted.mesh -nbv 200000");
      string bamgfn("bamg_adapted.mesh"), netgenfn("netgen.vol");
      Bamg2Netgen(bamgfn, netgenfn);
      system("mv bamg_adapted.mesh bamg.mesh");
    }
    else if(AnisotropyData::mesh_generator == AnisotropyData::angener && D==2 && !AnisotropyData::hp)
    {
      // angener system calls
      system("cp angener.mesh angener_prev.mesh");
      system("cp netgen.vol netgen_prev.vol");
      system("Angener90");
      string angenerfn("angener.mesh"), netgenfn("netgen.vol");
      Angener2Netgen(angenerfn, netgenfn);
      if(AnisotropyData::hp)
      {  
        OrderInterpolate("netgen_prev.vol", "nodeorder.bb", "netgen.vol", "nodeorder_new.bb");
      }

    }
    else if(AnisotropyData::mesh_generator == AnisotropyData::mmg2d && D==2)
    {
      system("cp adapted.mesh adapted_prev.mesh");
      system("cp netgen.vol netgen_prev.vol");
      system("mv adj_metric.mtr adj_metric.sol");
      system("mmg2d_O3 adapted.mesh -sol adj_metric.sol");
      system("cp adapted.o.mesh adapted.mesh");
      string mmgfn("adapted.mesh"), netgenfn("netgen.vol");
      Mmg2D2Netgen(mmgfn, netgenfn);
      if(AnisotropyData::hp)
      {  
        OrderInterpolate("netgen_prev.vol", "nodeorder.bb", "netgen.vol", "nodeorder_new.bb");
      }

    }
    else if(AnisotropyData::mesh_generator == AnisotropyData::madlib && D==2)
    {
      system("cp adapted.mesh adapted_prev.mesh");
      system("cp adapted.msh adapted_prev.msh");
      system("cp netgen.vol netgen_prev.vol");
      system("CustomAdaptation");
      string mmgfn("adapted.mesh"), netgenfn("netgen.vol");
      Mmg2D2Netgen(mmgfn, netgenfn);
      string madlibfn("adapted.msh");
      Netgen2Madlib2D(netgenfn, madlibfn);
      if(AnisotropyData::hp)
      {  
        OrderInterpolate("netgen_prev.vol", "nodeorder.bb", "netgen.vol", "nodeorder_new.bb");
      }

    }
    else if(AnisotropyData::mesh_generator == AnisotropyData::madlib && D==3)
    {
      system("cp adapted.mesh adapted_prev.mesh");
      system("cp adapted.msh adapted_prev.msh");
      system("cp netgen.vol netgen_prev.vol");
      system("CustomAdaptation3D");
      string mmgfn("adapted.mesh"), netgenfn("netgen.vol");
      Madlib3D2Netgen(mmgfn, netgenfn);
      string madlibfn("adapted.msh");
      Netgen2Madlib3D(netgenfn, madlibfn);
    }
    else if(AnisotropyData::mesh_generator == AnisotropyData::mmg3d && D==3)
    {
      system("cp adapted.mesh adapted_prev.mesh");
      system("cp netgen.vol netgen_prev.vol");
      system("mv adj_metric.mtr adj_metric.sol");
      system("mmg3d_O3 adapted.mesh -sol adj_metric.sol");
      system("cp adapted.o.mesh adapted.mesh");
      string mmgfn("adapted.mesh"), netgenfn("netgen.vol");
      Mmg3D2Netgen(mmgfn, netgenfn);
      if(AnisotropyData::hp)
      {  
        // OrderInterpolate(in_mesh, in_order, out_mesh, out_order);
        system("transmesh adapted_prev.mesh adapted_prev.meshb");
        system("transmesh adapted.mesh adapted.meshb");
        system("cp nodeorder.bb nodeorder.sol");
        system("transmesh nodeorder.sol nodeorder.solb");
        system("ref_interp_test --field adapted_prev.meshb nodeorder.solb adapted.meshb nodeorder_new.solb");
        system("transmesh nodeorder_new.solb nodeorder_new.sol");
        system("cp nodeorder_new.sol nodeorder_new.bb");
      }

    }
    else if(AnisotropyData::mesh_generator == AnisotropyData::refine && D==3)
    {
      system("cp adapted.meshb adapted_prev.meshb");
      system("cp netgen.vol netgen_prev.vol");
      system("mv adj_metric.mtr adj_metric.sol");
      system("transmesh adj_metric.sol adj_metric.solb");
      // Edit to have correct path for refine
      system("ref_driver -i adapted.meshb -m adj_metric.solb -x adapted.meshb");
      system("transmesh adapted.meshb adapted.mesh");
      string refinefn("adapted.mesh"), netgenfn("netgen.vol");
      Refine2Netgen(refinefn, netgenfn);
      if(AnisotropyData::hp)
      {  
        // OrderInterpolate(in_mesh, in_order, out_mesh, out_order);
        system("transmesh adapted_prev.mesh adapted_prev.meshb");
        system("transmesh adapted.mesh adapted.meshb");
        system("cp nodeorder.bb nodeorder.sol");
        system("transmesh nodeorder.sol nodeorder.solb");
        system("ref_interp_test --field adapted_prev.meshb nodeorder.solb adapted.meshb nodeorder_new.solb");
        system("transmesh nodeorder_new.solb nodeorder_new.sol");
        system("cp nodeorder_new.sol nodeorder_new.bb");
      }

    }
    else
    {
      cout<<"Mesh generator option set incorrectly. Using BAMG..."<<endl;
    // bamg system calls
      system("cp bamg.mesh bamg_prev.mesh");
      system("cp netgen.vol netgen_prev.vol");
      
      if (AnisotropyData::save_bamg_solution && !AnisotropyData::hp)
        system("bamg -b bamg.mesh -M adj_metric.mtr -rbb nodesolution.bb -v 3 -o bamg_adapted.mesh -wbb nodesolution_new.bb");
      else if(!AnisotropyData::save_bamg_solution && AnisotropyData::hp)
        system("bamg -b bamg.mesh -M adj_metric.mtr -rbb nodeorder.bb -v 3 -o bamg_adapted.mesh -wbb nodeorder_new.bb");
      else
        system("bamg -b bamg.mesh -M adj_metric.mtr -v 3 -o bamg.mesh");
      string bamgfn("bamg_adapted.mesh"), netgenfn("netgen.vol");
      Bamg2Netgen(bamgfn, netgenfn);
      system("mv bamg_adapted.mesh bamg.mesh");
    }
  }
  
}
