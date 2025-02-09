template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::SaveSolution(const string & filename, const Solution & sol, const Space & fspace, vector<double> &coordinates, vector<double> &solution, LocalHeap & lh) const {	                 

  string fname_out(filename);
  fname_out.append(".out");
  
  ofstream file(fname_out.c_str(), ios::binary);
  
  file.write((const char *)&sol.vecW[0], sizeof(double) * ndof_w_total * COMP);
  if (Model::Diffusion || Model::Source)
    file.write((const char *)&sol.vecQ[0], sizeof(double) * ndof_q_total * COMP);
  file.write((const char *)&sol.vecL[0], sizeof(double) * ndof_l_total * COMP);

  file.close();

  cout << "Solution saved to " << fname_out << endl;

  switch (outputType) {
  // case output::paraview_volume:
  //          // SaveVtu(filename, order, sol, fspace, lh);
  //   break;
  // case output::paraview_surface:
  //          // SaveVtuSurface(oss.str(), order, sol_old, fspace, lh);
  //   break;
  case output::tecplot_volume:
  {      

    SaveTecplot(filename, max_order, sol, fspace, coordinates, solution, lh);
    MeshMetric<D> metric(ne);

    string var = "mesh";
    SaveTecplotCellWise(order, var, metric, fspace, lh);

    if(D==3)
    {
      SaveTecplotSurface(filename, order, sol, fspace, lh);
    }
    break;
  }
  // case output::tecplot_surface:
  //      // SaveTecplotSurface(filename, order, sol, fspace, lh);
  //   break;
  case output::none:
    break;
  }
}
