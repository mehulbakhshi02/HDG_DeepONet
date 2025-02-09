/** @file 
 * @brief This function computes ONLY the size of the metric, \c metric. There are currently
 * two ways to compute the metric size.
 * - Using the old way of scaling the metric based on the adjoint error estimate
 * - Using the analytic optimization where we use the anisotropy information to compute the sizes
 * The two methods are written as two overloaded functions with different set of parameters.
 *
 * Scaling metric based on adjoint error estimate
 * @param[in] - error - Contains the adjoint based error estimate. Size = ne double
 * @param[out] - metric_size - Contains the metric sizes for each cell. Size = ne double
 *
 * Analytic optimization without hp
 * @param[in] - aniso_sol - Contains the anisotropy of the final solution with respect to which we want to compute the size. Size = ne*D*(D+1)/2 double
 * @param[out] - metric_size - Contains the metric sizes for each cell. Size = ne double
 * Analytic optimization with hp
 * @param[in] - aniso_sol - Contains the anisotropy of the final solution with respect to which we want to compute the size. Size = ne*D*(D+1)/2 double
 * @param[in] - aniso_sol - The reference size with respect to which we calculate the size. Size = ne double
 * @param[out] - metric_size - Contains the metric sizes for each cell. Size = ne double

*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::ComputeSolutionSize(const vector<double> & error, vector<double> & metric_size) {

  if (D != 2) {
    cout << "[ComputeSolutionSize] So far implemented for 2d only." << endl;
    exit(0);
  }
  cout << string(32, '*') << endl;
  cout << "Compute size using mesh fraction... " << endl;
  cout << string(32, '*') << endl;

  // Compute the current mesh metric
  MeshMetric<D> metric_implied(ne);
      // Hack for 2D
  ComputeMeshMetric(metric_implied);
    // Hack for 2D ends
  if (AnisotropyData::save_mesh_metric || AnisotropyData::save_adj_metric) {
    string fname("implied_metric.mtr");
    SaveMetric(fname, metric_implied);
  }
  
  vector<double> sorted(error);

  sort(sorted.begin(), sorted.end());
  double max_lerr = log10(sorted[ne-1]);
  double min_lerr = log10(sorted[0]);

  double ref_error = sorted[(1. - AnisotropyData::ref_err) * (ne - 1)];

  // // Write adjoint error into file
  // if (AdjointData::write_error) {
  //   ofstream errfile;
  //   stringstream oss(" ");
  //   oss << "error-" << ne << "-" << order;
  //   Model::GetFilename(oss);  
  //   errfile.open (oss.str().c_str());
  //   for (int i = 0; i < ne; ++i)
  //     errfile << sorted[i] << endl;
  //   errfile.close();
  // }
  
  cout << "Log Min Error is " << min_lerr << endl;
  cout << "Log Max Error is " << max_lerr << endl;
  cout << "Reference error is " << ref_error << endl;

  for (int i = 0; i < ne; ++i){
    ElementData<D,COMP> & ed = *eldata[i];
    // Determine requested area via implied metric and error indicator
    // double h_min, beta, theta;
    // metric_implied.GetPrincipal(i, h_min, beta, theta);
    // double h_max = h_min * beta;
    // double imp_Area = h_min * h_max;

    double imp_Area;
    vector<double> beta(D-1,0.0);
    vector<double> q_met(D*D,0.0);

    metric_implied.GetPrincipal(i, imp_Area, beta, q_met);

    double new_Area = imp_Area;

    double avg_lerr = log10(ref_error);
    if (log10(error[i]) > avg_lerr)
    {
      double r_fact = (log10(error[i])-avg_lerr)/(max_lerr - avg_lerr);
      r_fact = (AnisotropyData::r_max-1.0) * r_fact * r_fact + 1.0;
      new_Area = imp_Area/r_fact;
    }
    else
    {
      double c_fact = (log10(error[i])-avg_lerr)/(min_lerr - avg_lerr);
      c_fact = (AnisotropyData::c_max-1.0) * c_fact * c_fact + 1.0;
      new_Area = imp_Area * c_fact;
    }
    metric_size[i] = new_Area;
  }
  
}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::ComputeSolutionSize(const SolAniso<D> & aniso_sol, vector<double> & metric_size) {

  double q = AnisotropyData::norm;
  if(q!=1 && AnisotropyData::adjoint_based && AnisotropyData::opt_strategy==1)
  {   
    cout<<"The norm is set incorrectly to "<<q<<" must be changed to 1 for adjoint based analytic optimization"<<endl;
    exit(1);
  }
  cout << string(32, '*') << endl;
  cout << "Compute size using analytic optimization... " << endl;
  cout << string(32, '*') << endl;

// The normalization constant in the analytic optimization.
// \int_{\Omega}(A_{1}A_{2}A_{3})^{\frac{q}{q(p+1)+D}} in 3D
// \int_{\Omega}(A_{1}A_{2}A_{3})^{\frac{q}{q(p+1)+D}} = \sum_{K\in\sT}(A_{1}A_{2}A_{3})^{\frac{q}{q(p+1)+D}}*elidata.vol
  double norm_const = 0.0; 
  for(int i = 0; i < ne; i++)
  {
    ElementData<D,COMP> & ed = *eldata[i];
    double power = 0.0;
    double p = (double)order_array[i]; // The polynomial order of the solution. Remember that we have order_array as p
    if(q != -1)
    {
      power = q/( q * (p + 1.0) + D);//q/(q*(p+1)+D)
    }
    else
    {
      power = 1.0/(p+1.0);// lim_{q\to\infty}q/(q*(p+1)+D) = 1/(p+1)
    }
    vector<double> aniso_loc(D*(D+1), 0.0);
    aniso_sol.GetComponents(i, aniso_loc);// The second component is always 0 because we compute the size from the final reconciled anisotropy
    double aniso_product = aniso_sol.GetProd(i);
    // We dont divide by factorial because we have A_1, A_2, A_3 which already have the division of factorial done
    norm_const += pow(fabs(aniso_product), power) * ed.vol;
  }

  // Changing the anisotropic adaptation constant
  if(D == 3)
  {
    AnisotropyData::adapt_const = 8.0/(9.0*sqrt(3.0));
  }
  double Nnodes = AnisotropyData::dof_target * AnisotropyData::adapt_const;// The constant that determines the number of cells in the mesh
  // For 2D the relation between no of cells and volume is 3*sqrt(3)/4 while in 3D it is 8/(9*sqrt(3))
  // Hack for 2D ends
  for(int i = 0; i < ne; i++)
  {
    double power = 0.0;
    double p = order_array[i]; // The polynomial order
    ElementData<D,COMP> & ed = *eldata[i];
    if(q != -1){
      power = q/( q * (p + 1.0) + D);//q/(q*(p+1)+D)
    }
    else
    {
      power = 1.0/(p + 1);//1/(p+1)
    }

    vector<double> aniso_loc(D*(D+1), 0.0);
    aniso_sol.GetComponents(i, aniso_loc);
    double aniso_product = aniso_sol.GetProd(i);
    double new_Area = norm_const/(Nnodes * (pow(fabs(aniso_product), power)));//1/d = new_area we do not have the factor of pi
    metric_size[i] = new_Area;
  }

}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::ComputeSolutionSize(const SolAniso<D> & aniso_sol, const vector<double> & error, vector<double> & metric_size) {
  double q = (double)AnisotropyData::norm;
  for(int i = 0; i < ne; i++)
  {
    double p = (double)order_new_array[i]; // The polynomial order
    double cpq = 0.0;
    if(AnisotropyData::norm != -1)
        cpq = (double)D/(q*(p+1.0)+(double)D);//(D/(q(p+1)+D))
    else
      cpq = 1.0;
    double a_prod = aniso_sol.GetProd(i);
    double a_bar = pow(a_prod, 1.0/(double)D);
    double power = (double)D/(q*(p+1.0));
    // We use the error estimate for the optimum calculated
    // earlier to determine the size
    // The new area should be selected earlier in the
    // if conditions itself
    double new_Area = pow(error[i]/(cpq * pow(a_bar, q)), power);
    metric_size[i] = new_Area;
  }
}

// template <int D, int COMP, class Model>
// void UnifyingFramework<D, COMP, Model>
// ::ComputeSolutionSize(const vector<double> & error, const double tolerance, vector<double> & metric_size) {

//   for(int i = 0; i < ne; i++)
//   {
//     double alpha = 2.0 * (double)order_array[i]; // The alpha 
//     ElementData<D,COMP> & ed = *eldata[i];
//     double l_old = sqrt(ed.vol);
//     double err_equi = tolerance/(AnisotropyData::dof_target*error[i]);
//     double l_new = l_old * pow(err_equi, 1.0/alpha);
//     metric_size[i] = pow(l_new, 2.0);
//   }
// } 