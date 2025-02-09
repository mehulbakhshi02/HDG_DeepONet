  /** @file
 * @brief We use the anisotropy and size information to compute the metric. This is a generic function where
 * we just need these two data. We must be cautious about the polynomial order because here we are assuming
 * that the global array order_array[i] contains p+1 where p is the order of the primal solution in the cell i
 * In case we want to compute for all number of cells \c ne we dont pass p
 * @param[in] - aniso_sol - Contains the anisotropy information. Size = nexD*(D+1)/2 double
 * @param[in] - metric_size - Contains the size information based on some optimization. Size = ne double
 * @param[in] - p - Contains the order of the element for which are are doing the adaptation in case we want to do this for one cell. Size = 1 int
 * @param[out] - metric - Contains the metric computed for the given anisotropy and size. Size = nexDx(D+1)/2 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeSolutionMetric(const SolAniso<D> aniso_sol, const vector<double> & metric_size, MeshMetric<D> & metric) {

  int size_aniso = aniso_sol.GetSize();
  int size_size = metric_size.size();
  int size_metric = metric.GetSize();

  if(size_aniso!=size_size || size_aniso!=size_metric)
  {
    cout<<"Size of anisotropy and size vectors are not equal!"<<endl;
    exit(1);
  }


  // #pragma omp parallel for
  for (int i = 0; i < size_aniso; ++i)
  {
    // We are assuming that the order is fixed
    double p = (double)order_array[i];
    if(AnisotropyData::hp)
      p = (double)order_new_array[i];
    vector<double> aniso_loc(D*(D+1), 0.0);
    aniso_sol.GetComponents(i, aniso_loc);

    vector<double> beta(D-1,0.0);
    vector<double> q_met(D*D,0.0);
    // Beta should be stored in increasing order
    for(int ll = 0; ll < D-1; ++ll)
    {
      // We relate the solution anisotropy and the metric anisotropy parameters
      // Corrected beta calculation including the 3D version
      beta[ll] = pow(fabs(aniso_loc[D-(ll+2)]/aniso_loc[D-1]), 1.0/(p+1.0));
      // We limit the maximum allowed anisotropy 
      beta[ll] = min(beta[ll], AnisotropyData::beta_max);
    }
    // We change the ordering of the eigen vectors 
    // in the solution we stored it as e1, e2, e3 for corresponding A1, A2 and A3
    // Now we store the eigen vectors as e3, e2, e1 in Q_met
    for(int ll = 0; ll < D; ++ll)
    {
      for(int kk = 0; kk < D; ++kk)
        q_met[D*(D-1-ll)+kk] = aniso_loc[D+D*ll+kk];
    }
    metric.SetPrincipal(i, metric_size[i], beta, q_met);
  }

}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeSolutionMetric(const SolAniso<D> aniso_sol, const vector<double> & metric_size, const int p, MeshMetric<D> & metric) {

  int size_aniso = aniso_sol.GetSize();
  int size_size = metric_size.size();
  int size_metric = metric.GetSize();

  if(size_aniso!=size_size || size_aniso!=size_metric)
  {
    cout<<"Size of anisotropy and size vectors are not equal!"<<endl;
    exit(1);
  }

  // #pragma omp parallel for
  for (int i = 0; i < size_aniso; ++i)
  {
    vector<double> aniso_loc(D*(D+1), 0.0);
    aniso_sol.GetComponents(i, aniso_loc);
    vector<double> beta(D-1,0.0);
    vector<double> q_met(D*D,0.0);

    // Beta should be stored in increasing order    
    for(int ll = 0; ll < D-1; ++ll)
    {
      // We relate the solution anisotropy and the metric anisotropy parameters
      // Corrected beta calculation including the 3D version
      beta[ll] = pow(fabs(aniso_loc[D-(ll+2)]/aniso_loc[D-1]), 1.0/(p+1.0));
      // We limit the maximum allowed anisotropy 
      beta[ll] = min(beta[ll], AnisotropyData::beta_max);
    }
    // We change the ordering of the eigen vectors 
    // in the solution we stored it as e1, e2, e3 for corresponding A1, A2 and A3
    // Now we store the eigen vectors as e3, e2, e1 in Q_met
    for(int ll = 0; ll < D; ++ll)
    {
      for(int kk = 0; kk < D; ++kk)
        q_met[D*(D-1-ll)+kk] = aniso_loc[D+D*ll+kk];
    }
    metric.SetPrincipal(i, metric_size[i], beta, q_met);
    vector<double> metric_loc(D*(D+1)/2, 0.0);
    metric.GetComponents(i, metric_loc);
  }

}

