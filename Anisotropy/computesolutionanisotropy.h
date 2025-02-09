/** @file 
 * @brief This function computes ONLY the anisotropy of the solution, \c sol, 
 * based on the p+n^th order of the solution
 * @param[in] - sol - Contains the data of type Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[out] - aniso_sol_vec - Contains the anisotropy of the solution for all components (A1,A2,A3,theta_z,theta_y,theta_x)_ll. Size = COMP*ne*D*(D+1)/2 doubleå
 * @param[in] - lh - Local heap.
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeSolutionAnisotropy(Solution & sol, vector< SolAniso<D> > & aniso_sol_vec, const int add_order, LocalHeap & lh) {

  cout << string(32, '*') << endl;
  cout << "Compute solution anisotropy... " << endl;
  cout << string(32, '*') << endl;
// int max_order = *max_element(order_array.begin(), order_array.end());
  // if((!AnisotropyData::adjoint_based && Model::NumDerVar>0 && AnisotropyData::opt_strategy == AnisotropyData::analytic) || (AnisotropyData::adjoint_based && Model::NumDerVar>0 && AnisotropyData::opt_strategy == AnisotropyData::mesh_fraction))
  // {
  //   for(int ll = 0; ll < Model::NumDerVar; ++ll)
  //   {
  //     vector<double> rhs(ndof_w_max);
  //     vector<int> piv(ndof_w_max);
  //     vector<double> vecW_val;
  //     vecW_val.assign(ndof_w_total, 0.0);
  //     // #pragma omp parallel for
  //     for (int i = 0; i < ne; ++i) {

  //       HeapReset hr(lh);
        
  //       ElementData<D,COMP> & ed = *eldata[i];
  //       // This is computed for the order = order_array[i]
  //       int ndof_w = ed.ndof_w;
        
  //       int order_der_loc = order_array[i] + add_order;// order_der_loc = p+add_order

  //       int no_aniso = 1;// If we have advection then we have only COMP variables with ansiotropy
  //       double * sensor[no_aniso];
  //       fill(&rhs[0], &rhs[0] + ndof_w, 0.);
  //       double * vecW = &vecW_val[0];
  //       for (int j = 0; j < ed.nip; ++j) {

  //         const double   qw  = ed.qw[j];
  //         const double * qp  = &ed.qp[j * D];
  //         const double * pw  = &ed.w[j * COMP];
  //         const double * phi = &ed.phi[j * ndof_w];

  //         SpatialParams<D> sparam;
  //         for (int dd = 0; dd < D; ++dd)
  //           sparam.pos(dd) = qp[dd];
          
  //         Vec<COMP> state;
  //         for (int ll = 0; ll < COMP; ++ll)
  //           state(ll) = pw[ll];

  //         Vec<Model::NumDerVar> dervar;
  //         Model::EvalDerVar(state, sparam, dervar);

  //         double val = dervar(ll);
          
  //         for (int pp = 0; pp < ndof_w; ++pp)
  //           rhs[pp] += qw * val * phi[pp];
  //       }

  //       mygemm('n', 'n', ndof_w, 1, ndof_w, 1., &ed.inv_mass[0], &rhs[0], 0., vecW);
  //       sensor[0] = &vecW_val[0];

  //       for(int dd2 = 0; dd2 < no_aniso; ++dd2)
  //       {
  //         double *sensor_loc = sensor[dd2];
  //         vector<double> dw;
  //       // Input is i, no_aniso, order_der_loc, sensor, sensor_q
  //       // Output is dw which is of size no_aniso times order_der_loc+1
  //         ComputeHighOrderDerivative(i, order_der_loc, sensor_loc, dw, lh);
  //         // Added to compute and store the anisotropy of the solution
  //         vector<double> aniso_loc(D*(D+1),0.0);
  //         // The order to fill aniso loc is as follows:
  //         // First D terms should be the Ap_{i} in decreasing order
  //         // WARNING: Angles must already be ordered as required by the ellipse
  //         // The next D*(D-1)/2 terms should be the angles filled in the order of rotation z-y-x
  //         ComputeMaxDirectionalDerivatives(order_der_loc, dw, aniso_loc);
  //         aniso_sol_vec[ll*(no_aniso)+dd2].SetComponents(i, aniso_loc);
  //       }        
  //     }// end of loop over elements

  //   }// end of loop over dervar
  // return;
  // };
  // Construct the anisotropy for each component in the vector vecW
  for(int ll = 0; ll < COMP; ++ll)
  {
    // #pragma omp parallel for
    for (int i = 0; i < ne; ++i) {

      HeapReset hr(lh);
      
      ElementData<D,COMP> & ed = *eldata[i];
      // This is computed for the order = order_array[i]
      int ndof_w = ed.ndof_w;
      
      int order_der_loc = order_array[i] + add_order;// order_der_loc = p+add_order

      int no_aniso = 1;// If we have advection then we have only COMP variables with ansiotropy
      if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
      {
        no_aniso = D+1;// If we have diffusion then we have D derivatives and 1 primary
        // varialbe for each COMP whose anisitropy is to be stored
      } 
      double * sensor[no_aniso];
      // adapt with respect to W
      sensor[0] = &sol.vecW[COMP*ed.offset_w+ll*ndof_w];
      if ((Model::Diffusion || Model::Source) && AnisotropyData::adjoint_based)
      {
        for(int dd = 0; dd < D; ++dd)
        {
          sensor[dd+1] = &sol.vecQ[COMP*ed.offset_q+(dd+D*ll)*ndof_w];
        }
      } 

      // Reconstruct p+ith derivatives of w and q

      for(int dd2 = 0; dd2 < no_aniso; ++dd2)
      {
        double *sensor_loc = sensor[dd2];
        vector<double> dw;
      // Input is i, no_aniso, order_der_loc, sensor, sensor_q
      // Output is dw which is of size no_aniso times order_der_loc+1
        ComputeHighOrderDerivative(i, order_der_loc, sensor_loc, dw, lh);
        // Added to compute and store the anisotropy of the solution
        vector<double> aniso_loc(D*(D+1),0.0);
        // The order to fill aniso loc is as follows:
        // First D terms should be the Ap_{i} in decreasing order
        // WARNING: Angles must already be ordered as required by the ellipse
        // The next D*(D-1)/2 terms should be the angles filled in the order of rotation z-y-x
        ComputeMaxDirectionalDerivatives(order_der_loc, dw, aniso_loc);
        aniso_sol_vec[ll*(no_aniso)+dd2].SetComponents(i, aniso_loc);
      }

    }// end of loop over elements
  }// end of loop over components
}
