/** @file 
 * @brief This function computes ONLY the anisotropy of the solution, \c sol, 
 * based on the p+n^th order of the solution
 * @param[in] - sol - Contains the data of type Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[out] - aniso_sol_vec - Contains the anisotropy of the solution for all components (A1,A2,A3,theta_z,theta_y,theta_x)_ll. Size = COMP*ne*D*(D+1)/2 doubleå
 * @param[in] - lh - Local heap.
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeGradAnisotropy(Solution & sol, vector< SolAniso<D> > & aniso_sol_vec, const int add_order, LocalHeap & lh) {

  if(D!=2)
  {
    cout<<"Compute grad ansiotropy implemented only for 2D"<<endl;
    exit(1);
  }

  cout << string(32, '*') << endl;
  cout << "Compute gradient anisotropy... " << endl;
  cout << string(32, '*') << endl;

  // int max_order = *max_element(order_array.begin(), order_array.end());

  ReconstructSolution(sol);

  // Construct the anisotropy for each component in the vector vecW
  for(int ll = 0; ll < COMP; ++ll)
  {
    // #pragma omp parallel for
    for (int i = 0; i < ne; ++i) {

      HeapReset hr(lh);
      
      ElementData<D,COMP> & ed = *eldata[i];
      // This is computed for the order = order_array[i]
      int ndof_w = ed.ndof_w;

      // Hack for h only
      int order_der_loc = order_array[i] + add_order;


      int no_aniso = 1;// If we have advection then we have only COMP variables with ansiotropy
      double * sensor[no_aniso];
      // adapt with respect to W
      sensor[0] = &sol.vecW[COMP*ed.offset_w+ll*ndof_w];
      // Reconstruct p+ith derivatives of w and q

      for(int dd2 = 0; dd2 < no_aniso; ++dd2)
      {
        double *sensor_loc = sensor[dd2];
        vector<double> dw;
      // Input is i, no_aniso, order_der_loc, sensor, sensor_q
      // Output is dw which is of size no_aniso times order_der_loc+1
        ComputeHighOrderDerivative(i, order_der_loc, sensor_loc, dw, lh);

        // Added to compute and store the anisotropy of the solution
        // vector<double> aniso_loc(D*(D+1)/2,0.0);
        vector<double> aniso_loc(D*(D+1),0.0);
        // The order to fill aniso loc is as follows:
        // First D terms should be the Ap_{i} in decreasing order
        // WARNING: Angles must already be ordered as required by the ellipse
        // The next D*(D-1)/2 terms should be the angles filled in the order of rotation z-y-x
        ComputeMaxDirectionalDerivativesGrad(order_der_loc, dw, aniso_loc);
        aniso_sol_vec[ll*(no_aniso)+dd2].SetComponents(i, aniso_loc);
      }

    }// end of loop over elements
  }// end of loop over components
}
