/** @file 
 * @brief This function computes the residual with the split amongst them like the paper proposed
 * by Prof. Dolejsi
 * @param[in] - sol - Contains the data of type \c Solution. This is the original solution and not the
 * reconstructed solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[out] - residual - Contains the residuals for each triangle. The ordering is such that we have the error for each triangle together. Size = ne x 3 double
 * @param[in] - lh - Local heap.
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputePrimalResidual(const Solution & sol, vector<double> &residual, LocalHeap & lh) {

	// The residual is to be stored as follows:
	// For cell i we have residual[3*i+0] = residual over the cell
	// residual[3*i+1] = residual over the face of the cell multiplied by the test function
	// residual[3*i+2] = residual over the face of the cell multiplied by the gradient of the test functon

	// First we reconstruct the solution of order p where p is the original order in which we solved the problem
	GetElementInformation(fspacedual, fadata, eldata, ma, order_array, lh);
	Solution sol_p = sol;
    ReconstructSolution(sol_p);
	AnisotropyData::residual_mode = 0;
	for(int i = 0; i < ne; ++i)
	{	
		// For now we set the dual residual to be 0. This has to be computed
		vector<double> residual_local(3,0.0);
		ComputeLocalResidual(sol_p, i, residual_local);
		for(int ll = 0; ll < 3; ll++)
		{
			residual[i*3+ll] = residual_local[ll];
			// residual[i*3+ll] = 0.0;
		}
	}// end of loop over elements

}

