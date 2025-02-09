/** @file
 * @brief We take in the set of metrics for a given cell for all anisotropies and compute the intersected metric
 * @param[in] - cell_metric - Contains the metric for all the components for this given cell. Size = COMP*(D*(D+1)/2)) double for convection and COMP*D+1*(D*(D+1)/2)) double for diffusion
 * @param[out] - aniso_loc - Anisotropy of the intersected metric. Size = Dx(D+1)/2 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::IntersectMetric(const MeshMetric<D> & metric, const double total_weight, const int p, vector<double> & aniso_loc) {
	int size_metric = metric.GetSize();
	if(size_metric==0)
	{
		cout<<"Number of metrics provided is 0!"<<endl;
		exit(1);
	}
	vector<double> metric_int(D*(D+1)/2,0.0);
	vector<double> metric_2(D*(D+1)/2,0.0);
	// Take second metric to be the same as the 0th element
	metric.GetComponents(0, metric_2);
	for(int i = 0; i < size_metric; ++i)
	{
		vector<double> metric_1(D*(D+1)/2,0.0);
		// Calculate the local metric
		metric.GetComponents(i, metric_1);
		IntersectTwoMetric(metric_1, metric_2, metric_int);
		// IntersectTwoMetric2(metric_1, metric_2, metric_int);
		// Set the second metric to be the intersected metric
		metric_2 = metric_int;
	}

	MeshMetric<D> metric_temp(1);
	metric_temp.SetComponents(0, metric_int);
	double size = 0.0;
	vector<double> beta(D-1,0.0);
	vector<double> q_met(D*D,0.0);
	metric_temp.GetPrincipal(0, size, beta, q_met);
	// Gives the direction of the optimized metric
	// This has to be opposite to the direction of the optimized solution which is to be returned out of here
	// A_1 = total weight because this is the major component
	double tol = 1e-200;
	aniso_loc[0] = total_weight;
	aniso_loc[0] = max(tol, aniso_loc[0]);
	if(D==2)
	{	

		// Computing the remaining A_{i}s

		for(int dd = 0; dd < D-1; ++dd)
		{
			double rho = pow(beta[dd], (double)p+1.0);
			aniso_loc[1+dd] = aniso_loc[0]/rho;
			aniso_loc[1+dd] = max(tol, aniso_loc[1+dd]);
		}
		aniso_loc[0] = max(tol, aniso_loc[0]);
	}
	if(D==3)
	{	// Computing the remaining A_{i}s
		// beta1 = h1/h2 = (A2/A3)^{1/(p+1)}
		// beta2 = h1/h3 = (A1/A3)^{1/(p+1)}
		// This gives
		// A3 = A1/(beta2^{p+1})
		// A2 = A3 * beta1^{p+1}
		double tol = 1e-200;
		double rho = pow(beta[1], (double)p+1.0);
		aniso_loc[2] = aniso_loc[0]/rho;
		aniso_loc[1] = aniso_loc[2] * pow(beta[0], (double)p+1.0);

		aniso_loc[0] = max(tol, aniso_loc[0]);
		aniso_loc[1] = max(tol, aniso_loc[1]);
		aniso_loc[2] = max(tol, aniso_loc[2]);
		// double A3 = aniso_loc[2];
		// if(AnisotropyData::min_regularize>fabs(A3))
		// {	
		// 	AnisotropyData::min_regularize = fabs(A3);
		// }
	}	
	// Saving the theta
    for(int ll = 0; ll < D; ++ll)
    {
      for(int kk = 0; kk < D; ++kk)
        aniso_loc[D+D*ll+kk] = q_met[D*(D-1-ll)+kk];
    }
}
