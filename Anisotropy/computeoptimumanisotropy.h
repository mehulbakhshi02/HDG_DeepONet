#include "../quadrature_calculation.h"
/** @file
 * @brief This function takes in the solution, \c sol, and adapts the mesh either using
 * the anisotropic adaptation where the anisotropy aligns with the solution \f$u\f$ for scalar
 * equations and with the \f$Mach\f$ number for systems.  
 * The size of the cells can determined by a few methods:
 *   - adjoint based error estimate with \c c_max and \c r_max
 *   - continuous mesh optimization to minimise \f$L^q\f$ norm of \f$u\f$ 
 * @param[in] - sol - Contains the data of type Solution. Size = COMPxDxndof + COMPxndof + COMPxndof_skeleton double
 * @param[in] - lh - Local heap that is used to generate the finite element space for patchwise
 * reconstruction
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeOptimumAnisotropy(const SolAniso<D> &aniso_sol_u, const SolAniso<D> &aniso_grad, const SolAniso<D> &aniso_dual, const SolAniso<D> &aniso_dual_grad, const vector<double> &res_primal, const vector<double> &res_dual, SolAniso<D> &aniso_sol, vector<double> &res_err){

		// Compute the current mesh metric
	MeshMetric<D> metric_implied(ne);
	// Hack for 2D
	ComputeMeshMetric(metric_implied);
	// Hack for 2D ends
	for(int i = 0; i < ne; ++i)
	{
		double size = 0.0;
		vector<double> beta(D-1,0.0);
		vector<double> q_met(D*D,0.0);
		metric_implied.GetPrincipal(i, size, beta, q_met);
	    double lambda = sqrt(size);
		// Should be checked if this gives consistent results for all quadrants as we have aniso_sol_u to be between
		// 0 and PI
		double sigma_opt = sqrt(beta[0]);
	    double theta_opt = acos(q_met[D*(D-1)]);

	    double pk = (double)order_array[i];
		// Compute the factors for all the anisotropies
		vector<double> ap(6, 0.0);
		vector<double> q1(6, 0.0);
		vector<double> q2(6, 0.0);
		vector<double> rho(6, 0.0);
		vector<double> phi(6, 0.0);
		vector<double> residuals(6, 0.0);

		// Computing the value for sol
		vector<double> aniso_loc(D*(D+1), 0.0);
	    aniso_sol_u.GetComponents(i, aniso_loc);
	    ap[0] = aniso_loc[0];
	    q1[0] = pk + 1.0;
	    q2[0] = pk + 1.0;
	    rho[0] = aniso_loc[0]/aniso_loc[1];
		phi[0] = acos(aniso_loc[2]);
		residuals[0] = res_dual[i*3+0];

		// Computing the values for the sol on the cell boundaries
	    ap[1] = aniso_loc[0];
	    q1[1] = pk + 1.0;
	    q2[1] = pk + 1.0;
	    rho[1] = aniso_loc[0]/aniso_loc[1];
		phi[1] = acos(aniso_loc[2]);
		residuals[1] = res_dual[i*3+1];

		// Computing the values for the sol grad on the cell boundaries
	    aniso_grad.GetComponents(i, aniso_loc);
	    ap[2] = aniso_loc[0];
	    q1[2] = pk;
	    q2[2] = 2.0 * pk;
	    rho[2] = aniso_loc[0]/aniso_loc[1];
		phi[2] = acos(aniso_loc[2]);
		residuals[2] = res_dual[i*3+2];

		// Computing the values for the adj on the cell domains
	    aniso_dual.GetComponents(i, aniso_loc);
	    ap[3] = aniso_loc[0];
	    q1[3] = pk + 1.0;
	    q2[3] = pk + 1.0;
	    rho[3] = aniso_loc[0]/aniso_loc[1];
		phi[3] = acos(aniso_loc[2]);
		residuals[3] = res_primal[i*3+0];

		// Computing the values for the dual on the cell boundaries
	    ap[4] = aniso_loc[0];
	    q1[4] = pk + 1.0;
	    q2[4] = pk + 1.0;
	    rho[4] = aniso_loc[0]/aniso_loc[1];
		phi[4] = acos(aniso_loc[2]);
		residuals[4] = res_primal[i*3+1];
			
		// Computing the values for the dual grad on the cell boundaries
	    aniso_dual_grad.GetComponents(i, aniso_loc);
	    ap[5] = aniso_loc[0];
	    q1[5] = pk;
	    q2[5] = 2.0 * pk;
	    rho[5] = aniso_loc[0]/aniso_loc[1];
		phi[5] = acos(aniso_loc[2]);
		residuals[5] = res_primal[i*3+2];

		double size_comp = 0.0;
		double err_comp = 0.0;
		OptimizeAnisotropy(i, lambda, q1, q2, ap, rho, phi, residuals, size_comp, sigma_opt, theta_opt, err_comp);
		double beta_comp = pow(sigma_opt, 2.0);
		double A1 = fabs(size_comp) * pow(beta_comp, (pk+1.0)/2.0);
		double A2 = fabs(A1)/pow(beta_comp, pk+1.0);

		double theta_p = theta_opt + M_PI/2.0;
        vector<double> aniso_set(D*(D+1),0.0);
		aniso_set[0] = fabs(A1);// Ap.
		aniso_set[1] = fabs(A2);// Ap^{\perp}
		aniso_set[2] = cos(theta_p);
		aniso_set[3] = sin(theta_p);
		aniso_set[4] = -sin(theta_p);
		aniso_set[5] = cos(theta_p);
		aniso_sol.SetComponents(i, aniso_set);
		res_err[i] = err_comp;
	}
	if(AnisotropyData::regularize_anisotropy)
		RegularizeAnisotropy(aniso_sol);
}

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeOptimumAnisotropy(const SolAniso<D> aniso_sol, const vector<double> &total_weight, const int p, const double lambda, double sigma, double theta, vector<double> &aniso_loc){

	int size_aniso = aniso_sol.GetSize();
	int size_size = total_weight.size();

	if(size_aniso!=size_size)
	{
		cout<<"Size of anisotropy and size vectors are not equal!"<<endl;
		exit(1);
	}

	vector<double> ap(size_aniso, 0.0);
	vector<double> q1(size_aniso, 0.0);
	vector<double> q2(size_aniso, 0.0);
	vector<double> rho(size_aniso, 0.0);
	vector<double> phi(size_aniso, 0.0);
    double pk = (double)p;
	for(int i = 0; i < size_aniso; ++i)
	{
		// Computing the value for sol
		vector<double> aniso_loc(D*(D+1), 0.0);
	    aniso_sol.GetComponents(i, aniso_loc);
	    ap[i] = aniso_loc[0];
	    q1[i] = (pk + 1.0)/2.0;
	    q2[i] = pk + 1.0;
	    rho[i] = aniso_loc[0]/aniso_loc[1];
		phi[i] = acos(aniso_loc[2]);
	}
	double size_comp = 0.0;
	double err_comp = 0.0;
	OptimizeAnisotropy(pk, lambda, q1, q2, ap, rho, phi, total_weight, size_comp, sigma, theta, err_comp);
	double beta_comp = pow(sigma, 2.0);
	double A1 = fabs(size_comp) * pow(beta_comp, (pk+1.0)/2.0);
	double A2 = fabs(A1)/pow(beta_comp, pk+1.0);
	double theta_p = theta + M_PI/2.0;
	aniso_loc[0] = fabs(A1);// Ap.
	aniso_loc[1] = fabs(A2);// Ap^{\perp}
	aniso_loc[2] = cos(theta_p);
	aniso_loc[3] = sin(theta_p);
	aniso_loc[4] = -sin(theta_p);
	aniso_loc[5] = cos(theta_p);
}