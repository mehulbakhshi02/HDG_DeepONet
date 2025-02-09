/** @file 
 * @brief This function computes the maximum direction and the angle of this maximum for the gradient. Basically the error bound terms used by Prof. Dolejsi
 * in his 2014 paper
 * @param[in] - dw - Contains the highest order derivatives. Size = If the solution is computed in polynomial p then projected to polynomial
 * k then this contains k+1 terms double
 * @param[in] - order_der_loc - Order upto which we need to compute the directional derivatives(this should be the same as k)
 * @param[out] - aniso_loc - Contains anisotropy of the elements. Size = D*(D+1)/2 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeMaxDirectionalDerivativesGrad(int order_der_loc, vector<double> & dw, vector<double> &aniso_loc) {

	if(D!=2)
	{
		cout<<"Compute max directional derivative for gradient implemented only for 2D"<<endl;
		exit(1);
	}
	// order_der_loc = order_array which is set in anisotropicadap.h now it is p+1 for the h only case
	// hack for h only

	int nterms = order_der_loc+1;
	vector<double> alpha(nterms,0.0);

	vector<double> dw_loc(order_der_loc+1,0.0);

	if(AnisotropyData::projection_basis == AnisotropyData::monomial_basis)
	{
		for(int i = 0; i < order_der_loc + 1; ++i)
		{
			dw_loc[i] = dw[order_der_loc - i];
		}
	}
	else
	{
		for(int i = 0; i < order_der_loc + 1; ++i)
		{
			dw_loc[i] = dw[i];
		}
	}
	// We not have dw_loc to contain highest x derivative in the first element and then decreasing in powers of x

	// Order_der_loc = p+1
	// nterms = p+2
	// Loop runs from 0...p+1 inclusive
	// 			for (int ni=0; ni < nterms; ni++)
	for(int ni=0; ni<nterms;++ni)
	{
		double factor = 1.0/(Factorial(order_der_loc - ni)*Factorial(ni));
		alpha[ni] = factor * dw_loc[nterms - 1 - ni];
	}
	// The beta required by the rations in prof dolejsi error estimates
	vector<double> beta1(order_der_loc,0.0);
	vector<double> beta2(order_der_loc,0.0);
	// Loop runs from 0...p inclusive
	SpatialParams<D> sparam;
    Vec<COMP> w_h(0.);
	Vec<D> der_diff;
	Model::DerDiffFlux(w_h, sparam, der_diff);
	vector<vector<double> > A(D, vector<double>(D, 0.0));
	A[0][0] = der_diff(0);
	A[1][1] = der_diff(0);
	// Loop runs from 0 to p inclusive
	for(int l=0; l<order_der_loc;++l)
	{
		beta1[l] = A[0][0] * (l+1.0) * alpha[l+1] + A[0][1] * (order_der_loc-l) * alpha[l];
		beta2[l] = A[1][0] * (l+1.0) * alpha[l+1] + A[1][1] * (order_der_loc-l) * alpha[l];
		// cout<<"beta1 "<<beta1[l]<<" beta2 "<<beta2[l]<<endl;
	}
	int p = order_der_loc - 1;
	// Size is 2p+1
	vector<double> delta(2*p+1,0.0);
	// Loop runs from 0...p inclusive
	for(int i = 0; i<=p; ++i)
	{
		for(int j=0; j<=i; ++j)
		{
			delta[i] += beta1[j]*beta1[i-j] + beta2[j]*beta2[i-j];
			delta[2*p-i] += beta1[p-j]*beta1[p-i+j] + beta2[p-j]*beta2[p-i+j];
		}
	}
	// // They contain the magnutude of the maximum derivatives
	double A1 = 0., A2 = 0.;

	double theta_max = 0., theta_min = 0.;

	// // Search max and min derivative
	// Set to 2(p+1)-1=2p+1
	nterms = 2*p+1;
	double theta = 0.0;
	while (theta <= M_PI)
	{
		double der = 0;
		// Loop runs from 0 to 2p inclusive
		for (int ni=0; ni < nterms; ni++)
			der +=  pow(cos(theta),ni) * pow(sin(theta),nterms-1-ni) * delta[ni];
		if (abs(der) > A1) 
		{
			A1 = abs(der);
			theta_max = theta;
		}
		theta += M_PI/180.;
	}// end of while loop

	theta_min = theta_max - M_PI / 2.;      

	A2 = 0.;
	for (int ni=0; ni < nterms; ni++)
	{		
		A2 += pow(cos(theta_min),ni) * pow(sin(theta_min),nterms-1-ni) * delta[ni];
	}

	double tol = 1e-14;
	A1 = max(fabs(A1), tol);
	A2 = max(fabs(A2), tol);

	if(fabs(A1)<fabs(A2))
	{
		cout<<"Error A1<A2"<<endl;
		cout<<A1<<" "<<A2<<endl;
		exit(1);
	}
	// // When setting the aniso_loc we need to do it in decreasing order
	aniso_loc[0] = fabs(A1);// Ap.
	aniso_loc[1] = fabs(A2);// Ap^{\perp}
	// We are going to store the eigen vectors of the rotation matrix
	// Q = [cos(theta_max) -sin(theta_max); sin(theta_max) cos(theta_max)]
	// The eigen vectors are e1 = [cos(theta_max) sin(theta_max)] and e2 = [-sin(theta_max) cos(theta_max)]
	// We store them in the order e1 followed by e2. So the order for storing is
	// cos(theta_max) sin(theta_max) -sin(theta_max) cos(theta_max)
	// This is the maximum direction of the solution NOT OF THE OPTIMUM ELLIPSE
	// Remember that the ordering of the eigen has to be switched when
	// calculating the rotation matrix for the ellipse
	aniso_loc[2] = cos(theta_max);// Theta_max is maximum direction of A1
	aniso_loc[3] = sin(theta_max);// Theta_max is maximum direction of A1
	aniso_loc[4] = -sin(theta_max);// Theta_max is maximum direction of A1
	aniso_loc[5] = cos(theta_max);// Theta_max is maximum direction of A1

}
