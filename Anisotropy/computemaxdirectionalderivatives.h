/** @file 
 * @brief This function computes the maximum direction and the angle of this maximum. Basically the error bound terms used by Prof. Dolejsi
 * in his 2014 paper
 * @param[in] - dw - Contains the highest order derivatives. Size = If the solution is computed in polynomial p then projected to polynomial
 * k then this contains k+1 terms double
 * @param[in] - order_der_loc - Order upto which we need to compute the directional derivatives(this should be the same as k)
 * @param[out] - aniso_loc - Contains anisotropy of the elements. Size = D*(D+1)/2 double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeMaxDirectionalDerivatives(int order_der_loc, vector<double> & dw, vector<double> &aniso_loc) {
	if(D==2)
	{

		// Hack for 2D

		// order_der_loc will contain the order upto which the original solution has been projected and the order upto
		// which we have calculated the derivates. Example original solution is computed for p=2 then it has been
		// projected using patchwise projection to p=4 then we want to compute the fourth derivatives. The fourth derivatives
		// will contain 5 terms in 2D. 
		// nterms contains this number. It is the number of terms in the sum used in the error model
		int nterms = order_der_loc+1;

		// They contain the magnutude of the maximum derivatives
		double A1 = 0., A2 = 0.;

		double theta_max = 0., theta_min = 0.;
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
		// The ordering of dw_loc is such that we have first the highest x derivative and then the last element is the
		// highest y derivative
		// Search max and min derivative
		// We start with theta of pi/2 because the the theta_min which is the orientation of the ellipse will be in [0,pi]

		double theta = 0.0;
		while (theta <= M_PI)
		{
			double der = 0;
			for (int ni=0; ni < nterms; ni++)
			{				
				double factor = 1.0/(Factorial(order_der_loc - ni)*Factorial(ni));
				der += factor * pow(cos(theta),nterms-1-ni) * pow(sin(theta),ni) * dw_loc[ni];
			}
			if (fabs(der) > A1) 
			{
				A1 = fabs(der);
				theta_max = theta;
			}

			theta += M_PI/180.;
		}// end of while loop

		// theta_max in [PI/2, 3PI/2] => theta_min in [0, PI]

		theta_min = theta_max - M_PI / 2.;      

		A2 = 0.;
		for (int ni=0; ni < nterms; ni++)
		{		
			double factor = 1.0/(Factorial(order_der_loc - ni)*Factorial(ni));
			A2 += factor * pow(cos(theta_min), nterms-1-ni) * pow(sin(theta_min),ni) * dw_loc[ni];
		}
		// // Hack to check
		double tol = 1e-200;
		A1 = max(fabs(A1), tol);
		A2 = max(fabs(A2), tol);
		// // hack to check
		if(fabs(A1)<fabs(A2))
		{
			cout<<"Error A1<A2"<<endl;
			cout<<fabs(A1)<<" "<<fabs(A2)<<endl;
			exit(1);
		}
		if(isnan(A1) || isnan(A2))
		{
			cout<<"Anisotropies are nan"<<endl;
			exit(1);
		}
 		// When setting the aniso_loc we need to do it in decreasing order
		aniso_loc[0] = fabs(A1);// Ap. Remember that order_der_loc = p+n where p is the polynomail order of the original solution
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
	if(D==3)
	{
		double xbar,ybar,zbar;

		vector<double> e1,e2,e3; //stores the principlal axis for each cell.

		e1.resize(3,0.0);
		e2.resize(3,0.0);
		e3.resize(3,0.0);

		double A1 = 0.0, A2 = 0.0, A3 = 0.0; //stores the principal values along the principal Directions.

		int nterms = order_der_loc + 1;

		double theta_1 = 0.0, theta_2 = 0.0;
		while(theta_1 <= M_PI/2.0)
		{
			while(theta_2 < 2*M_PI)
			{
				xbar=sin(theta_1)*cos(theta_2);
				ybar=sin(theta_1)*sin(theta_2);
				zbar=cos(theta_1);
				int index = 0;
				int l = 0;
				double der = 0.0;
				for(int m = 0; m < nterms; ++m)
				{
					for(int n = 0; n <(nterms-m); ++n,++index)
					{
						l = order_der_loc-m-n;
						der += (1.0/(Factorial(m)*Factorial(n)*Factorial(l)))*dw[index]*pow(xbar,m)*pow(ybar,n)*pow(zbar,l);
					}
				}
				if(A1 <= fabs(der))
				{
					A1 = fabs(der);
					e1[0] = xbar;
					e1[1] = ybar;
					e1[2] = zbar;
				}
				theta_2 += M_PI/180.0;
			}// end of loop over j
			theta_2 = 0.0; //reset theta_2
			theta_1 += M_PI/180.0;
		}// end of loop over i

		// Alternative way of calculating the A1 using the multi level splitting of the sphere
		// double level = 10;
		// double theta_0 = 0.0,phi_0 = 0.0;
		// double mx = 20,nx = 10;
		// double del_theta = 0.0,del_phi = 0.0;

		// for(double lv = 0;lv < level; lv++)
		// {
		// 	int i_max =0.0, j_max = 0.0; 
		// 	del_theta = 2.0*M_PI/(pow(mx,lv+1.0));
		// 	del_phi = M_PI/(2*pow(nx,lv+1.0));

		// 	for(int j = 0; j < nx; j++)
		// 	{
		// 		theta_1 = phi_0 + del_phi*(j+0.5);                          
		// 		for(int i = 0; i < mx; i++)
		// 		{
		// 			theta_2 = theta_0 + del_theta*(i+0.5); 
		// 			xbar=sin(theta_1)*cos(theta_2);
		// 			ybar=sin(theta_1)*sin(theta_2);
		// 			zbar=cos(theta_1);
		// 			int index = 0;
		// 			int l = 0;
		// 			double der = 0.0;
		// 			for(int m = 0; m < nterms; ++m)
		// 			{
		// 				for(int n = 0; n <(nterms-m); ++n,++index)
		// 				{
		// 					l = order_der_loc-m-n;
		// 					der += (1.0/(Factorial(m)*Factorial(n)*Factorial(l)))*dw[index]*pow(xbar,m)*pow(ybar,n)*pow(zbar,l);
		// 				}
		// 			}   
		// 			if(A1 <= fabs(der))
		// 			{
		// 				A1 = fabs(der);
		// 				e1[0] = xbar;
		// 				e1[1] = ybar;
		// 				e1[2] = zbar;
		// 				i_max = i;
		// 				j_max = j;
		// 			}
		// 		}

		// 	}
		// 	theta_0 = theta_0 + del_theta*((double)i_max+0.0);
		// 	phi_0 = phi_0 + del_phi*((double)j_max+0.0);

		// }
		//............................calculation of Rotation Matrix..................................//
		double mag;
		vector<double> vx,ycap,xcap,xax,xp,xg;
		//resizing the vectors.
		vx.resize(3,0.0);
		ycap.resize(3,0.0);
		xcap.resize(3,0.0);
		xax.resize(3,0.0);
		xp.resize(3,0.0);
		xg.resize(3,0.0);

		vector<vector<double> > R;
		R.resize(3,vector<double> (3,0.0));

		vx[0] = e1[0]; vx[1] = e1[1]; vx[2] = e1[2];
		xax[0] = 1.0;xax[1]=0.0;xax[2]=0.0;

		// new y axis = vx X xax
		ycap[0] = vx[1]*xax[2]-vx[2]*xax[1];
		ycap[1] = vx[2]*xax[0]-vx[0]*xax[2];
		ycap[2] = vx[0]*xax[1]-vx[1]*xax[0];

		mag = pow(ycap[0],2.0)+pow(ycap[1],2.0)+pow(ycap[2],2.0);
		mag = sqrt(mag);

		ycap[0]=ycap[0]/mag;ycap[1]=ycap[1]/mag;ycap[2]=ycap[2]/mag;

		//xcap = ycap X vx
		xcap[0] = ycap[1]*vx[2]-ycap[2]*vx[1];
		xcap[1] = ycap[2]*vx[0]-ycap[0]*vx[2];
		xcap[2] = ycap[0]*vx[1]-ycap[1]*vx[0];

		mag = pow(xcap[0],2.0)+pow(xcap[1],2.0)+pow(xcap[2],2.0);
		mag = sqrt(mag);

		xcap[0]=xcap[0]/mag;xcap[1]=xcap[1]/mag;xcap[2]=xcap[2]/mag;

		//formation of rotation matrix

		R[0][0]=xcap[0];R[0][1]=ycap[0];R[0][2]=vx[0];
		R[1][0]=xcap[1];R[1][1]=ycap[1];R[1][2]=vx[1];
		R[2][0]=xcap[2];R[2][1]=ycap[2];R[2][2]=vx[2];
 //---------------------------------------------------------------------------------------------
		double theta_3 = 0.0;
		while(theta_3 < 2.0*M_PI) //initially it was theta_3 < 2.0*M_PI
		{
			xp[0]=cos(theta_3);
			xp[1]=sin(theta_3);
			xp[2]=0.0;

			xg[0] = R[0][0]*xp[0]+R[0][1]*xp[1]+R[0][2]*xp[2];
			xg[1] = R[1][0]*xp[0]+R[1][1]*xp[1]+R[1][2]*xp[2];
			xg[2] = R[2][0]*xp[0]+R[2][1]*xp[1]+R[2][2]*xp[2];

			mag=sqrt(xg[0]*xg[0]+xg[1]*xg[1]+xg[2]*xg[2]);
			xg[0]=xg[0]/mag; //normalization of xg
			xg[1]=xg[1]/mag;
			xg[2]=xg[2]/mag;

			int index = 0;
			int l = 0;
			double der = 0.0;
			for(int m = 0; m < nterms; ++m)
			{
				for(int n = 0; n <(nterms-m); ++n,++index)
				{
					l = order_der_loc-m-n;
					der += (1.0/(Factorial(m)*Factorial(n)*Factorial(l)))*dw[index]*pow(xg[0],m)*pow(xg[1],n)*pow(xg[2],l);
				}
			}

			if(A2<=fabs(der))
			{
				A2=fabs(der);
				e2[0]=xg[0];
				e2[1]=xg[1];
				e2[2]=xg[2];
			}

			theta_3 += M_PI/180.0;
		}// end of loop over angle
		//e3 represents the third direction.
		// This is computed using the cross product of e1 and e2
		e3[0]=e1[1]*e2[2]-e1[2]*e2[1];
		e3[1]=e1[2]*e2[0]-e1[0]*e2[2];
		e3[2]=e1[0]*e2[1]-e1[1]*e2[0];

		mag=pow(e3[0],2.0)+pow(e3[1],2.0)+pow(e3[2],2.0);
		mag=sqrt(mag);

		e3[0]=e3[0]/mag;
		e3[1]=e3[1]/mag;
		e3[2]=e3[2]/mag;
		int index = 0;
		int l = 0;

		for(int m = 0; m<nterms ; ++m)
		{
			for(int n = 0;n < nterms-m; ++n, ++index)
			{
				l = order_der_loc-m-n;
				A3 += (1.0/(Factorial(m)*Factorial(n)*Factorial(l)))*dw[index]*pow(e3[0],m)*pow(e3[1],n)*pow(e3[2],l);
			}
		}

		A3=fabs(A3);

		// Making sure that we do not get zero as the directional derivative
		double tol = 1e-14;
		A1 = max(fabs(A1), tol);
		A2 = max(fabs(A2), tol);
		if(fabs(A3)<tol || isnan(fabs(A3)))
		{
			A3 = tol;
		}
		
		if(fabs(A1) < fabs(A2) || fabs(A2) < fabs(A3))
		{		
			double rel1 = fabs(fabs(A1)-fabs(A2))/(max(fabs(A1),fabs(A2)));
			if((rel1 < 1e-3 && rel1 > 0.0))
			{
				swap(A1,A2);// Swap the values of A1 and A2
				e1.swap(e2);// Swap the vectors of A1 and A2
			}
			double rel2 = fabs(fabs(A2)-fabs(A3))/(max(fabs(A2),fabs(A3)));
			if((rel2 < 1e-3 && rel2 > 0.0))
			{
				swap(A2,A3);// Swap the values of A1 and A2
				e2.swap(e3);// Swap the vectors of A1 and A2
			}
		}

		if(fabs(A1) < fabs(A2) || fabs(A2) < fabs(A3))
		{
			cout<<"Either A1 < A2 or A2 < A3"<<endl;
			cout<<A1<<" "<<A2<<" "<<A3<<endl;
			exit(1);
		}
		aniso_loc[0] = A1;
		aniso_loc[1] = A2;
		aniso_loc[2] = A3;

		// We are going to store the eigen vectors of the rotation matrix
		// The eigen vectors are e1 e2 e3 corresponding to A1, A2 and A3 respectively
		// We store them in the order e1 followed by e2 followed by e3.
		// This is the maximum direction of the solution NOT OF THE OPTIMUM ELLIPSE
		// Remember that the ordering of the eigen has to be switched when
		// calculating the rotation matrix for the ellipse
	    // Storing the first column as the direction of A3
	    aniso_loc[D+0] = e1[0];aniso_loc[D+1] = e1[1];aniso_loc[D+2] = e1[2];
	    // Storing the second column as the direction of A2
	    aniso_loc[D+3] = e2[0];aniso_loc[D+4] = e2[1];aniso_loc[D+5] = e2[2];
	    // Storing the first column as the direction of A1
	    aniso_loc[D+6] = e3[0];aniso_loc[D+7] = e3[1];aniso_loc[D+8] = e3[2];
		// Need to check if this is correct
	}
}
