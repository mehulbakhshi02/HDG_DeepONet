/** @file 
 * @brief This function is used to calculate the highest order derivative.
 * @param[in] - i - Contains the index of the cell. Size = 1 int
 * @param[in] - no_aniso - Contains the order upto which we need to reconstruct. Size = 1 int
 * @param[in] - sensor - Contains coefficients for the solution for one component. Size = 1 double pointer
 * @param[out] - dw - Contains the highest order derivatives for the term computed by sensor. Size = (order_der_loc+1) double
 * @param[in] - lh - Local heap.
*/
template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeHighOrderDerivative(const int i, const int order_der_loc, double* sensor, vector<double> &dw, LocalHeap & lh)
{
  // Hack for scalar. Dubiner basis only works for scalar equations
  if(D == 2 && AnisotropyData::projection_basis == AnisotropyData::dubiner_basis)
  {
    ElementData<D, COMP> & ed = *eldata[i];
    ElementTransformation & eltrans = ma->GetTrafo(i, lh);
    // int order_i_loc = ed.order;//Currently p+1 where p is the solution order
    // int ndof_w = ed.ndof_w;
    int order_i_loc = order_der_loc;//Currently p+1 where p is the solution order
    int ndof_w = (order_i_loc + 1) * (order_i_loc + 2) / 2; //number of monomial basis function.

    // ComputeDerivativeOrthogonal(i, dw_temp, sensor);
    FlatMatrix<> ndshape_ref(ndof_w, order_der_loc+1, lh);
    Array<int> vnums; 
    vnums = ma->GetElement(ElementId(VOL,i)).Vertices();
    const IntegrationRule & ir2d = SelectIntegrationRule (ET_TRIG, 3*(order_i_loc+1));
    vector<double>dwt(order_der_loc+1, 0.0);
    // Resizing the vector that will contain the highest order derivatives
    // As Dubiner basis is only in 2D we have no other size
    dw.resize(order_der_loc+1,0.0);
    // //only taking the first integration point since p+1st der is constant for p (only for p+1 der)
    for (int j = 0; j < 1; ++j) {
      ndshape_ref = ComputeNthDerDubinerBasis<D>(order_i_loc, order_der_loc, ir2d[j], vnums, lh);

      for (int pp = 0; pp < ndof_w; ++pp)
        for (int dd2 = 0; dd2 < order_der_loc+1; ++dd2)
        {
          dw[dd2] += ndshape_ref(pp, dd2) * sensor[pp];
        }


      // Bring it to the physical domain
      MappedIntegrationPoint<D,D> sip (ir2d[0], eltrans);
      Mat<D> inv_jac = sip.GetJacobianInverse();

      double eta_x = inv_jac(0,0);
      double eta_y = inv_jac(0,1);
      double xi_x = inv_jac(1,0);
      double xi_y = inv_jac(1,1);
      
      int n = order_der_loc;
      for (int p=n ; p >= 0; p--) {
        double sum_r = 0;
        int k = n-p;
        for(int r=0; r <=k; r++) {
          double sum_l=0.;
          for(int l=0; l<=p; l++) 
          {
            double der_val = dw[n-r-l];
            sum_l += BiCoe(p,l) * der_val * pow(xi_x,p-l) * pow(eta_x,l);
          }
          sum_r += BiCoe(k,r) * pow(xi_y,k-r) * pow(eta_y,r) * sum_l;
        }
        dwt[n-p] = sum_r;
      }

      for (int pp = 0; pp < order_der_loc+1; ++pp)
          dw[pp] = dwt[pp];

    }// end of loop over integration points(just over the first integration point because we have only cell wise
  }
  else if((D == 2 || D == 3) && AnisotropyData::projection_basis == AnisotropyData::monomial_basis)
  {
    ElementData<D, COMP> & ed = *eldata[i];

    // referencing the current element.
    // Need to check if this is correct for hp
    // Hack for h only
    // We need to use the order upto which we want to compute
    // the higher order derivative
    int p = order_der_loc;
    int nip = ed.nip; //number of integration point in the element.
    int ndof = int((p + 1.) * (p + 2.) / 2.); //number of monomial basis function.
    // Setting the Dof correctly for 3D
    if((D == 3))
    {
      ndof = int((p + 1.) * (p + 2.) * (p + 3.)/6.);
    }
    vector<double> vecWH1(ndof_w_max);
    vector<double> H1basis(ndof * nip_max);
    // Calculating the actual monomial basis
    int index2 = 0;
    for(int m=0;m<=p;++m)
    {
      for(int n=0;n<=p-m;++n)
      {
        for(int l=0;l<=(p-m-n)*(D%2);++l)
        {
          for(int j=0; j < nip ; ++j)
          {
            int index3 = index2 * nip + j;
            double x = ed.qp[j*D];
            double y = ed.qp[j*D+1];
            double z ;
            x -= ed.cntr[0];
            y -= ed.cntr[1];
            x = x/pow(ed.vol, 1.0/(double)D);
            y = y/pow(ed.vol, 1.0/(double)D);

            if(D == 3)
            {
              z = ed.qp[j*D+2];
              z -= ed.cntr[2];
              z = z/pow(ed.vol, 1.0/(double)D);
            }
            else
            {
              z = 1.0;
            }
            H1basis[index3] = pow(x,m)*pow(y,n)*pow(z,l);
          }
          index2 = index2 + 1;
        }
      }
    }
    //Gram schmidt Orthogonalization using L2 norm.
    vector<double> dot_coeff(((ndof-1)*ndof)/2); //one dimesional array that stores all the dot products of basis functions calculated while performing
    vector<double> basis_norm(ndof,0.0); //stores the L2 norm each basis function in each element.
    //gram schimdt orthogonalization.
    for (int pp = 0,index1=0; pp < ndof; ++pp) {

      // Orthogonalize twice(why)
      for (int n = 0; n < 1; ++n) {

        // Projection
        for (int qq = 0; qq < pp; ++qq,++index1) {

          double rpq = 0.;

          // Compute dot-product (phi_m, phi_l)_H1 (step 1)
          for (int j = 0; j < nip; ++j) {

            int idx_p = pp * nip + j;
            int idx_q = qq * nip + j;

            rpq += ed.qw[j] *   H1basis[idx_p] *   H1basis[idx_q];
          }
          dot_coeff[index1]=rpq;//storing the dot products.
          // Subtract projection (step 2)
          for (int j = 0; j < nip; ++j) {

            int idx_p = pp * nip + j;
            int idx_q = qq * nip + j;

            H1basis[idx_p]   -= rpq *   H1basis[idx_q];
          }
        }
      }
      //normalization in H1 norm(step 3)
      // Compute norm aka dot-product (phi_m, phi_m)_H1
      double rpp = 0.;

      for (int j = 0; j < nip; ++j) {

        int idx_p = pp * nip + j;
        //(norm)^2 compuatations
        rpp += ed.qw[j] *   H1basis[idx_p] *   H1basis[idx_p];
      }

      rpp = sqrt(rpp);
      basis_norm[pp]=rpp; //storing the l2 norm of each basis vector.
      // Normalization
      for (int j = 0; j < nip; ++j) {

        int idx_p = pp * nip + j;

        H1basis[idx_p]   /= rpp;
      }
    }
    // Need to check what this ndof_w
    int ndof_w = ed.ndof_w;
    double * ed_w;
    ed_w = new double [ed.nip];
    mygemm('t', 'n', 1, nip, ndof_w, 1., sensor, &ed.phi[0], 0., ed_w);

    //projection calclulation.
    double val = 0.;
    // for (int ll = 0; ll < COMP; ++ll) {
      for (int pp = 0; pp < ndof; ++pp) {

        for (int j = 0; j < ed.nip; ++j) {

          const double   qw =  ed.qw[j]; //weights associated with the gauss points
          int idx_pp = pp * nip + j;
          //H1 projection of solution on pp th basis  over the patch.
          //val=<phi_pp,W_ll>_{H1,patch} W_ll is the ll th componenet of Solution vector
          val += qw *   H1basis[idx_pp] * ed_w[j];
        }
        vecWH1[pp] = val;
        val = 0.0;
      }
    // }

    int index = 0;
    int mem = 0;
    // Calculating the memory allocation for calculating
    // the coefficients
    if(D == 2)
    {
      dw.resize(p+1,0.0);
      mem = (p+1)*(12*ndof-4*p*p-14*p);
      mem = mem/12;
    }
    else
    {
      dw.resize((p+1)*(p+2)/2,0.0);
      int temp = 0, temp1 = 0,temp2 = 0;
      for(int m = 0; m <= p; ++m ){
        for(int n=0; n <= (p-m) ; ++n){
          int  l = p-m-n;
          temp = ((p+1)*(p+2)*m);
          temp1 = (m*(m-1)*(2*p+3));
          temp1 = temp1/2;
          temp2 =  (m*(m-1)*(2*m-1));
          temp2 = temp2/6;
          temp= (temp - temp1 + temp2);
          index = index + temp/2;
          index = index + (p+1-m)*n - (n*(n-1))/2;
          index = index + l + 1;
          mem = mem + (ndof - index+1);
          index = 0;
        }
      }
    }
    vector<double> coeffs(mem,0.0);
    int check =0;
    index = 0;
    if(D == 2)
    {
      for(int m=0,index_coeff=0,index_derv=0;m<=p;++m,++index_derv)
      {
        int n = p-m;
        if(m == 0)
          index = n;
        else
          index = p*(m+1)-(m*(m-1))/2;
          for(int j = index; j < ndof ; ++j,++index_coeff)
          {
            if (j==index)
            {
              coeffs[index_coeff]=Factorial(n)*Factorial(m)/basis_norm[j];
              dw[index_derv] = dw[index_derv] + coeffs[index_coeff]* vecWH1[j] ;
            }
            else
            {
              for(int k = index; k < j ; k++)
              {
                coeffs[index_coeff]=coeffs[index_coeff]+
                (coeffs[index_coeff-j+k]*-1.0*dot_coeff[(j*(j-1))/2+k]/basis_norm[j]);
              }
              dw[index_derv] = dw[index_derv] + coeffs[index_coeff]*vecWH1[j];
            }
          }
          index = 0;
        }
    }
    else
    {
      int temp = 0, temp1 = 0,temp2 = 0;
      for(int m = 0,index_coeff = 0,index_derv = 0; m <= p; ++m )
      {
        for(int n=0; n <= (p-m) ; ++n,++index_derv)
        {
          int  l = p-m-n;
          temp = ((p+1)*(p+2)*m);
          temp1 = (m*(m-1)*(2*p+3));
          temp1 = temp1/2;
          temp2 =  (m*(m-1)*(2*m-1));
          temp2 = temp2/6;
          temp= (temp - temp1 + temp2);
          index = index + temp/2;
          index = index + (p+1-m)*n - (n*(n-1))/2;
          index = index + l;
          for(int j = index; j < ndof ; ++j,++index_coeff)
          {
            if(j == index)
            {
              coeffs[index_coeff]=Factorial(l)*Factorial(n)*Factorial(m)/basis_norm[j];
              dw[index_derv] = dw[index_derv] + coeffs[index_coeff]* vecWH1[j] ;

            }
            else
            {
              for(int k = index; k < j ; k++)
              {

                coeffs[index_coeff]=coeffs[index_coeff]+
                (coeffs[index_coeff-j+k]*-1.0*dot_coeff[(j*(j-1))/2+k]/basis_norm[j]);
              }
              dw[index_derv] = dw[index_derv] + coeffs[index_coeff]*vecWH1[j];
            } 
          }
          index = 0;
        }
      }
    }
    int der_sz = p+1;
    if(D==3)
      der_sz = (p+1)*(p+2)/2;
    for(int pp=0;pp<der_sz;pp++)
      dw[pp] = dw[pp]/pow(ed.vol, (double)p*1.0/(double)D);
  }
  else
  {
    cout<<"Dubiner Basis not implemented for 3D"<<endl;
    cout<<"Check projection_basis paramter in pde file!"<<endl;
    exit(0);
  }
}