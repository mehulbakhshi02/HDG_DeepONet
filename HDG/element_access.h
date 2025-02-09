// template<int D> FlatVector<>   ComputeDubinerBasis(int, IntegrationPoint &, Array<int> &, LocalHeap &);
// template<int D> FlatMatrix<>   ComputeDerDubinerBasis(int, IntegrationPoint &, Array<int> &, LocalHeap &);
template<int D> FlatMatrix<>   ComputeNthDerDubinerBasis(int, int, IntegrationPoint &, Array<int> &, LocalHeap &);
template<int D> FlatVector<>   JacobiPolynomial(int, double, double, double, LocalHeap &);
int Factorial(int);
double BiCoe(int, int);

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::GetElementInformation(const Space & space, vector<FacetData<D,COMP>*> & fadata, vector<ElementData<D,COMP>*> & eldata, shared_ptr<MeshAccess> ma, vector<int> & order_array, LocalHeap & lh){

  ndof_converter_segm.resize(max_order+1);
  for (int oo = 0; oo < max_order+1; ++oo) {
    ndof_converter_segm[oo].resize(get_ndof(oo, ET_SEGM));
    transfer_ndof(max_order, oo, ET_SEGM, &ndof_converter_segm[oo][0]);
  }

  ndof_converter_tri.resize(max_order+1);
  for (int oo = 0; oo < max_order+1; ++oo) {
    ndof_converter_tri[oo].resize(get_ndof(oo, ET_TRIG));
    transfer_ndof(max_order, oo, ET_TRIG, &ndof_converter_tri[oo][0]);
  }

  ndof_converter_quad.resize(max_order+1);
  for (int oo = 0; oo < max_order+1; ++oo) {
    ndof_converter_quad[oo].resize(get_ndof(oo, ET_QUAD));
    transfer_ndof(max_order, oo, ET_QUAD, &ndof_converter_quad[oo][0]);
  }

  ndof_converter_tet.resize(max_order+1);
  for (int oo = 0; oo < max_order+1; ++oo) {
    ndof_converter_tet[oo].resize(get_ndof(oo, ET_TET));
    transfer_ndof(max_order, oo, ET_TET, &ndof_converter_tet[oo][0]);
  }

  ndof_converter_hex.resize(max_order+1);
  for (int oo = 0; oo < max_order+1; ++oo) {
    ndof_converter_hex[oo].resize(get_ndof(oo, ET_HEX));
    transfer_ndof(max_order, oo, ET_HEX, &ndof_converter_hex[oo][0]);
  }
  
  int nse = ma->GetNSE();
  nf  = ma->GetNFacets();
  ne  = ma->GetNE();
  min_y = 10000000000000.;

  ndof_w_total = ndof_q_total = ndof_l_total = 0;

  int order_geo = 0; // 4;
  int order_add = 0; // for turbulence

  Array<int> fnums, vnums, dnums, elnums, pnums;
  
  for (int i = 0; i < fadata.size(); ++i)
    delete fadata[i];
  fadata.clear();

  for (int i = 0; i < eldata.size(); ++i)
    delete eldata[i];
  eldata.clear();

  fadata.resize(nf);
  eldata.resize(ne);
    
  ndof_l_max = ndof_w_max = ndof_q_max = nip_max  = 0;

  int index = 0;
  for (int i = 0; i < nf; ++i) {    
    ma->GetFacetElements(i, elnums);

    if (elnums.Size() == 0) { fadata[i] = 0; continue; }

    int elnr1 = elnums[0];
    int elnr2 = (elnums.Size() == 2) ? elnums[1] : -1;

    ELEMENT_TYPE elt1 = ma->GetElement(ElementId(VOL,elnr1)).GetType();
    ELEMENT_TYPE elt2;
    ELEMENT_TYPE fat = ma->GetFacetType(i);

    int order1 = order_array[elnr1];
    int ndof_w1  = get_ndof(order1, elt1);
    int ndof_q1  = (Model::Diffusion || Model::Source) ? D*ndof_w1 : 0;
    ndof_w_max    = max(ndof_w1, ndof_w_max);
    ndof_q_max    = max(ndof_q1, ndof_q_max);

    int order2 = 0;
    int ndof_w2  = 0;
    int ndof_q2  = 0;
    if (elnr2 != -1) {
      elt2 = ma->GetElement(ElementId(VOL,elnr2)).GetType();
      order2 = order_array[elnr2]; 
      ndof_w2  = get_ndof(order2, elt2);
      ndof_q2  = (Model::Diffusion || Model::Source) ? D*ndof_w2 : 0;
      ndof_w_max = max(ndof_w2, ndof_w_max);
      ndof_q_max = max(ndof_q2, ndof_q_max);
    }

    int order_l = max(order1, order2);
    
    int ndof_l  = get_ndof(order_l, fat);
    if (ndof_w2 == 0) ndof_l = 0;
    ndof_l_max  = max(ndof_l_max, ndof_l); 
    
    //    const IntegrationRule & ir_facet = SelectIntegrationRule (fat, 3*(max_order+2));
    int order_int = 2 * order_l + 3 + order_geo + order_add;
    //    int order_int = 2 * order_l + 3 + order_geo + order_add;

    //    if (fat == ET_QUAD)
    //      order_int += 2;

    const IntegrationRule & ir_facet = SelectIntegrationRule (fat, order_int);

    fadata[i] = new FacetData<D,COMP> (index, ir_facet.GetNIP(), elnr1, elnr2, ndof_w1, ndof_w2, ndof_q1, ndof_q2, ndof_l);

    FacetData<D,COMP> & fd = *fadata[i];

    nip_max = max(nip_max, fd.nip);

    fd.order1  = order1;
    fd.order2  = order2;
    fd.order_l = order_l;
    fd.type = fat;

    ma->GetFacetPNums(i, pnums);
    fd.nodes.resize(D * pnums.Size());
    for (int k = 0; k < pnums.Size(); ++k) {
      Vec<D> coord = ma->GetPoint<D>(pnums[k]);
      for (int dd = 0; dd < D; ++dd)
        fd.nodes[D*k+dd] = coord(dd);
    }
  }

  cout << "nipf_max=" << nip_max << endl;

  int zmf_num = 0; // zero-measure faces
  for (int i = 0; i < nse; ++i) {   
    fnums = ma->GetElFacets(ElementId(BND, i));
    if (!fadata[fnums[0]]) continue;
    fadata[fnums[0]]->bcnr = ma->GetElIndex(ElementId(BND,i));

    if (D == 3 && ma->SurfaceElementVolume(i) < 1.e-16) {
      fadata[fnums[0]]->bcnr = -2; // zero-measure face
      //      cout << "Zero-measure face: " << ma.SurfaceElementVolume(i) << endl;
      zmf_num++;
    }
  }

  if (zmf_num > 0)
    cout << "Warning: " << zmf_num << " zero-measure faces found. These will be ignored" << endl;
  
  vector<double> work(ndof_w_max * ndof_w_max);
  vector<int> ipiv(ndof_w_max);
  
  int offset_w = 0;
  int offset_q = 0;

  nf_max = 0;

  for (int i = 0; i < ne; ++i) {
    HeapReset hr(lh);

    int order_wi  = order_array[i];

    fnums = ma->GetElFacets(i);
    vnums = ma->GetElVertices(ElementId(VOL,i));

    const ScalarFiniteElement<D> &      fel_w = dynamic_cast<const ScalarFiniteElement<D>&> (space.fes_w->GetFE(i, lh));
    const FacetVolumeFiniteElement<D> & fel_l = dynamic_cast<const FacetVolumeFiniteElement<D>&> (space.fes_l->GetFE(i, lh));

    // This allows quads to be used! 
    // However, slight changes are to be expected in transfer_ndofs.h
    // Take a look at elementtopology.hpp in netgen include directory. 
    ELEMENT_TYPE eltype = fel_w.ElementType();
    // Number of element faces.
    int elnf = ElementTopology::GetNFacets (eltype);
    int elnv = ElementTopology::GetNVertices (eltype);

    int ndof_w = get_ndof(order_wi, eltype);
    int ndof_q = (Model::Diffusion || Model::Source) ? D * ndof_w : 0;

    ndof_w_total += ndof_w;
    ndof_q_total += ndof_q;

    int * conv_loc_w;
    if (eltype == ET_TRIG)
      conv_loc_w = &ndof_converter_tri[order_wi][0];
    else if (eltype == ET_QUAD)
      conv_loc_w = &ndof_converter_quad[order_wi][0];
    else if (eltype == ET_TET)
      conv_loc_w = &ndof_converter_tet[order_wi][0];
    else if (eltype == ET_HEX)
      conv_loc_w = &ndof_converter_hex[order_wi][0];
    else {
      cout << "Unsupported element type" << endl;
      exit(0);
    }      

    // ************************************************************************
    // Interface with Netgen. 
    // Prerequisites: fes and fes_sigma are Finite Element spaces belonging
    // to the maximum allowable polynomial degree. 
    ElementTransformation & eltrans = ma->GetTrafo(i, lh);

    int order_int = 2 * order_wi + MLParameters::quadadd + order_geo + order_add;
    //    int order_int = 2 * order_wi + 2 + order_geo + order_add;

    //    if (eltype == ET_QUAD || eltype == ET_HEX)
    //      order_int += D;   
    
    const IntegrationRule & ir_vol = SelectIntegrationRule (eltype, order_int);
    MappedIntegrationRule<D, D> mir_vol(ir_vol, eltrans, lh);

    // Normals
    Facet2ElementTrafo transform(eltype, vnums);
    FlatVector< Vec<D> > normals = ElementTopology::GetNormals<D>(eltype);    
    
    // Allocate a new ElementData item. 
    eldata[i] = new ElementData<D,COMP> (i, ir_vol.GetNIP(), elnf, ndof_w, ndof_q);
    ElementData<D, COMP> & elidata = *eldata[i];

    nip_max = max(nip_max, elidata.nip);

    elidata.order    = order_wi;
    elidata.offset_w = offset_w;
    elidata.offset_q = offset_q;

    elidata.type = eltype;

    elidata.ndof_lt = 0;

    vector<int> nodeorder(elnv);    
    for (int k = 0; k < elnv; ++k)
      nodeorder[k] = k;

    for (int k = 0; k < elnf-1; ++k)
      if (vnums[nodeorder[k]] > vnums[nodeorder[k+1]]) swap(nodeorder[k], nodeorder[k+1]);

    if (vnums[nodeorder[0]] > vnums[nodeorder[1]]) swap(nodeorder[0], nodeorder[1]);

    elidata.nodes.resize(D * elnv);
    if (eltype == ET_TRIG) {
      nodeorder[0] = 1; nodeorder[1] = 2; nodeorder[2] = 0;
    }
    for (int k = 0; k < elnv; ++k) {
      Vec<D> coord = ma->GetPoint<D>(vnums[k]);
      for (int dd = 0; dd < D; ++dd)
        elidata.nodes[D*nodeorder[k]+dd] = coord(dd);
    }
    int ndof_vol_max = get_ndof(max_order, eltype);
    FlatVector<> shapeW(ndof_vol_max, lh);
    FlatMatrixFixWidth<D> dshapeW(ndof_vol_max, lh); 

    nf_max = max(nf_max, elnf);

    double minlength = 1000000000.;
    double maxlength = 0.;

    double surf = 0.0;
    for (int j = 0; j < elnf; ++j) {
      HeapReset hr(lh);

      ELEMENT_TYPE etfacet = ElementTopology::GetFacetType (eltype, j);
      FacetData<D, COMP> & fd = *fadata[fnums[j]];

      int ndof_l = fd.ndof_l;
      elidata.ndof_lt += ndof_l;
      int order_li = fd.order_l;

      int * conv_loc_l;
      if (etfacet == ET_SEGM)
        conv_loc_l = &ndof_converter_segm[order_li][0];
      else if (etfacet == ET_TRIG)
        conv_loc_l = &ndof_converter_tri[order_li][0];
      else if (etfacet == ET_QUAD)
        conv_loc_l = &ndof_converter_quad[order_li][0];
      else {
        cout << "Unsupported element type" << endl;
        exit(0);
      }

      int order_int = 2 * order_li + 3 + order_geo + order_add;
      //    int order_int = 2 * order_wi + 2 + order_geo + order_add;

      //      if (etfacet == ET_QUAD)
      //        order_int += 2; 
      
      const IntegrationRule & ir_facet = SelectIntegrationRule (etfacet, order_int);
      IntegrationRule & ir_facet_vol = transform(j, ir_facet, lh);
      MappedIntegrationRule<D, D> mir_facet(ir_facet_vol, eltrans, lh);

      Vec<D> normal_ref = normals[j];
      
      elidata.faces[j] = fnums[j];

      if (fd.bcnr == -2) continue; // zero-measure face      

      if (i == fd.elnr1)  {
        fd.offset_w1 = offset_w;
        fd.offset_q1 = offset_q;
        if (Model::Diffusion)
          fd.q1.resize(COMP * D * fd.nip);
      } else {
        fd.offset_w2 = offset_w;
        fd.offset_q2 = offset_q;
        if (Model::Diffusion)
          fd.q2.resize(COMP * D * fd.nip);
      }

      int ndof_facet_max = get_ndof(max_order, etfacet);
      FlatVector<> shapeL(ndof_facet_max, lh);
      double facetsurf = 0.0;

      for (int k = 0; k < fd.nip; ++k) {

        // Transform the normal
        Mat<D> inv_jac = mir_facet[k].GetJacobianInverse();
        double det     = mir_facet[k].GetJacobiDet();
        Vec<D> normal = det * Trans (inv_jac) * normal_ref;       
        double len    = L2Norm (normal);
        normal       /= len;

        // Calculate the ansatz functions at the boundary
        fel_w.CalcShape(ir_facet_vol[k], shapeW);
        if (i == fd.elnr1)  {
          for (int pp = 0; pp < ndof_w; ++pp)
            fd.phi1[pp+ndof_w*k] = shapeW(conv_loc_w[pp]);
          for (int dd = 0; dd < D; ++dd)
            fd.n1[dd+D*k] = normal[dd];
        } else {
          for (int pp = 0; pp < ndof_w; ++pp)
            fd.phi2[pp+ndof_w*k] = shapeW(conv_loc_w[pp]);
          for (int dd = 0; dd < D; ++dd)
            fd.n2[dd+D*k] = normal[dd];
        }

        // Calculate the hybrid functions at the boundary
        if (fd.ndof_l != 0 && elidata.ndof_w == max(fd.ndof_w1, fd.ndof_w2)) {
          fel_l.Facet(j).CalcShape(ir_facet_vol[k], shapeL);

          for (int pp = 0; pp < ndof_l; ++pp)
            fd.mu[pp+ndof_l*k] = shapeL(conv_loc_l[pp]);
        }

        fd.qw[k] = len * ir_facet[k].Weight();

        for (int dd = 0; dd < D; ++dd)
          fd.qp[dd+D*k] = mir_facet[k].GetPoint()[dd];

        surf += fd.qw[k];	
        facetsurf += fd.qw[k];
      }
      fd.h = facetsurf;
      if (fd.h < 1.e-10)
        cout << "Zero: " << i << " " << j << " " << fd.h << endl;
      minlength = min(fd.h, minlength);
      maxlength = max(fd.h, maxlength);
    }
    // Calculate for the CFL number a measure of max(|DX|, |DY|)
    // by using the edges
    //    elidata.h = minlength;
    elidata.surf = surf;

    min_y = min(min_y, 0.5 * minlength);

    // Assemble space for the reconstructed variables
    double vol = 0.0;
    for (int dd = 0; dd < D; ++dd)
      elidata.cntr[dd] = 0.0;

    for (int j = 0; j < elidata.nip; j++) {
    
      fel_w.CalcShape(ir_vol[j], shapeW);
      fel_w.CalcMappedDShape(mir_vol[j], dshapeW);
      
      for (int pp = 0; pp < ndof_w; ++pp) {
        elidata.phi[pp+ndof_w*j]           = shapeW(conv_loc_w[pp]);
        for (int dd = 0; dd < D; ++dd)
          elidata.dphi[pp+ndof_w*(dd+D*j)] = dshapeW(conv_loc_w[pp], dd);
      }

      elidata.qw[j] = mir_vol[j].GetWeight();

      for (int dd = 0; dd < D; ++dd)
        elidata.qp[dd+D*j] = mir_vol[j].GetPoint()[dd];

      for (int dd = 0; dd < D; ++dd)
        elidata.cntr[dd] += elidata.qw[j] * elidata.qp[dd+D*j];

      vol += elidata.qw[j];
    }
    
    elidata.vol = vol;
    elidata.h   = 2. * vol / maxlength;
    //    min_y = min(min_y, elidata.h);

    for (int dd = 0; dd < D; ++dd)
      elidata.cntr[dd] /= elidata.vol;

    for (int p = 0; p < elidata.ndof_w; p++) {
    	for (int q = 0; q < elidata.ndof_w; q++) {
        double res = 0.0;
    	  for (int j = 0; j < elidata.nip; j++) 
    	    res += elidata.qw[j] * elidata.phi[ndof_w*j+p] * elidata.phi[ndof_w*j+q];
    	  elidata.mass[p*ndof_w+q] = res;
    	}
    }

    elidata.inv_mass = elidata.mass;

    int info;
    int lwork = ndof_w * ndof_w;

    dgetrf_(&ndof_w, &ndof_w, &elidata.inv_mass[0], &ndof_w, &ipiv[0], &info);
    dgetri_(&ndof_w, &elidata.inv_mass[0], &ndof_w, &ipiv[0], &work[0], &lwork, &info);

    elidata.is_curved = ma->GetElement(ElementId(VOL,i)).is_curved;

    offset_w += ndof_w;
    offset_q += ndof_q;
  }

  cout << "nip_max="<< nip_max << endl;

  int lfd_index = 0;

  for (int i = 0; i < nf; ++i) {    
    if (!fadata[i]) continue;

    FacetData<D, COMP> & fd = *fadata[i];

    if (fd.ndof_l == 0) continue;

    int ndof_l = fd.ndof_l;
    fd.offset_l = lfd_index;

    for (int p = 0; p < fd.ndof_l; p++) {
      for (int q = 0; q < fd.ndof_l; q++) {
        double res = 0.0;
        for (int j = 0; j < fd.nip; j++) 
          res += fd.qw[j] * fd.mu[ndof_l*j+p] * fd.mu[ndof_l*j+q];
        fd.mass[p*ndof_l+q] = res;
      }
    }
    
    fd.inv_mass = fd.mass;

    int info;
    int lwork = ndof_l * ndof_l;

    dgetrf_(&ndof_l, &ndof_l, &fd.inv_mass[0], &ndof_l, &ipiv[0], &info);
    dgetri_(&ndof_l, &fd.inv_mass[0], &ndof_l, &ipiv[0], &work[0], &lwork, &info);

    lfd_index += ndof_l;
  }
  ndof_l_total = lfd_index;

  cout << min_y << endl;
}

// template<int D>
// FlatVector<> ComputeDubinerBasis(int order_i, IntegrationPoint &ip, Array<int> & vnums, LocalHeap &lh) {

//   int ndof_sml = get_ndof(order_i, ET_TRIG);
//   int ndof_1d = order_i +1;  

//   FlatVector<> res(ndof_sml, lh);
//   FlatVector<> psi_a(ndof_1d, lh);
//   FlatMatrix<> psi_b(ndof_1d, ndof_1d, lh);

//   for(int i=0; i<ndof_1d; i++)
//     for(int j=0; j< ndof_1d; j++)
//       psi_b.Row(i)[j]=0.;
    
//   double x,y,a,b,xi,eta;
//   int nn;
//   x = ip(0); 
//   y = ip(1);
    
//   double lam[3] = { x, y, 1-x-y };
//   INT<4> f = ET_trait<ET_TRIG>::GetFaceSort (0, vnums);
//   xi = lam[f[0]];  
//   eta = lam[f[1]];

//   a = 2*eta/(1-xi) - 1.0;  // notice the interchange of x and y to be consistent with ngsolve
//   b = 2*xi - 1.;

//   psi_a = JacobiPolynomial<D>( order_i, 0, 0, a, lh);

//   for (int n=0; n < ndof_1d; n++) {
//         nn = order_i - n;
// 	psi_b.Rows(0,nn+1).Col(n) =  pow((.5*(1.-b)),n) * JacobiPolynomial<D>(nn, 2*n+1,0,b, lh);
//   }

//   int ii=0;
//   for (int n=0; n < ndof_1d; n++) {
//     for (int j=0; j < ndof_1d - n; j++) {
//       res[ii]=psi_a[j] * psi_b.Row(n)[j]; // non-normalised 
// 	ii=ii+1;
//        }
//     }
  

//   return res;
// }

// template<int D>
// FlatMatrix<> ComputeDerDubinerBasis(int order_i, IntegrationPoint &ip, Array<int> & vnums, LocalHeap &lh) {

//   int ndof_sml = get_ndof(order_i, ET_TRIG);
//   int ndof_1d = order_i +1;  

//   FlatMatrix<> res(ndof_sml, D, lh);
//   FlatVector<> psi_a(ndof_1d, lh);
//   FlatMatrix<> psi_b(ndof_1d, ndof_1d, lh);
//   FlatMatrix<> psi_a_d(ndof_1d,1,lh);
//   FlatMatrix<> psi_b_d(ndof_1d,ndof_1d,lh);

//   for(int i=0; i<ndof_1d; i++)
//     for(int j=0; j< ndof_1d; j++)
//       psi_b.Row(i)[j]=0.;
    
//   double x,y,a,b, xi, eta;
//   int nn;
//   x = ip(0); 
//   y = ip(1);

//   double lam[3] = { x, y, 1-x-y };
//   INT<4> f = ET_trait<ET_TRIG>::GetFaceSort (0, vnums);
//   xi = lam[f[0]];  
//   eta = lam[f[1]];

//   double d_xi_x,d_xi_y,d_eta_x,d_eta_y;
//   d_xi_x = 0.;
//   d_xi_y = 0.;
//   d_eta_x = 0.;
//   d_eta_y = 0.;

//   if(f[0] == 0){ 
//     d_xi_x = 1.;  d_xi_y = 0.;
//   }

//   if(f[0] == 1){ 
//     d_xi_x = 0.;  d_xi_y = 1.;
//   }

//   if(f[0] == 2){ 
//     d_xi_x = -1.; d_xi_y = -1.;
//   }

//   if(f[1] == 0){ 
//     d_eta_x = 1.; d_eta_y = 0.;
//   }

//   if(f[1] == 1){ 
//     d_eta_x = 0.; d_eta_y = 1.;
//   }

//   if(f[1] == 2){ 
//     d_eta_x = -1.; d_eta_y = -1.;
//   }

//   a = 2*eta/(1-xi) - 1.0; 
//   b = 2*xi - 1.;

//   psi_a = JacobiPolynomial<D>( order_i, 0, 0, a, lh);

//   for (int n=0; n < ndof_1d; n++) {
//         nn = order_i - n;
// 	psi_b.Rows(0,nn+1).Col(n) =  JacobiPolynomial<D>(nn, 2*n+1,0,b, lh);
//   }

//   for(int i=0; i< ndof_1d; i++)
//     psi_a_d(i) = 0.;

//   psi_a_d.Rows(1,ndof_1d) = JacobiPolynomial<D>( order_i-1, 1, 1, a, lh);

//   for (int i=1; i < ndof_1d; i++)
//     psi_a_d(i)=0.5* (i+1) * psi_a_d(i);


//   FlatMatrix<> JP(ndof_1d,1,lh);
//   double jj;
//   for (int n=0; n< ndof_1d; n++){
//         nn = order_i - n;
//         if (nn > 0){
// 	  JP.Rows(0,nn) = JacobiPolynomial<D>(nn-1, 2*n+2, 1, b, lh);
// 	   for (int m=1; m <=nn ; m++){
//              jj = 0.5*(2*n+2 + m);
// 	     psi_b_d(m,n) = jj* JP(m-1);
// 	   }
//         }
//   }

  
//   double fact=0.5*(1.0-b);
//   int n1;
//   double res0, res1;

//    int ii=0;
//     for (int n=0; n < ndof_1d; n++) {
//       for (int j=0; j < ndof_1d-n; j++) {
//        n1 = max(j-1,0); // prevent division by zero...
//        res(ii,1) = 2 * psi_a_d(j) * psi_b(n,j) * pow(fact,n1);
//        res(ii,0) = psi_a_d(j) * psi_b(n,j) * pow(fact,n1);
//        res(ii,0) = res(ii,0) * (1.0+a);
//        res(ii,0) = res(ii,0) - 2 * psi_a(j)* j * 0.5 * psi_b(n,j) * pow(.5*(1.-b),n1);
//        res(ii,0) = res(ii,0) + 2 * psi_a(j) * psi_b_d(n,j) * pow(.5*(1.-b),j);
//        res0 = res(ii,0);
//        res1 = res(ii,1);
//        res(ii,0) = res0 * d_xi_x + res1 * d_eta_x;
//        res(ii,1) = res0 * d_xi_y + res1 * d_eta_y;
//        ii = ii+1;
//       }
//      }

//   return res;
// }

/*!
 * Computes the Nth derivative of the Dubiner basis
 * @param[in] order_i - Contains the order of the current cell which is p+1 in the current place
 * @param[in] order_der - Contains the order of the derivative to be calculated. 
 * @param[in] ip - contains the integration point at which we want to calculate the derivative
 * @param[in] vnums - contains the vertices of the element used in the transformation
 * @param[in] lh - It is the local heap used to pass to some of the other functions
 */
template<int D>
FlatMatrix<> ComputeNthDerDubinerBasis(int order_i, int order_der, IntegrationPoint &ip, Array<int> & vnums, LocalHeap &lh) { 
				        
  // The ndof_sml is got using the order of the element which is p+1 now
  // ndof_sml = (order_i+1)*(order_i+2)/2
  // ndof_sml = (p+1+1)*(p+1+2)/2
  int ndof_sml = get_ndof(order_i, ET_TRIG);
  // Is set to be p+2 as this is for 1D
  int ndof_1d = order_i +1;  
  // Dimension is ndof_sml*order_der
  FlatMatrix<> res0(ndof_sml, order_der+1, lh);
  // Dimension is ndof_sml*order_der
  FlatMatrix<> res(ndof_sml, order_der+1, lh);
    // Dimension is p+2
  FlatVector<> psi_a(ndof_1d, lh);
  FlatMatrix<> psi_b(ndof_1d, ndof_1d, lh);
  FlatMatrix<> psi_a_d(ndof_1d,order_der+1,lh);
  FlatMatrix<> psi_b_d(ndof_1d*(order_der+1),ndof_1d,lh);
  FlatMatrix<> phi_d(order_der+1,order_der+1,lh);
  phi_d=0.;
  FlatMatrix<> der_a(order_der+1,order_der+1,lh);
  der_a = 0.;

  for(int i=0; i<ndof_1d; i++)
    for(int j=0; j< ndof_1d; j++)
      psi_b.Row(i)[j]=0.;
    
  double x,y,a,b,xi,eta;
  int nn;
  x = ip(0); 
  y = ip(1);

  double lam[3] = { x, y, 1-x-y };
  INT<4> f = ET_trait<ET_TRIG>::GetFaceSort (0, vnums);

  xi = lam[f[0]];  
  eta = lam[f[1]];

  double d_xi_x,d_xi_y,d_eta_x,d_eta_y;
  d_xi_x = 0.;
  d_xi_y = 0.;
  d_eta_x = 0.;
  d_eta_y = 0.;

  if(f[0] == 0){ 
    d_xi_x = 1.; d_xi_y = 0.;
  }

  if(f[0] == 1){ 
    d_xi_x = 0.; d_xi_y = 1.;
  }

  if(f[0] == 2){ 
    d_xi_x = -1.; d_xi_y = -1.;
  }

  if(f[1] == 0){ 
    d_eta_x = 1.; d_eta_y = 0.;
  }

  if(f[1] == 1){ 
    d_eta_x = 0.; d_eta_y = 1.;
  }

  if(f[1] == 2){ 
    d_eta_x = -1.; d_eta_y = -1.;
  }

  a = 2*eta/(1-xi) - 1.0;  
  b = 2*xi - 1.;

  psi_a = JacobiPolynomial<D>( order_i, 0, 0, a, lh);

  for (int n=0; n < ndof_1d; n++) {
        nn = order_i - n;
	psi_b.Rows(0,nn+1).Col(n) =  JacobiPolynomial<D>(nn, 2*n+1,0,b, lh);
  }


  psi_a_d.Rows(0,ndof_1d).Col(0) = psi_a; 
  for(int i=0; i< ndof_1d; i++)
    for (int j =0; j < order_der; j++)
      psi_a_d(i,j+1) = 0.;

  for (int k=0; k< order_der; k++)
    psi_a_d.Rows(k+1,ndof_1d).Col(k+1) = JacobiPolynomial<D>( order_i-1-k, k+1, k+1, a, lh);

  double factor;
  for (int k=0; k < order_der; k++)
    for (int i= k+1; i < ndof_1d; i++){
      factor  = Factorial(i+1+k) / (pow(2,k+1) * Factorial(i));
      psi_a_d(i,k+1) = factor * psi_a_d(i,k+1);
    }

  for (int n=0; n< ndof_1d *(order_der+1); n++)
    for(int m=0; m < ndof_1d; m++)
      psi_b_d(n,m) = 0.;

  for (int n=0; n < ndof_1d; n++)
    psi_b_d.Rows(0,ndof_1d).Col(n) = psi_b.Rows(0,ndof_1d).Col(n);

  FlatMatrix<> JP(ndof_1d,1,lh);
  int ind;
  for (int k=0; k < order_der; k++){
    for (int n=0; n< ndof_1d; n++){
        nn = order_i - n;
        if (nn > 0){
	  JP.Rows(0,nn-k) = JacobiPolynomial<D>(nn-(k+1), 2*n+1+(k+1), (k+1), b, lh);
	   for (int m=k+1; m <=nn ; m++){
             factor =  Factorial(2*n+1+m+(k+1)) / (pow(2,k+1) * Factorial(2*n+1+m));
	     ind = (k+1)*ndof_1d + m;
	     psi_b_d(ind,n) = factor * JP(m-k-1);
	   }
        }
    }
  }
 

  int n1;
  
  double a_xi,a_eta,b_xi, b_eta;
  a_xi = 2*(1+a)/(1-b);
  a_eta = 4/(1-b);
  b_xi = 2.;
  b_eta = 0.;
  
  double d_g;
  double sumk, suml, sumr;
  double t1,t2,t3,t4,t5,t6;

  for(int p=0; p<= order_der; p++){
    phi_d(p,0) = 0.;
    for(int j=1; j <= order_der; j++){
      phi_d(p,j) = pow(2*eta,j) * Factorial(j+p-1) * pow(1.-xi,-j-p) / Factorial(j-1);
    }
  }
  phi_d(0,0) = 1.; 

  double A_pi, der_f;
  for (int p=0; p<=order_der; p++){
    for(int j=0; j<=order_der; j++){
      der_a(p,j) = 0;
      for (int i=0; i<=p; i++){
	A_pi = 0;
	for(int k=0; k<=i; k++){
	  A_pi += pow(-1,i-k) * pow(a+1,i-k) * phi_d(p,k) / (Factorial(k)*Factorial(i-k));
	}
	der_f = Factorial(j) * pow(a,j-i) / Factorial(j-i);
	if(j-i < 0) der_f = 0;
	if(j==0 && i==0) der_f =1.;
	der_a(p,j) += A_pi * der_f;
      }
    }
  }
 
  double tt4;
  int ii=0;
    for (int n=0; n < ndof_1d; n++) {
      for (int j=0; j < ndof_1d-n; j++) {
        for (int r=order_der ; r >= 0; r--){
	  suml = 0.;
          for(int l=0; l <= r; l++){
	    sumk = 0.;
	    for(int k=0; k <= order_der-r; k++){
	      t5 = 0.;
	      for(int q=0; q <= l+k; q++){
		d_g =  pow((1.-b),(j-l-k+q));
		if(j-l-k+q < 0) d_g = 0.; 
		t6 = pow(0.5,j) * Factorial(j) * d_g * pow((-1),(l+k-q)) / Factorial(j-l-k+q);
		ind = q * ndof_1d + n;
		t5 += BiCoe(l+k,q) * t6 * psi_b_d(ind,j);
	      }
	      t2 = pow(b_xi,l) * pow(b_eta,k) * t5;
	      t1 = 0.;
	      for (int p=0; p<=r-l; p++){
		t3 = pow(2,order_der-r-k) * Factorial(order_der-k-l-p-1) * pow(1-xi,p+k+l-order_der) / Factorial(order_der-r-k-1);
		if(order_der-r-k == 0){
		  t3 = 0.;
		  if(r-l-p ==0 )  t3 = 1.;
		}

		t4=0;
		for(int i=0; i<=p; i++){
		  A_pi=0;
		  for(int m=0; m<=i; m++){
		    A_pi += pow(-1,i-m) * pow(a,i-m) * der_a(p,m) / (Factorial(m) * Factorial(i-m));
		  }
		  t4 += A_pi * psi_a_d(j,order_der-r-k+i);
		}

	       	t1 += BiCoe(r-l,p) * t3 * t4;
	      }

	      sumk = sumk + BiCoe(order_der-r,k) * t1 * t2;
	    }
	    suml = suml + BiCoe(r,l) * sumk;
	  }

	  res0(ii,order_der-r) = suml;
	}
	ii = ii+1;
      }
    }
     
  for (int ii=0; ii < ndof_sml; ii++){
    for (int r=order_der ; r >=0.; r--){
      res(ii,order_der-r) = 0.;
      for (int l=0; l <=r; l++){
	for (int k=0; k <= order_der-r; k++){
	  res(ii, order_der-r) += BiCoe(r,l) * BiCoe(order_der-r,k) * pow(d_xi_x,r-l) * pow(d_eta_x,l) * pow(d_xi_y,order_der-r-k) * pow(d_eta_y,k) * res0(ii,l+k);
	}
      }
    }
  }

 
  
  return res;

}


template<int D>
FlatVector<> JacobiPolynomial(int order_i, double alpha, double beta, double q, LocalHeap &lh) {

  int ndof_sml = order_i +1;
  FlatVector<> res(ndof_sml, lh);
  double c1,c2,c3,c4;

  res[0] = 1.0;
  res[1] = ( 1.0 + 0.5 * ( alpha + beta ) ) * q  + 0.5 * ( alpha - beta );
 
  for (int i = 2 ; i< ndof_sml ; i++){

    c1 = 2 * i * ( i + alpha + beta ) * ( 2 * i - 2 + alpha + beta );
    c2 = ( 2 * i - 1 + alpha + beta ) * ( 2 * i + alpha + beta ) * ( 2 * i - 2 + alpha + beta );
    c3 = ( 2 * i - 1 + alpha + beta ) * ( alpha + beta ) * ( alpha - beta );
    c4 = - 2 * ( i - 1 + alpha ) * ( i - 1 + beta )  * ( 2 * i + alpha + beta );
    res[i] = ( ( c3 + c2 * q ) * res[i-1] + c4 * res[i-2] ) / c1;

  }

  return res;
}



int Factorial(int x){
  if(x<=0) {
    return 1.;
  }
  return (x==1 ? x : x*Factorial(x-1) );
}


double BiCoe(int n, int r){
  double res;
  res = Factorial (n)/ (Factorial(r) * Factorial(n-r));
  return res;
}
