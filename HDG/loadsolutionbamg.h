template <int D, int COMP, class Model>
bool UnifyingFramework<D, COMP, Model>
::LoadSolutionBamg(const string & filename, Solution & sol, LocalHeap & lh) const {

  if (D != 2) {
    cout << "[LoadSolutionBamg] This routine is only available for 2d problems" << endl;
    return false;
  }

  if (nf_max > 3) {
    cout << "[LoadSolutionBamg] This routine is only available for triangular meshes" << endl;
    return false;
  }

  ifstream file(filename.c_str(), ios::in);

  if (!file.is_open())
    return false;
  
  int nnode;
  double dummy;
  
  file >> dummy >> dummy >> nnode >> dummy;

  if (nnode != ma->GetNV()) {
    cout << "[LoadSolutionBamg] Number of nodes does not coincide with input mesh" << endl;
    return false;}

  // Read nodal values
  vector<double> w(nnode*COMP), q1(nnode*COMP), q2(nnode*COMP);

  for (int i = 0; i < nnode; ++i) {

    for (int ll = 0; ll < COMP; ++ll)
      file >> w[COMP*i+ll];

    if (Model::Diffusion) {
      for (int ll = 0; ll < COMP; ++ll)
        file >> q1[COMP*i+ll];
      for (int ll = 0; ll < COMP; ++ll)
        file >> q2[COMP*i+ll];
    }
  }
  
  file.close();

  Array<int> vnums;

  int order_nodal = 1;
  sol.vecQ.assign(ndof_q_total * COMP, 0.0);
  sol.vecW.assign(ndof_w_total * COMP, 0.0);
  sol.vecL.assign(ndof_l_total * COMP, 0.0);

  for(int i = 0; i < ne; ++i){

    HeapReset hr(lh);

    ElementData<D,COMP> & ed = *eldata[i];    
    
    const ScalarFiniteElement<D> & fel_nodal = dynamic_cast<const ScalarFiniteElement<D>&> (fes_nodal->GetFE(i, lh));
    const ScalarFiniteElement<D> & fel = dynamic_cast<const ScalarFiniteElement<D>&> (fspace.fes_w->GetFE(i, lh));

    int order_int = 2 * ed.order + 2 + 0 + 5;

    ELEMENT_TYPE eltype = fel.ElementType();
    ElementTransformation & eltrans = ma->GetTrafo(i, lh);
    const IntegrationRule & ir2d = SelectIntegrationRule (eltype, order_int);

    MappedIntegrationRule<D, D> mir_vol(ir2d, eltrans, lh);
    
    vnums = ma->GetElVertices(ElementId(VOL,i));

    int ndof_nodal = get_ndof(order_nodal, eltype);

    FlatVector<> shape_nodal (ndof_nodal, lh);
    FlatMatrixFixWidth<D> dshape_nodal (ndof_nodal, lh);

    int ndof_w = ed.ndof_w;
    
    vector<double> ww(ndof_w*COMP, 0.), qq(ndof_w*COMP*D, 0.);

    double * vecW = &sol.vecW[COMP * ed.offset_w];;
    double * vecQ = &sol.vecQ[COMP * ed.offset_q];;
    
    for (int pp = 0; pp < ndof_w; ++pp)
      for (int ll = 0; ll < COMP; ++ll)
        for (int j = 0; j < ed.nip; ++j){
          IntegrationPoint ip = ir2d[j];
          fel_nodal.CalcShape  (ip, shape_nodal);
          double val = 0.;
          val +=  w[vnums[0]*COMP+ll] * shape_nodal[0];
          val +=  w[vnums[1]*COMP+ll] * shape_nodal[1];
          val +=  w[vnums[2]*COMP+ll] * shape_nodal[2];
          ww[pp+ndof_w*ll] += ed.qw[j] * val * ed.phi[pp+ndof_w*j];
        }

    mygemm('n', 'n', ndof_w, COMP, ndof_w, 1., &ed.inv_mass[0], &ww[0], 0., vecW);
    
    if(Model::Diffusion) {
      for (int pp = 0; pp < ndof_w; ++pp)
        for (int ll = 0; ll < COMP; ++ll)
          for (int j = 0; j < ed.nip; ++j){
            fel_nodal.CalcMappedDShape(mir_vol[j], dshape_nodal);

            double val1 = 0.;	   
            val1 +=  w[vnums[0]*COMP+ll] * dshape_nodal(0,0);
            val1 +=  w[vnums[1]*COMP+ll] * dshape_nodal(1,0);
            val1 +=  w[vnums[2]*COMP+ll] * dshape_nodal(2,0);
            qq[pp+ndof_w*(0+D*ll)] += ed.qw[j] * val1 * ed.phi[pp+ndof_w*j];
            double val2 = 0.;
            val2 +=  w[vnums[0]*COMP+ll] * dshape_nodal(0,1);
            val2 +=  w[vnums[1]*COMP+ll] * dshape_nodal(1,1);
            val2 +=  w[vnums[2]*COMP+ll] * dshape_nodal(2,1);
            qq[pp+ndof_w*(1+D*ll)] += ed.qw[j] * val2 * ed.phi[pp+ndof_w*j];
          }

      mygemm('n', 'n', ndof_w, COMP * D, ndof_w, 1., &ed.inv_mass[0], &qq[0], 0., vecQ);
    }     
  }
  
  for (int i = 0; i < nf; ++i) {
    if (!fadata[i]) continue;

    FacetData<D,COMP> &fd = *fadata[i];

    int ndof_l = fd.ndof_l;
    int ndof_w1 = fd.ndof_w1;
    int ndof_w2 = fd.ndof_w2;
    int nip    = fd.nip;

    if (ndof_l == 0)
      continue;

    vector<double> rhs(ndof_l*COMP, 0.);
    
    for (int j = 0; j < nip; ++j) {

      double w[COMP] = {0.};
      
      for (int pp = 0; pp < ndof_w1; ++pp) {
        double val = fd.phi1[ndof_w1*j+pp];
        for (int ll = 0; ll < COMP; ++ll)
          w[ll] += 0.5 * val * sol.vecW[COMP*fd.offset_w1+pp+ndof_w1*ll];
      }
      for (int pp = 0; pp < ndof_w2; ++pp) {
        double val = fd.phi2[ndof_w2*j+pp];
        for (int ll = 0; ll < COMP; ++ll)
          w[ll] += 0.5 * val * sol.vecW[COMP*fd.offset_w2+pp+ndof_w2*ll];
      }

      for (int ll = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof_l; ++pp)
          rhs[pp+ndof_l*ll] += fd.qw[j] * w[ll] * fd.mu[pp+ndof_l*j];
    }

    mygemm('n', 'n', ndof_l, COMP, ndof_l, 1., &fd.inv_mass[0], &rhs[0], 0., &sol.vecL[COMP * fd.offset_l]);
  }

  cout << "Bamg solution loaded from " << filename << endl;
  
  return true;
}
