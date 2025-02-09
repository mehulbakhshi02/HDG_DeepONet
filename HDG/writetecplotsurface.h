extern"C" {
  void table_delaunay_wrapper_(int*, double*, int*, int*);
}

template<int D, int COMP, class Model>
  void UnifyingFramework<D, COMP, Model>
::SaveTecplotSurface(const string & filename, int order, const Solution & sol, const Space & sp, LocalHeap & lh) const {

  string fname_dat(filename);
  fname_dat.append(".dat");
  fname_dat.insert (0,"surface-");
  fstream output;
  output.open(fname_dat.c_str(),ios::out);

  bool adjoint  = assemblyMode == AM_AssemblyAdjoint || assemblyMode == AM_BackSolveAdjoint;  

  char res_out[160];
  output << "TITLE = DG2D" << endl;
  output << "VARIABLES = \"X\" \"Y\" ";
  if (D == 3)
    output << "\"Z\" ";
  for (int ll = 0; ll < COMP; ++ll) {
    sprintf(res_out, "\"W%i\" ", ll+1);
    output << res_out;
  }
  string dervar;
  if (!adjoint)
    for (int i = 0; i < Model::NumDerVar; ++i) {
      Model::GetDerVarName(i, dervar);
      output << "\""<<dervar << "\"" << " ";
    }
  output << endl;
  // Calculate a triangulation of the unit triangle

  int nlt = 2 * (order+1);

  int npt = 0.5 * (nlt + 2) * (nlt + 1);

  vector<double> xllt(2*npt);

  int l = 0;
  for (int j = 0; j <= nlt; j++) {
    double rl1 = 1.*j / (1. * nlt);
    for (int k = 0; k <= nlt; k++) {
      double rl2 = 1. * k / (1. * nlt);
      if ( (rl2 + rl1) < 1. + 1e-9) {	
      	double rl3 = 1. - rl2 - rl1;      
      	xllt[2*l] = rl2;
      	xllt[2*l+1] = rl3;  
      	l++;
      }
    }
  }

  vector<int> nlct(9*npt);
  int ntr = 0;

  table_delaunay_wrapper_(&npt, &xllt[0], &ntr, &nlct[0]);

  // Calculate a triangulation of the unit square

  int nls = 2 * (order+1);

  int nps = (nls + 1) * (nls + 1);

  vector<double> xlls(2*nps);

  l = 0;
  for (int j = 0; j <= nls; j++) {
    double rl1 = 1.*j / (1. * nls);
    for (int k = 0; k <= nls; k++) {
      double rl2 = 1. * k / (1. * nls);    
      xlls[2*l] = rl1;
      xlls[2*l+1] = rl2;  
      l++;
    }
  }

  int nsq = nls * nls;
  vector<int> nlcs(4 * nsq);

  int tri = 0;
  l = 1;
  for (int j = 0; j < nls; ++j) {
    for (int k = 0; k < nls; ++k) {

      nlcs[4 * tri] = l;
      nlcs[4 * tri + 1] = l+nls+1;
      nlcs[4 * tri + 2] = l+nls+2;
      nlcs[4 * tri + 3] = l+1;

      tri++;

      /*      nlcs[3 * tri] = l;
      nlcs[3 * tri + 1] = l+nls+1;
      nlcs[3 * tri + 2] = l+nls+2;

      tri++;*/

      l++;
    }
    l++;
  }

  // Now get the shape functions evaluated at these help nodes

  // Loop over the boundary faces and write data out. Each triangle is a zone
  for (int i = 0; i < nf; i++) {

    HeapReset hr(lh);
    if (!fadata[i]) continue;
    FacetData<D,COMP> & fd = *fadata[i];

    if (fd.bcnr == -2) continue;
    if (fd.ndof_w2 != 0) continue;
    if (fd.bcnr == 0) continue;
    int np = 0;
    double * xll;

    if (fd.type == ET_TRIG) {
      np = npt;
      xll = &xllt[0];
    } else {
      np = nps;
      xll = &xlls[0];
    }
   
    int elnr = fd.elnr1;
    ElementData<D, COMP> & elidata = *eldata[elnr];

    const ScalarFiniteElement<D> &      fel_w = dynamic_cast<const ScalarFiniteElement<D>&> (sp.fes_w -> GetFE(elnr, lh));
    const FacetVolumeFiniteElement<D> & fel_l = dynamic_cast<const FacetVolumeFiniteElement<D>&> (sp.fes_l->GetFE(elnr, lh));    

    ElementTransformation & eltrans = ma->GetTrafo(elnr, lh);

    // On which local face are we?
    int ff = 0;
    for (int j = 0; j < elidata.nf; ++j)
      if (elidata.faces[j] == i)
        ff = j;

    Array<int> vnums;
    vnums = ma->GetElVertices(ElementId(VOL,elnr));

    ELEMENT_TYPE eltype = fel_w.ElementType();

    // We have to check whether the element is degenerated
    bool admissible = true;
    int nv = ElementTopology::GetNVertices(eltype);
    for (int l = 0; l < nv && admissible; ++l) {
      Vec<D> p1 = ma->GetPoint<D>(vnums[l]);
      for (int k = l+1; k < nv && admissible; ++k) {
        Vec<D> p2 = ma->GetPoint<D>(vnums[k]);
        if (L2Norm(p1 - p2) < 1.e-15)
          admissible = false;                 
      }
    }    

    if (!admissible) continue;    

    
    Facet2ElementTrafo transform(eltype, vnums);
    const NORMAL * normals = ElementTopology::GetNormals(eltype);

    Vec<D> normal_ref;
    for (int dd = 0; dd < D; ++dd)
      normal_ref(dd) = normals[ff][dd];

    for (int j = 0; j < np; j++) {

      // x-coordinate: xll[2*j], y-coordinate: xll[2*j + 1]

      IntegrationPoint xi_ref(xll[2*j], xll[2*j+1]);
      IntegrationPoint xi = transform(ff, xi_ref);

      MappedIntegrationPoint<D,D> sip (xi, eltrans);

      if (abs(sip.GetPoint()[0]) > tecplot_x || abs(sip.GetPoint()[1]) > tecplot_y || abs(sip.GetPoint()[2]) > tecplot_z) {
        admissible = false;
        break;
      }
    }

    if (!admissible) continue;

    int ndof_vol_max = get_ndof(max_order, eltype);
    int ndof_facet_max = get_ndof(max_order, fd.type);
    
    FlatVector<> shape(ndof_vol_max, lh);
    FlatVector<> shapeL(ndof_facet_max, lh);    

    stringstream oss(" ");
    Model::GetBCName(fd.bcnr, oss);

    char res_out[160];
    strcpy (res_out,"Zone T=\"");
    strcat (res_out,oss.str().c_str());

    char res_temp[160];
    if (fd.type == ET_TRIG)
      sprintf(res_temp, "\" N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", np, ntr);
    else
      sprintf(res_temp, "\" N=%i, E=%i, F=FEPOINT, ET=QUADRILATERAL", np, nsq);
    strcat (res_out,res_temp);    

    output << res_out << endl;

    for (int j = 0; j < np; j++) {

      // x-coordinate: xll[2*j], y-coordinate: xll[2*j + 1]

      IntegrationPoint xi_ref(xll[2*j], xll[2*j+1]);
      IntegrationPoint xi = transform(ff, xi_ref);

      MappedIntegrationPoint<D,D> sip (xi, eltrans);

      fel_w.CalcShape (xi, shape);
      if (fd.ndof_l != 0)
        fel_l.Facet(ff).CalcShape(xi, shapeL);

      Mat<D> jac     = sip.GetJacobian();
      Mat<D> inv_jac = sip.GetJacobianInverse();
      double det     = sip.GetJacobiDet();
      
      Vec<D> normal = det * Trans (inv_jac) * normal_ref;       
      double len    = L2Norm (normal);
      normal       /= len;

      SpatialParams<D> sparam;
      for (int dd = 0; dd < D; ++dd)
        sparam.pos(dd) = sip.GetPoint()[dd];

      Vec<COMP> w, w_bc;
      w = 0.;

      if (fd.ndof_l == 0)
        for (int ll = 0; ll < COMP; ++ll)
          for (int pp = 0; pp < elidata.ndof_w; ++pp) {
            w(ll) += shape(pp) * sol.vecW[COMP*elidata.offset_w+pp+ll*elidata.ndof_w];
          }
      else
        for (int ll = 0; ll < COMP; ++ll)
          for (int pp = 0; pp < fd.ndof_l; ++pp) {
            w(ll) += shapeL(pp) * sol.vecL[COMP*fd.offset_l+pp+ll*fd.ndof_l];
          }        
        
      if (fd.ndof_l == 0) {
        Model::EvalBdryState(fd.bcnr, w, sparam, w_bc);
        w = w_bc;
      }

      for (int dd = 0; dd < D; ++dd) {
        sprintf(res_out, "%10.10f ", sparam.pos(dd));
        output << res_out;
      }
      for (int ll = 0; ll < COMP; ++ll) {
        sprintf(res_out, "%10.10f ", w(ll));
        output << res_out;
      }

      if (!adjoint) {
        Vec<Model::NumDerVar> dervar;
        Model::EvalDerVar(w, sparam, dervar);

        for (int dv = 0; dv < Model::NumDerVar; ++dv) {
          sprintf(res_out, "%10.10f ", dervar(dv));
          output << res_out;
        }          
      }

      output << endl;
    }

    if (fd.type == ET_TRIG) {
      for (int j = 0; j < ntr; j++) {
        sprintf(res_out, "%i %i %i", nlct[3*j], nlct[3*j+1], nlct[3*j+2]);
        output << res_out << endl;
      }
    } else {
      for (int j = 0; j < nsq; j++) {
        sprintf(res_out, "%i %i %i %i", nlcs[4*j], nlcs[4*j+1], nlcs[4*j+2], nlcs[4*j+3]);
        output << res_out << endl;
      }
    }    

  }

  output.close();

  cout << "Tecplot surface solution written to " << fname_dat << endl;
}
