extern"C" {
  void table_delaunay_wrapper_(int*, double*, int*, int*);
  void table_tet_wrapper_(int*, double*, int*, int*);
}

// So far only 2d!

template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::SaveTecplotCellWise(int order, const string & var, const MeshMetric<D> & metric, const Space & sp, LocalHeap & lh) const {


  stringstream oss(" ");
  oss<< var << "-" << ne << "-" << order;
  string filename(oss.str());

  string fname_dat(filename);
  fname_dat.append(".dat");
  
  fstream output;
  output.open(fname_dat.c_str(),ios::out);

  bool adjoint  = assemblyMode == AM_AssemblyAdjoint || assemblyMode == AM_BackSolveAdjoint;    

  char res_out[160];
  output << "TITLE = DG2D" << endl;
  output << "VARIABLES = \"X\" \"Y\" ";
  if (D == 3)
    output << "\"Z\" ";
  output<<"\"" << "Order"<<"\""<<" \"log-Size\" "<<"\"Aspect ratio\"";
  if(D == 3)
    output<<" \"Aspect ratio 2\"";
  output << endl;

  // VVector<double> & vecwd = dynamic_cast<VVector<double> &> (gf_wd->GetVector());  

  // Calculate a triangulation of the unit triangle

  int nlt = 1.0;

  int npt = 0.5 * (nlt + 2) * (nlt + 1);

  vector<double> xllt(2*npt);

  int ll = 0;
  for (int j = 0; j <= nlt; j++) {
    double rl1 = 1.*j / (1. * nlt);
    for (int k = 0; k <= nlt; k++) {
      double rl2 = 1. * k / (1. * nlt);
      if ( (rl2 + rl1) < 1. + 1e-9) {	
      	double rl3 = 1. - rl2 - rl1;      
      	xllt[2*ll] = rl2;
      	xllt[2*ll+1] = rl3;  
      	ll++;
      }
    }
  }

  vector<int> nlct(9*npt);
  int ntr = 0;

  table_delaunay_wrapper_(&npt, &xllt[0], &ntr, &nlct[0]);

  // Calculate a triangulation of the unit square

  int nls = 1.0;

  int nps = (nls + 1) * (nls + 1);

  vector<double> xlls(2*nps);

  ll = 0;
  for (int j = 0; j <= nls; j++) {
    double rl1 = 1.*j / (1. * nls);
    for (int k = 0; k <= nls; k++) {
      double rl2 = 1. * k / (1. * nls);    
      xlls[2*ll] = rl1;
      xlls[2*ll+1] = rl2;  
      ll++;
    }
  }

  int nsq = nls * nls;
  vector<int> nlcs(4 * nsq);

  int tri = 0;
  int l = 1;
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

  // Calculate a triangulation of the unit cube

  int nlc = 1.0;

  int npc = (nls + 1) * (nls + 1) * (nls + 1);

  vector<double> xllc(3*npc);

  ll = 0;
  for (int j = 0; j <= nlc; j++) {
    double rl1 = 1.*j / (1. * nlc);
    for (int k = 0; k <= nlc; k++) {
      double rl2 = 1. * k / (1. * nlc);
      for (int l = 0; l <= nlc; l++) {
        double rl3 = 1. * l / (1. * nlc);
        xllc[3*ll]   = rl1;
        xllc[3*ll+1] = rl2;
        xllc[3*ll+2] = rl3;
        ll++;
      }
    }
  }

  int ncu = nlc * nlc * nlc;
  vector<int> nlcc(8 * ncu);

  int cubes = 0;

  for (int j = 0; j < nlc; ++j) {
    for (int k = 0; k < nlc; ++k) {
      for (int l = 0; l < nlc; ++l) {

        nlcc[8 * cubes]     = 1 + l + (nlc+1) * (k + (nlc+1) * j);
        nlcc[8 * cubes + 1] = 1 + l + (nlc+1) * (k + (nlc+1) * (j+1));
        nlcc[8 * cubes + 2] = 1 + l + (nlc+1) * ((k+1) + (nlc+1) * (j+1));
        nlcc[8 * cubes + 3] = 1 + l + (nlc+1) * ((k+1) + (nlc+1) * j);
        nlcc[8 * cubes + 4] = 1 + l+1 + (nlc+1) * (k + (nlc+1) * j);
        nlcc[8 * cubes + 5] = 1 + l+1 + (nlc+1) * (k + (nlc+1) * (j+1));
        nlcc[8 * cubes + 6] = 1 + l+1 + (nlc+1) * ((k+1) + (nlc+1) * (j+1));
        nlcc[8 * cubes + 7] = 1 + l+1 + (nlc+1) * ((k+1) + (nlc+1) * j);

        cubes++;
      }
    }
  }  


  int nlte = 1.0;

  int npte = (1.0/6.0) * (nlte + 3) * (nlte + 2) * (nlte + 1);
  vector<double> xllte;
  vector<int> nlcte;
  int ntet = 0;
  if(D==3)
  {
    xllte.resize(3*npte);
    ll = 0;
    for (int j = 0; j <= nlte; j++) 
    {
      double rl1 = 1.*j / (1. * nlte);
      for (int k = 0; k <= nlte; k++) 
      {
        double rl2 = 1. * k / (1. * nlte);
        for (int l = 0; l <= nlte; l++)
        {
          double rl3 = 1. * l / (1. * nlte);
          if ( (rl2 + rl1 + rl3) < 1. + 1e-9)
          { 
            xllte[3*ll] = rl1;
            xllte[3*ll+1] = rl2;  
            xllte[3*ll+2] = rl3;
            ll++;
          }     
        }
      }
    }

    nlcte.resize(16*npte);
    table_tet_wrapper_(&npte, &xllte[0], &ntet, &nlcte[0]);
  }

  // Loop over the boundary faces and write data out. Each triangle is a zone
  int nhm1 = 0;
  for (int i = 0; i < ne; i++) {
    HeapReset hr(lh);
    ElementData<D, COMP> & elidata = *eldata[i];

    const ScalarFiniteElement<D> & fel_w = dynamic_cast<const ScalarFiniteElement<D>&> (sp.fes_w -> GetFE(i, lh));

    ELEMENT_TYPE eltype = fel_w.ElementType();

    ElementTransformation & eltrans = ma->GetTrafo(i, lh);

    Array<int> vnums;
    vnums = ma->GetElVertices(ElementId(VOL,i));

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

    int np = 0;
    double * xll;

    if (eltype == ET_TRIG) {
      np = npt;
      xll = &xllt[0];
    } else if (eltype == ET_QUAD) {
      np = nps;
      xll = &xlls[0];
    } else if (eltype == ET_HEX) {
      np = npc;
      xll = &xllc[0];
    } else if (eltype == ET_TET) {
      np = npte;
      xll = &xllte[0];
    }

    for (int j = 0; j < np; j++) {

      // x-coordinate: xll[2*j], y-coordinate: xll[2*j + 1]

      IntegrationPoint xi;
      for (int dd = 0; dd < D; ++dd)
        xi(dd) = xll[D*j+dd];
      //      cout << xi << endl;

      MappedIntegrationPoint<D,D> sip (xi, eltrans);

      if (abs(sip.GetPoint()[0]) > tecplot_x || abs(sip.GetPoint()[1]) > tecplot_y) {
        admissible = false;
        break;
      }
    }

    if (!admissible) continue;

    int order_wi = elidata.order;
    
    int * conv_loc;
    if (eltype == ET_TRIG)
      conv_loc = &ndof_converter_tri[order_wi][0];
    else if (eltype == ET_QUAD)
      conv_loc = &ndof_converter_quad[order_wi][0];
    else if (eltype == ET_TET)
      conv_loc = &ndof_converter_tet[order_wi][0];
    else if (eltype == ET_HEX)
      conv_loc = &ndof_converter_hex[order_wi][0];     

    int ndof_vol_max = get_ndof(max_order, eltype);
    


    char res_out[160];
    if (eltype == ET_TRIG)
      sprintf(res_out, "Zone T=\"INTERIOR\" N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", np, ntr);
    else if (eltype == ET_QUAD)
      sprintf(res_out, "Zone T=\"INTERIOR\" N=%i, E=%i, F=FEPOINT, ET=QUADRILATERAL", np, nsq);
    else if (eltype == ET_HEX)
      sprintf(res_out, "Zone T=\"INTERIOR\" N=%i, E=%i, F=FEPOINT, ET=BRICK", np, cubes);
    else if (eltype == ET_TET)
      sprintf(res_out, "Zone T=\"INTERIOR\" N=%i, E=%i, F=FEPOINT, ET=TETRAHEDRON", np, ntet);
    output << res_out << endl;

    for (int j = 0; j < np; j++) {


      // x-coordinate: xll[2*j], y-coordinate: xll[2*j + 1]

      IntegrationPoint xi; //(xll[2*j], xll[2*j+1]);
      for (int dd = 0; dd < D; ++dd)
        xi(dd) = xll[D*j+dd];

      MappedIntegrationPoint<D,D> sip (xi, eltrans);

      FlatVector<> shape(ndof_vol_max, lh);
      fel_w.CalcShape (xi, shape);

      SpatialParams<D> sparam;
      for (int dd = 0; dd < D; ++dd)
        sparam.pos(dd) = sip.GetPoint()[dd];

      Vec<COMP> w = 0.;

      for (int dd = 0; dd < D; ++dd) {
        sprintf(res_out, "%10.10f ", sparam.pos(dd));
        output << res_out;
      }
      for (int ll = 0; ll < 1; ++ll) {
        sprintf(res_out, "%10.10f ", (double)(order_array[i]));
        output << res_out;
      }
      if(D==2)
      {
        double imp_Area;
        vector<double> beta(D-1,0.0);
        vector<double> q_met(D*D,0.0);

        metric.GetPrincipal(i, imp_Area, beta, q_met);
        sprintf(res_out, "%10.10f ", log(imp_Area));
        output << res_out;
        for(int dd=0;dd<D-1;++dd)
        {
          sprintf(res_out, "%10.10f ", beta[dd]);
          output << res_out;        
        }

      }

      output << endl;
    }

    if (eltype == ET_TRIG) {
      for (int j = 0; j < ntr; j++) {
        sprintf(res_out, "%i %i %i", nlct[3*j], nlct[3*j+1], nlct[3*j+2]);
        output << res_out << endl;
      }
    } else if (eltype == ET_QUAD) {
      for (int j = 0; j < nsq; j++) {
        sprintf(res_out, "%i %i %i %i", nlcs[4*j], nlcs[4*j+1], nlcs[4*j+2], nlcs[4*j+3]);
        output << res_out << endl;
      }
    } else if (eltype == ET_HEX) {
      for (int j = 0; j < cubes; j++) {
        sprintf(res_out, "%i %i %i %i %i %i %i %i", nlcc[8*j], nlcc[8*j+1], nlcc[8*j+2], nlcc[8*j+3], nlcc[8*j+4], nlcc[8*j+5],  nlcc[8*j+6], nlcc[8*j+7]);
        output << res_out << endl;
      }      
    }
    else if (eltype == ET_TET) {
      for (int j = 0; j < ntet; j++) {
        sprintf(res_out, "%i %i %i %i", nlcte[4*j+0], nlcte[4*j+1], nlcte[4*j+2], nlcte[4*j+3]);
        output << res_out << endl;
      }      
    }

    // nhm1 += get_ndof(max_order, elidata.type) - 1;

  }

  output.close();

  cout << "Tecplot solution written to " << fname_dat << endl;  
}
