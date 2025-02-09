template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::SaveSolutionBamg(const string & filename, const Solution & sol, LocalHeap & lh) const {

  if (D != 2)
    cout << "[SaveSolutionBamg] This routine is only available for 2d problems" << endl;

  if (nf_max > 3)
    cout << "[SaveSolutionBamg] This routine is only available for triangular meshes" << endl;

  int size = Model::Diffusion ? (D+1) * COMP : COMP;
  if (AnisotropyData::hp) size++;

  ofstream file(filename.c_str(), ios::out);
  file.precision(16);
  file.setf(ios::scientific, ios::floatfield);

  int nv = ma->GetNV();

  file << "2 " << size << " " << nv << " 2" << endl;

  Array<int> elns, vnums;
  
  for (int i = 0; i < nv; ++i) {

    HeapReset hr(lh);
    
    ma->GetVertexElements(i, elns);

    double w[COMP] = {0.0};
    double q1[COMP] = {0.0};
    double q2[COMP] = {0.0};

    for (int k = 0; k < elns.Size(); ++k){
      int elnr = elns[k];
      ElementData<D,COMP> & ed = *eldata[elnr];

      vnums = ma->GetElVertices(ElementId(VOL,elnr));
      const ScalarFiniteElement<D> & fel_w = dynamic_cast<const ScalarFiniteElement<D>&> (fspace.fes_w->GetFE(elnr, lh));      

      ElementTransformation & eltrans = ma->GetTrafo(i, lh);

      double x1 = ed.nodes[0];
      double y1 = ed.nodes[1];
      double x2 = ed.nodes[2];
      double y2 = ed.nodes[3];
      double x3 = ed.nodes[4];
      double y3 = ed.nodes[5];
      Vec<D> coord = ma->GetPoint<D>(i);
      double xx = coord(0);
      double yy = coord(1);  
      double den = (x2-x1) * (y3-y1) - (y2-y1) * (x3-x1); 
      double xi = (xx-x1) * (y3-y1) + (yy-y1) * (x1-x3);
      xi /= den;
      double eta = (xx-x1) * (y1-y2) + (yy-y1) * (x2-x1);
      eta /= den;
      IntegrationPoint ip;
      ip(0) = xi;
      ip(1) = eta; 
      int ndof = ed.ndof_w;

      FlatVector<> shape(ndof, lh);
      fel_w.CalcShape(ip, shape);

      for (int pp = 0; pp < ndof; ++pp)
        for (int ll = 0; ll < COMP; ++ll)
	  w[ll] += shape[pp] * sol.vecW[ed.offset_w*COMP+pp+ndof*ll];

      if (Model::Diffusion) {
        for (int pp = 0; pp < ndof; ++pp) {
	  for (int ll = 0; ll < COMP; ++ll) {
	    q1[ll] += shape(pp) * sol.vecQ[COMP*ed.offset_q+pp+ndof*(0+D*ll)];
	    q2[ll] += shape(pp) * sol.vecQ[COMP*ed.offset_q+pp+ndof*(1+D*ll)];
	  }
	}
      }      
    }

    for (int ll = 0; ll < COMP; ++ll)
      file << setw(24) << w[ll] / double(elns.Size());

    if (Model::Diffusion) {
      for (int ll = 0; ll < COMP; ++ll)
        file << setw(24) << q1[ll] / double(elns.Size());
      for (int ll = 0; ll < COMP; ++ll)
        file << setw(24) << q2[ll] / double(elns.Size());
    }
    file << endl;
  }

  file.close();

  cout << "Bamg solution saved to " << filename << endl;
}
