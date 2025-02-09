#ifndef COMPUTEWALLDISTANCE_H
#define COMPUTEWALLDISTANCE_H

template <int D>
double ComputeDistanceToSegment(Vec<D> p0, Vec<D> p1, Vec<D> p) {
  Vec<D> v(p1-p0);
  Vec<D> w(p-p0);

  double c1 = InnerProduct(v, w);
  double c2 = InnerProduct(v, v);

  if (c1 <= 0.)
    return L2Norm(p-p0);
  else if (c2 <= c1)
    return L2Norm(p-p1);

  double b = c1 / c2;
  Vec<D> pb(p0+b*v);
  return L2Norm(p-pb);
}

// Some models require information about the distance to certain boundaries (e.g. wall distance for Spalart Almaras)
// Right now, this is computed via a brute-force approach. Using search-tree structures, this could be accelerated
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ComputeWallDistance() {

  // Collect boundary facets with correct boundary conditions
  vector<int> bdry_facets;
  
  for (int i = 0; i < nf; ++i) {
    if (!fadata[i]) continue;
    FacetData<D, COMP> & fd = *fadata[i];

    if (fd.ndof_l != 0) continue;
    if (fd.bcnr   != 1) continue;
    
    bdry_facets.push_back(i);
  }

  int bdry_n = bdry_facets.size();

  double wd_min = std::numeric_limits<double>::max();
  
  // Iterate over all elements (and quadrature points) and compute wall distances
  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];

    for (int j = 0; j < ed.nip; ++j) {
      const double * qp = &ed.qp[D * j];

      double wd = std::numeric_limits<double>::max();

      // Loop over boundary facets
      for (int ii = 0; ii < bdry_n; ++ii) {
        FacetData<D, COMP> & fd = *fadata[bdry_facets[ii]];

        for (int jj = 0; jj < fd.nip; ++jj) {
          const double * qp_bdry = &fd.qp[D * jj];

          double wd_loc = 0.;

          if (D == 2) {
            Vec<2> p(qp[0], qp[1]);
            
            if (jj == 0) {

              Vec<2> p0(qp_bdry[0], qp_bdry[1]);
              Vec<2> p1(fd.nodes[2], fd.nodes[3]);

              wd_loc = ComputeDistanceToSegment(p0, p1, p);
              wd = min(wd, wd_loc);
            } else  if (jj == fd.nip - 1) {

              Vec<2> p0(qp_bdry[0], qp_bdry[1]);
              Vec<2> p1(fd.nodes[0], fd.nodes[1]);

              wd_loc = ComputeDistanceToSegment(p0, p1, p);              
              wd = min(wd, wd_loc);

              continue;
            }

            Vec<2> p0(qp_bdry[0], qp_bdry[1]);
            Vec<2> p1(qp_bdry[2], qp_bdry[3]);

            wd_loc = ComputeDistanceToSegment(p0, p1, p);
            wd = min(wd, wd_loc);
          }
        }
      }
      ed.wd[j] = wd;

      wd_min = min(wd_min, ed.wd[j]);
    }
  }

  cout << "wd_min: " << wd_min << endl;

  // HACK
  if (assemblyMode == AM_AssemblyAdjoint)
    return;
  
  // Just for visualization purposes
  vector<double> rhs(ndof_w_max), sol(ndof_w_max);
  // VVector<double> & vecwd = dynamic_cast<VVector<double> &> (gf_wd->GetVector());
  FlatVector<Vec<1>> vecwd = gf_wd->GetVector().FV<Vec<1>>();

  vecwd = 0.;
  
  int nhm1 = 0;
  for (int i = 0; i < ne; ++i) {
    ElementData<D, COMP> & ed = *eldata[i];

    int ndof_w = ed.ndof_w;
    int nip    = ed.nip;

    fill(&rhs[0], &rhs[0] + ndof_w, 0.0);

    for (int j = 0; j < nip; ++j) {
      const double qw    = ed.qw[j];
      const double wd    = ed.wd[j];
      const double * phi = &ed.phi[j * ndof_w];

      for (int pp = 0; pp < ndof_w; ++pp)
        rhs[pp] += qw * phi[pp] * wd;
    }

    // Compute expansion coefficients
    mygemv('n', ndof_w, ndof_w, 1., &ed.inv_mass[0], &rhs[0], 0., &sol[0]);

    int order_loc = ed.order;
    
    int * conv_loc;
    if (ed.type == ET_TRIG)
      conv_loc = &ndof_converter_tri[order_loc][0];
    else if (ed.type == ET_QUAD)
      conv_loc = &ndof_converter_quad[order_loc][0];
    else if (ed.type == ET_TET)
      conv_loc = &ndof_converter_tet[order_loc][0];
    else if (ed.type == ET_HEX)
      conv_loc = &ndof_converter_hex[order_loc][0];

    for (int pp = 0; pp < ndof_w; ++pp) {
      int new_pp = conv_loc[pp];
      int index = (pp == 0) ? i : ne + nhm1 + new_pp - 1;
      vecwd(index) = sol[pp];
    }

    nhm1 += get_ndof(max_order, ed.type) - 1;
  }
}

#endif
