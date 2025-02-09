#include "../../helper_blas.h"

template <int D, int COMP, class Model>
  void UnifyingFramework<D, COMP, Model>
  ::AssembleHybridAdjointRhs(const int i, vector<double> & matR, vector<double> & matS, vector<double> & vecF, vector<double> & vecG) {

  ElementData<D, COMP> & ed = *eldata[i];
  
  int ndof_q  = ed.ndof_q;
  int ndof_w  = ed.ndof_w;
  int ndof_lt = ed.ndof_lt;

  mygemv('t', COMP * ndof_w, COMP * ndof_lt, 1., &matS[COMP * ndof_w], &vecG[0], 0., &insert[0]);
  if (Model::Diffusion)
    mygemv('t', COMP * ndof_q, COMP * ndof_lt, 1., &matR[COMP * ndof_q], &vecF[0], 1., &insert[0]);

  int offset = 0;
  // Facet contributions
  for (int ff = 0; ff < ed.nf; ++ff) {
    FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];

    if (fd.ndof_l == 0) continue;
    
    int ndof_l    = fd.ndof_l;
    int upperleft = fd.offset_l;

    // Apply scaling
    for (int ll = 0, index = offset; ll < COMP; ++ll)
      for (int pp = 0; pp < ndof_l; ++pp, ++index)
        insert[index] *= residual_scale[ll];
    
    petsc_add_adj_vector(COMP * ndof_l, COMP * upperleft, &insert[offset]);

    offset += COMP * ndof_l;    
  }
}
