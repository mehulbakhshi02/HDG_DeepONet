#include "../../helper_blas.h"

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::AssembleHybridSystemLocal(const int i, vector<double> & matQ, vector<double> & matW,
			                   vector<double> & matL, vector<double> & matM) {

  ElementData<D, COMP> & ed = *eldata[i];

  int ndof_q  = ed.ndof_q;
  int ndof_w  = ed.ndof_w;

  int offset_lw = 0;
  int offset_lq = 0;

  for (int ff = 0; ff < ed.nf; ++ff) {
    FacetData<D, COMP> & fd = *fadata[ed.faces[ff]];

    int ndof_l = fd.ndof_l;
    int offset_l = fd.offset_l;

    // No boundary faces
    if (ndof_l == 0) continue;

    if (assemblyMode != AM_AssemblyAdjoint) {
      // M * W0
      mygemv('n', COMP * ndof_l, COMP * ndof_w, -1., &matM[offset_lw], &matW[0], 0., &insert[0]);
      // L * Q0
      if (Model::Diffusion)
        mygemv('n', COMP * ndof_l, COMP * ndof_q, -1., &matL[offset_lq], &matQ[0], 1., &insert[0]);

      // Apply scaling
      for (int ll = 0, index = 0; ll < COMP; ++ll)
        for (int pp = 0; pp < ndof_l; ++pp, ++index)
          insert[index] *= residual_scale[ll];

      petsc_add_vector(COMP * ndof_l, COMP * offset_l, &insert[0]);
    }
    
    int offset_lw2 = COMP * ndof_w;
    int offset_lq2 = COMP * ndof_q;
    
    for (int edge2 = 0; edge2 < ed.nf; ++edge2) {
      int edge_number2 = ed.faces[edge2];
      FacetData<D,COMP> & fd2 = *fadata[edge_number2];
      
      int offset_l2 = fd2.offset_l;
      int ndof_l2 = fd2.ndof_l;

      // No boundary faces
      if (ndof_l2 == 0) continue;

      // M * W'
      mygemm('t', 't', COMP * ndof_l2, COMP * ndof_l, COMP * ndof_w, 1., &matW[offset_lw2], &matM[offset_lw], 0., &insert[0]);
      // L * Q'
      if (Model::Diffusion)
        mygemm('t', 't', COMP * ndof_l2, COMP * ndof_l, COMP * ndof_q, 1., &matQ[offset_lq2], &matL[offset_lq], 1., &insert[0]);

      // Apply scaling
      int nn;
      for (int ll = 0, index = 0; ll < COMP; ++ll) {
        if (assemblyMode == AM_AssemblyPrimal) nn = ll;
        for (int pp = 0; pp < ndof_l; ++pp)
          for (int mm = 0; mm < COMP; ++mm) {
            if (assemblyMode == AM_AssemblyAdjoint) nn = mm;
            for (int qq = 0; qq < ndof_l2; ++qq, ++index)
              insert[index] *= residual_scale[nn];
          }
      }

      petsc_add_submatrix(COMP * ndof_l, COMP * ndof_l2, COMP * offset_l, COMP * offset_l2, &insert[0]);
      
      offset_lw2 += COMP * ndof_l2 * COMP * ndof_w;   
      offset_lq2 += COMP * ndof_l2 * COMP * ndof_q;
    }

    offset_lw += COMP * ndof_l * COMP * ndof_w;
    offset_lq += COMP * ndof_l * COMP * ndof_q;
  }
}
