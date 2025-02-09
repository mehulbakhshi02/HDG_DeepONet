// **********************************************************************
// Transfer the degrees of freedom to netgen arrays (netgen uses another
// ordering)
// **********************************************************************

template<int D, int COMP, class Model>
  void UnifyingFramework<D, COMP, Model>
  ::TransferCoefficients(Solution & sol, GFunction & gfunc) {
    // if(COMP==1)
    // {  
    //   VVector<double> & vecw = dynamic_cast<VVector<double> &> (gfunc.gf_w -> GetVector());
    //   vecw = 0.0;

    //   const int size = (Model::Diffusion || Model::Source) ? COMP * D : COMP;
    //   if(size == 1)
    //   {      
    //     VVector<double> & vecq = dynamic_cast<VVector<double> &> (gfunc.gf_q -> GetVector());
    //     vecq = 0.0;

    //     int nhm1 = 0; 

    //     for (int i = 0; i < ne; ++i) {
    //       ElementData<D, COMP> & ed = *eldata[i];
          
    //       int ndof_w    = ed.ndof_w;
    //       int order_loc = ed.order;
          
    //       int * conv_loc;
    //       if (ed.type == ET_TRIG)
    //         conv_loc = &ndof_converter_tri[order_loc][0];
    //       else if (ed.type == ET_QUAD)
    //         conv_loc = &ndof_converter_quad[order_loc][0];
    //       else if (ed.type == ET_TET)
    //         conv_loc = &ndof_converter_tet[order_loc][0];
    //       else if (ed.type == ET_HEX)
    //         conv_loc = &ndof_converter_hex[order_loc][0];

    //       for (int pp = 0; pp < ndof_w; ++pp) {
    //         int new_pp = conv_loc[pp];
    //         int index = (pp == 0) ? i : ne + nhm1 + new_pp - 1;
    //         for (int ll = 0; ll < COMP; ++ll)
    //           vecw(index) = sol.vecW[COMP*ed.offset_w+pp+ll*ndof_w];
    //       }

    //       if (Model::Diffusion || Model::Source) 
    //         for (int pp = 0; pp < ndof_w; ++pp) {
    //           int new_pp = conv_loc[pp];
    //           int index = (pp == 0) ? i : ne + nhm1 + new_pp - 1;
    //           for (int ll = 0; ll < COMP; ++ll)
    //             for (int dd = 0; dd < D; ++dd)
    //               vecq(index) = sol.vecQ[COMP*ed.offset_q+pp+ndof_w*(dd+D*ll)];
    //         }

    //       nhm1 += get_ndof(max_order, ed.type) - 1;
    //     }
    //   }
    //   else
    //   {
    //     VVector<Vec<size> > & vecq = dynamic_cast<VVector<Vec<size> > &> (gfunc.gf_q -> GetVector());
    //     vecq = 0.0;

    //     int nhm1 = 0; 

    //     for (int i = 0; i < ne; ++i) {
    //       ElementData<D, COMP> & ed = *eldata[i];
          
    //       int ndof_w    = ed.ndof_w;
    //       int order_loc = ed.order;
          
    //       int * conv_loc;
    //       if (ed.type == ET_TRIG)
    //         conv_loc = &ndof_converter_tri[order_loc][0];
    //       else if (ed.type == ET_QUAD)
    //         conv_loc = &ndof_converter_quad[order_loc][0];
    //       else if (ed.type == ET_TET)
    //         conv_loc = &ndof_converter_tet[order_loc][0];
    //       else if (ed.type == ET_HEX)
    //         conv_loc = &ndof_converter_hex[order_loc][0];

    //       for (int pp = 0; pp < ndof_w; ++pp) {
    //         int new_pp = conv_loc[pp];
    //         int index = (pp == 0) ? i : ne + nhm1 + new_pp - 1;
    //         for (int ll = 0; ll < COMP; ++ll)
    //           vecw(index) = sol.vecW[COMP*ed.offset_w+pp+ll*ndof_w];
    //       }

    //       if (Model::Diffusion || Model::Source) 
    //         for (int pp = 0; pp < ndof_w; ++pp) {
    //           int new_pp = conv_loc[pp];
    //           int index = (pp == 0) ? i : ne + nhm1 + new_pp - 1;
    //           for (int ll = 0; ll < COMP; ++ll)
    //             for (int dd = 0; dd < D; ++dd)
    //               vecq(index)[dd+D*ll] = sol.vecQ[COMP*ed.offset_q+pp+ndof_w*(dd+D*ll)];
    //         }

    //       nhm1 += get_ndof(max_order, ed.type) - 1;
    //     }
    //   }
    // }
    // else
    // {
    //   VVector<Vec<COMP> > & vecw = dynamic_cast<VVector<Vec<COMP> > &> (gfunc.gf_w -> GetVector());
    //   vecw = 0.0;

    //   const int size = (Model::Diffusion || Model::Source) ? COMP * D : COMP;
    //   VVector<Vec<size> > & vecq = dynamic_cast<VVector<Vec<size> > &> (gfunc.gf_q -> GetVector());
    //   vecq = 0.0;

    //   int nhm1 = 0; 

    //   for (int i = 0; i < ne; ++i) {
    //     ElementData<D, COMP> & ed = *eldata[i];
        
    //     int ndof_w    = ed.ndof_w;
    //     int order_loc = ed.order;
        
    //     int * conv_loc;
    //     if (ed.type == ET_TRIG)
    //       conv_loc = &ndof_converter_tri[order_loc][0];
    //     else if (ed.type == ET_QUAD)
    //       conv_loc = &ndof_converter_quad[order_loc][0];
    //     else if (ed.type == ET_TET)
    //       conv_loc = &ndof_converter_tet[order_loc][0];
    //     else if (ed.type == ET_HEX)
    //       conv_loc = &ndof_converter_hex[order_loc][0];

    //     for (int pp = 0; pp < ndof_w; ++pp) {
    //       int new_pp = conv_loc[pp];
    //       int index = (pp == 0) ? i : ne + nhm1 + new_pp - 1;
    //       for (int ll = 0; ll < COMP; ++ll)
    //         vecw(index)[ll] = sol.vecW[COMP*ed.offset_w+pp+ll*ndof_w];
    //     }

    //     if (Model::Diffusion || Model::Source) 
    //       for (int pp = 0; pp < ndof_w; ++pp) {
    //         int new_pp = conv_loc[pp];
    //         int index = (pp == 0) ? i : ne + nhm1 + new_pp - 1;
    //         for (int ll = 0; ll < COMP; ++ll)
    //           for (int dd = 0; dd < D; ++dd)
    //             vecq(index)[dd+D*ll] = sol.vecQ[COMP*ed.offset_q+pp+ndof_w*(dd+D*ll)];
    //       }

    //     nhm1 += get_ndof(max_order, ed.type) - 1;
    //   }

    // }

    FlatVector<Vec<COMP>> vecw = gfunc.gf_w->GetVector().FV<Vec<COMP>>();
      vecw = 0.0;

      const int size = (Model::Diffusion || Model::Source) ? COMP * D : COMP;
      FlatVector<Vec<size>> vecq = gfunc.gf_q->GetVector().FV<Vec<size>>();

      vecq = 0.0;

      int nhm1 = 0; 

      for (int i = 0; i < ne; ++i) {
        ElementData<D, COMP> & ed = *eldata[i];
        
        int ndof_w    = ed.ndof_w;
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
          for (int ll = 0; ll < COMP; ++ll)
            vecw(index)[ll] = sol.vecW[COMP*ed.offset_w+pp+ll*ndof_w];
        }

        if (Model::Diffusion || Model::Source) 
          for (int pp = 0; pp < ndof_w; ++pp) {
            int new_pp = conv_loc[pp];
            int index = (pp == 0) ? i : ne + nhm1 + new_pp - 1;
            for (int ll = 0; ll < COMP; ++ll)
              for (int dd = 0; dd < D; ++dd)
                vecq(index)[dd+D*ll] = sol.vecQ[COMP*ed.offset_q+pp+ndof_w*(dd+D*ll)];
          }

        nhm1 += get_ndof(max_order, ed.type) - 1;
      }
}
