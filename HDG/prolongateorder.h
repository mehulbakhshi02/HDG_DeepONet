template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ProlongateOrder(const Solution & sol_old, Solution & sol_new, const int delta_order) {

  ProlongateOrder(sol_old, sol_new, delta_order, fadata, eldata);

}


template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::ProlongateOrder(const Solution & sol_old, Solution & sol_new, const int delta_order, vector<FacetData<D, COMP> *> & fadata, vector<ElementData<D, COMP> *> & eldata) {

  sol_new.vecQ.assign(ndof_q_total * COMP, 0.);
  sol_new.vecW.assign(ndof_w_total * COMP, 0.);
  sol_new.vecL.assign(ndof_l_total * COMP, 0.);
  
  vector<int> ndof_converter_loc(ndof_w_max);

  // Transfer old coefficient to the "new expansion"
  int index   = 0;
  int index_q = 0;
  
  for (int i = 0; i < ne; ++i) {
    ElementData<D,COMP> & elidata = *eldata[i];
    int order_w    = elidata.order;
    int order_wold = order_w - delta_order;
    int ndof_w     = elidata.ndof_w;
    int ndof_wold  = get_ndof(order_wold, elidata.type);

    transfer_ndof(order_w, order_wold, elidata.type, &ndof_converter_loc[0]);

    for (int ll = 0; ll < COMP; ++ll)
      for (int pp = 0; pp < ndof_wold; ++pp) {
        int newpp = ndof_converter_loc[pp];
        sol_new.vecW[COMP*elidata.offset_w+newpp+ll*ndof_w] = sol_old.vecW[index];
        index++;
      }

    if (Model::Diffusion || Model::Source) 
      for (int ll = 0; ll < COMP; ++ll)
        for (int dd = 0; dd < D; ++dd)
          for (int pp = 0; pp < ndof_wold; ++pp) {
            int newpp = ndof_converter_loc[pp];
            sol_new.vecQ[COMP*elidata.offset_q+newpp+ndof_w*(dd+D*ll)] = sol_old.vecQ[index_q];
            index_q++;
          }

  }
  
  index = 0;  
  for (int i = 0; i < nf; ++i) {
    if (!fadata[i]) continue;

    FacetData<D,COMP> & fd = *fadata[i];

    if (fd.ndof_l == 0) continue;

    // Get the maximum order of w and sigma in the previous computation
    int order_l    = max(eldata[fd.elnr1]->order, eldata[fd.elnr2]->order);
    int order_lold = max(eldata[fd.elnr1]->order-delta_order, eldata[fd.elnr2]->order-delta_order);

    int ndof_l    = fd.ndof_l;
    int ndof_lold = get_ndof(order_lold, fd.type);

    transfer_ndof(order_l, order_lold, fd.type, &ndof_converter_loc[0]);

    for (int ll = 0; ll < COMP; ++ll)
      for (int pp = 0; pp < ndof_lold; ++pp) {
        int newpp = ndof_converter_loc[pp];
        sol_new.vecL[COMP*fd.offset_l+newpp+ll*ndof_l] = sol_old.vecL[index];
        index++;
      }
  }  

  cout << "Solution prolongated." << endl;	                 

}
