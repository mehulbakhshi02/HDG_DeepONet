template <int D, int COMP, class Model>
UnifyingFramework<D, COMP, Model>
  ::UnifyingFramework(shared_ptr<PDE> apde, const Flags & flags) : NumProc(apde) {

  // **********************************************
  // Load all necessary parameters
  // **********************************************	

  Model::LoadParameters(apde);

  LoadParameters(apde);

  // **********************************************
  // Initialize function spaces
  // **********************************************	

  Flags hflags;
  hflags.SetFlag ("dim", COMP);
  hflags.SetFlag ("order", max_order);
  fspace.fes_w = make_shared<L2HighOrderFESpace> (ma, hflags);
  apde->AddFESpace("w_primal", fspace.fes_w);

  fspace.fes_l =  make_shared<FacetFESpace> (ma, hflags);
  apde->AddFESpace ("w_facet", fspace.fes_l);

  if (Model::Diffusion || Model::Source) 
  {
    hflags.SetFlag ("dim", COMP*D);
    hflags.SetFlag ("order", max_order);
    fspace.fes_q = make_shared<L2HighOrderFESpace> (ma, hflags);
    apde->AddFESpace("q_primal", fspace.fes_q);
  } else {
    fspace.fes_q = fspace.fes_w;
  }

  hflags.SetFlag ("dim", 1);
  fes_scalar = make_shared<L2HighOrderFESpace> (ma, hflags);
  apde->AddFESpace("scalar", fes_scalar);
  // Dual spaces
  hflags.SetFlag ("dim", COMP);
  hflags.SetFlag ("order", max_order+1);
  fspacedual.fes_w = make_shared<L2HighOrderFESpace> (ma, hflags);
  apde->AddFESpace("w_dual", fspacedual.fes_w);

  fspacedual.fes_l = make_shared<FacetFESpace> (ma, hflags);
  apde->AddFESpace ("w_facet_dual", fspacedual.fes_l);

  if (Model::Diffusion || Model::Source) {
    hflags.SetFlag ("dim", COMP*D);
    hflags.SetFlag ("order", max_order+1);
    fspacedual.fes_q = make_shared<L2HighOrderFESpace> (ma, hflags);
    apde->AddFESpace("q_dual", fspacedual.fes_q);
  } else {
    fspacedual.fes_q = fspacedual.fes_w;
  }

  // Constant space
  hflags.SetFlag ("dim", 1.);
  hflags.SetFlag ("order", 0.);
  fes_const = make_shared<L2HighOrderFESpace> (ma, hflags);
  apde->AddFESpace("const", fes_const);
  
  if(AnisotropyData::read_bamg_solution)
  {
    int order_nodal = 1;
    Flags hflags;
    hflags.SetFlag ("dim", 1);
    hflags.SetFlag ("order", order_nodal);
    fes_nodal = make_shared<NodalFESpace> (ma, hflags);
    apde->AddFESpace("w_nodal", fspace.fes_w);
  }

  if(AnisotropyData::hp)
  {
    Flags hflags;
    hflags.SetFlag ("dim", COMP);
    hflags.SetFlag ("order", max_order+2);
    fspacehp.fes_w = make_shared<L2HighOrderFESpace> (ma, hflags);
    apde->AddFESpace("w_hp", fspacehp.fes_w);

    fspacehp.fes_l =  make_shared<FacetFESpace> (ma, hflags);
    apde->AddFESpace ("w_hp", fspacehp.fes_l);

    if (Model::Diffusion || Model::Source) 
    {
      hflags.SetFlag ("dim", COMP*D);
      hflags.SetFlag ("order", max_order+2);
      fspacehp.fes_q = make_shared<L2HighOrderFESpace> (ma, hflags);
      apde->AddFESpace("q_hp", fspacehp.fes_q);
    } else {
      fspacehp.fes_q = fspacehp.fes_w;
    }    
  }
  
  // **********************************************
  // Register GridFunction
  // **********************************************	

  Flags gfflags;
  // if(COMP==1)
  //   gfunc.gf_w = make_shared<T_GridFunction<double> > (*(fspace.fes_w), "w", gfflags);
  // else
  //   gfunc.gf_w = make_shared<T_GridFunction<Vec<COMP> > > (*(fspace.fes_w), "w", gfflags);

  // apde->AddGridFunction ("w", gfunc.gf_w);

  // if (Model::Diffusion || Model::Source) {

  //   gfunc.gf_q = make_shared<T_GridFunction<Vec<COMP*D> > > (*(fspace.fes_q), "q", gfflags);
  //   apde->AddGridFunction ("q", gfunc.gf_q);
  // } else {
  //   gfunc.gf_q = gfunc.gf_w;    
  // }

  // gf_wd = make_shared<T_GridFunction<double> >(*fes_scalar, "WallDistance", gfflags);
  // apde->AddGridFunction ("WallDistance", gf_wd);
  // if (visualize_adjoint) {
  //   if(COMP==1)
  //     gfuncdual.gf_w =  make_shared<T_GridFunction<double> > (*(fspacedual.fes_w), "wdual", gfflags);
  //   else
  //     gfuncdual.gf_w =  make_shared<T_GridFunction<Vec<COMP> > > (*(fspacedual.fes_w), "wdual", gfflags);
  //   apde->AddGridFunction ("wdual", gfuncdual.gf_w);

  //   if (Model::Diffusion || Model::Source) {
  //     gfuncdual.gf_q = make_shared<T_GridFunction<Vec<COMP*D> > >  (*(fspacedual.fes_q), "qdual", gfflags);
  //     apde->AddGridFunction ("qdual", gfuncdual.gf_q);
  //   } else {
  //     gfuncdual.gf_q = gfuncdual.gf_w;        
  //   }
  // }

  // gf_const = make_shared<T_GridFunction<double> > (*fes_const, "eps", gfflags);
  // apde->AddGridFunction ("eps", gf_const);

  // gf_err = make_shared<T_GridFunction<double> > (*fes_const, "err", gfflags);
  // apde->AddGridFunction ("err", gf_err);

  // gf_excoe = make_shared<T_GridFunction<double> > (*fes_const, "excoe", gfflags);
  // apde->AddGridFunction ("excoe", gf_excoe);
  // gfflags.SetFlag ("dim", 1);
  gf_wd = make_shared<S_GridFunction<double> >(fes_scalar, "WallDistance", gfflags);
  apde->AddGridFunction ("WallDistance", gf_wd);
  gfunc.gf_w = make_shared<S_GridFunction<double> > ((fspace.fes_w), "w", gfflags);
  apde->AddGridFunction ("w", gfunc.gf_w);
  if (Model::Diffusion || Model::Source) {

    gfunc.gf_q = make_shared<S_GridFunction<double> > ((fspace.fes_q), "q", gfflags);
    apde->AddGridFunction ("q", gfunc.gf_q);
  } else {
    gfunc.gf_q = gfunc.gf_w;    
  }
  if (visualize_adjoint) {
    gfuncdual.gf_w =  make_shared<S_GridFunction<double> > ((fspacedual.fes_w), "wdual", gfflags);
    apde->AddGridFunction ("wdual", gfuncdual.gf_w);

    if (Model::Diffusion || Model::Source) {
      gfuncdual.gf_q = make_shared<S_GridFunction<double> >  ((fspacedual.fes_q), "qdual", gfflags);
      apde->AddGridFunction ("qdual", gfuncdual.gf_q);
    } else {
      gfuncdual.gf_q = gfuncdual.gf_w;        
    }
  }

  gf_const = make_shared<S_GridFunction<double> > (fes_const, "eps", gfflags);
  apde->AddGridFunction ("eps", gf_const);

  gf_err = make_shared<S_GridFunction<double> > (fes_const, "err", gfflags);
  apde->AddGridFunction ("err", gf_err);

  gf_excoe = make_shared<S_GridFunction<double> > (fes_const, "excoe", gfflags);
  apde->AddGridFunction ("excoe", gf_excoe);
}
