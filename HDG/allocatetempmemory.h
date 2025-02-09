#ifndef ALLOCATETEMPMEMORY_H
#define ALLOCATETEMPMEMORY_H

template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
  ::AllocateTempMemory() {
  
  matA.resize(COMP * ndof_q_max * COMP * ndof_q_max, 0.);
  matB.assign(COMP * ndof_q_max * COMP * ndof_w_max, 0.);
  matC.assign(COMP * ndof_w_max * COMP * ndof_q_max, 0.);
  matD.assign(COMP * ndof_w_max * COMP * ndof_w_max, 0.);
  matL.assign(COMP * ndof_l_max * nf_max * COMP * ndof_q_max, 0.);
  matM.assign(COMP * ndof_l_max * nf_max * COMP * ndof_w_max, 0.);

  matQ.assign(COMP * ndof_q_max * (1 + COMP * nf_max * ndof_l_max), 0.);
  matW.assign(COMP * ndof_w_max * (1 + COMP * nf_max * ndof_l_max), 0.);
    
  insert.assign(max(COMP * ndof_l_max * COMP * ndof_l_max, COMP * ndof_l_max * nf_max), 0.);
  insert2.assign(COMP * ndof_l_max * COMP * ndof_l_max, 0.);  

  //  temp.assign(COMP * ndof_w_max * COMP * max(COMP * nip_max, 1 + COMP * nf_max * ndof_l_max) * D, 0.);
  temp.assign(COMP * ndof_w_max * D * max(COMP * nip_max, 1 + COMP * nf_max * ndof_l_max), 0.);
  temp2.assign(COMP * ndof_w_max * D * COMP * nip_max * D, 0.);
  temp3.assign(COMP * ndof_q_max * COMP * nip_max, 0.);
  temp4.assign(COMP * ndof_q_max * COMP * nip_max, 0.);
  temp5.assign(COMP * ndof_q_max * COMP * nip_max, 0.);

  #ifdef BR2
  if (Model::Diffusion) {
    temp_lift.assign(nf_max * COMP * D * max(nip_max, ndof_w_max) * (1 + COMP * ndof_w_max + COMP * ndof_l_max), .0);
    temp_lift2.assign(nf_max * COMP * D * max(nip_max, ndof_w_max) * (1 + COMP * ndof_w_max + COMP * ndof_l_max), .0);
  }
  #endif

  if (shock_capturing > 0)
    deps.assign(COMP * ndof_w_max * ((Model::Diffusion || Model::Source) ? D+1 : 1), 0.);

  cout << "Temporary memory allocated" << endl;  
}

#endif
