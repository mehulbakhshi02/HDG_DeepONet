using namespace std;
void Refine2Netgen (string refine, string netgen) {

  // system("transmesh adapted.meshb adapted.mesh");

  // system("gmsh adapted.mesh -3 -o output.msh -format msh");
  // system("python3 testgmsh.py");
  // system("rm output.msh");
  double dummy;
  int gedges;
  string dummy_bc;

  fstream inmesh, outmesh;
  int nv;
  int ntris;
  int nelem;
  string line;
  
  //opening bamg mesh file
  inmesh.open(refine.c_str(), ios::in);
  while(true){
    getline(inmesh,line);
    if (line == "Vertices") break;
  }
  inmesh >> nv;
  // Assuming 3D meshes
  vector<double> vert1(nv), vert2(nv), vert3(nv);
  vector<int> v_surfid(nv);
  for (int i = 0; i < nv; ++i)
    inmesh >> vert1[i] >> vert2[i] >> vert3[i] >> v_surfid[i];  

  // while(true){
  //   getline(inmesh,line);
  //   if (line == "Edges") break;
  // }
  // inmesh >> gedges;
  // vector<int> seg1(gedges), seg2(gedges), bcnr(gedges);
  // for (int i = 0; i < gedges; ++i)
  //   inmesh >> seg1[i] >> seg2[i] >> bcnr[i];

  while(true){
    getline(inmesh,line);
    if (line == "Triangles") break;
  }
  inmesh >> ntris;
  vector<int> p1(ntris), p2(ntris), p3(ntris), bcnr(ntris);
  for (int i = 0; i < ntris; ++i)
    inmesh >> p1[i] >> p2[i] >> p3[i] >> bcnr[i] ; // might hav to be changed for non-square meshes.
 while(true){
    getline(inmesh,line);
    if (line == "Tetrahedra") break;
  }
  inmesh >> nelem; 
   vector<int> v1(nelem), v2(nelem), v3(nelem), v4(nelem);
  for (int i = 0; i < nelem; ++i)
    inmesh >> v1[i] >> v2[i] >> v3[i] >> v4[i] >> dummy; // might hav to be changed for non-square meshes.
	 

  inmesh.close();

  // Fixing bcnr for delta wing
  // Hack only for delta wing
  for(int i=0;i<ntris;++i)
  {
    int bc = bcnr[i];
    int bc_new = -1;
    if(bc==1 || bc==2 || bc==3 || bc==4)
      bc_new = 6;
    else if(bc==5 || bc==6 || bc==8 || bc==9 || bc==10)
      bc_new = 1;
    else if(bc==7)
      bc_new = 3;
    bcnr[i] = bc_new;
  }
  //creating netgen mesh file
  outmesh.open(netgen.c_str(), ios::out);
  outmesh.precision(16);
  outmesh.setf(ios::scientific, ios::floatfield);

  outmesh << "mesh3d" << endl << "dimension" << endl << "3" << endl << "geomtype" << endl << "0" << endl;
  outmesh << endl;
  outmesh << "#" << " " << "surfnr" << "    " << "bcnr" << "    " << "domin" << "    " << "domout" << "    " << "np" << "      " << "p1" << "      " << "p2" << "      " << "p3" << endl;
  outmesh << "surfaceelements" << endl;
  outmesh << ntris << endl;
  for (int j = 0; j < ntris; ++j) {

    outmesh << "2"<< setw(8) << bcnr[j]<< setw(8) << "1"<< setw(8)<<"0"<< setw(8)<<"3";
    outmesh << setw(8) << p1[j] << setw(8) << p2[j] << setw(8) << p3[j] << endl;
  }
  outmesh << endl << endl;
  outmesh << "#" << " " << "matnr" << "    " << "np" << "    " << "p1" << "    " << "p2" << "    " << "p3" << "    " << "p4" << endl;
  outmesh << "volumeelements" << endl;
  outmesh << nelem <<endl;
  for (int j = 0; j < nelem; ++j) {
    outmesh << "1"<< setw(8)<<"4";
    // outmesh << setw(8) << v1[j] << setw(8) << v2[j] << setw(8) << v3[j] << setw(8) << v4[j] << endl;
    outmesh << setw(8) << v1[j] << setw(8) << v2[j] << setw(8) << v4[j] << setw(8) << v3[j] << endl;
  }


  // outmesh << "#" << " " << "surfid" << "       " << "0" << "      " << "p1" << "       " << "p2" << "   " << "trinum1" << "  " << "trinum2" << " " << "domin/sf1" << " " << "domout/sf2" << " " << "ednr1" << "    " << "dist1" << "     " <<  "ednr2" << "     " << "dist2" << endl;
  // outmesh << "edgesegmentsgi2" << endl;
  // outmesh << gedges << endl;
  // for (int j = 0; j < gedges; ++j) {
  //   outmesh.width(8);
  //   outmesh << bcnr[j] << "       " << 0;
  //   outmesh.width(8);
  //   outmesh << seg1[j];
  //   outmesh.width(8);
  //   outmesh << seg2[j] << "       " << -1 << "       " << -1 << "        " << 1 << "        " << 0;
  //   outmesh.width(8);
  //   outmesh << ednr1[seg1[j]];
  //   outmesh.width(24);
  //   outmesh << dist1[seg1[j]];
  //   outmesh.width(8);
  //   outmesh << ednr2[j];
  //   outmesh.width(24);
  //   outmesh << dist2[seg2[j]]; 
  //   outmesh << endl;
  // }
  outmesh << endl << endl;
  outmesh << "#" << "     " << "X" << "     " << "Y" << "     " << "Z" << endl;
  outmesh << "points" << endl;
  outmesh << nv << endl;


  for (int j = 0; j < nv; ++j) {
    outmesh.width(24);
    double vert_1;
    vert_1 = vert1[j];
    double vert_2;
    vert_2 = vert2[j];
    double vert_3;
    vert_3 = vert3[j];
    
    outmesh << vert_1 << "     ";
    outmesh.width(24);
    outmesh << vert_2 << "     ";
    outmesh.width(24);
    outmesh << vert_3<<endl;
  }


  outmesh << "materials" << endl; //hardcoded region
  outmesh <<  "1" << endl;
  outmesh << "1" << " " << "domain1" << endl;

  outmesh << endl << "endmesh" << endl;
  
  outmesh.close();
}
