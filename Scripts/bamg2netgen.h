using namespace std;

void Bamg2Netgen(string bamg, string netgen) {
  double dummy;
  int gedges;
  string dummy_bc;

  fstream inmesh, outmesh;
  int nv;
  int ntris;
  string line;
  
  //opening bamg mesh file
  inmesh.open(bamg.c_str(), ios::in);
  while(true){
    getline(inmesh,line);
    if (line == "Vertices") break;
  }
  inmesh >> nv;
  vector<double> vert1(nv), vert2(nv);
  vector<int> v_surfid(nv);
  for (int i = 0; i < nv; ++i)
    inmesh >> vert1[i] >> vert2[i] >> v_surfid[i];  

  while(true){
    getline(inmesh,line);
    if (line == "Edges") break;
  }
  inmesh >> gedges;
  vector<int> seg1(gedges), seg2(gedges), bcnr(gedges);
  for (int i = 0; i < gedges; ++i)
    inmesh >> seg1[i] >> seg2[i] >> bcnr[i];

  while(true){
    getline(inmesh,line);
    if (line == "Triangles") break;
  }
  inmesh >> ntris;
  vector<int> p1(ntris), p2(ntris), p3(ntris);
  for (int i = 0; i < ntris; ++i)
    inmesh >> p1[i] >> p2[i] >> p3[i] >> dummy ; // might hav to be changed for non-square meshes.

  vector<int> ednr1(nv+1, 0); // since vertices numbers are from 1 to nv+1;
  vector<double> dist1(nv+1, 0.), dist2(nv+1, 0.);

  while(true){
    getline(inmesh,line);
    if (line == "VertexOnGeometricVertex") break;
  }
  int nvv;
  inmesh >> nvv;
  vector<int> pv(nvv);
  for (int i = 0; i < nvv; ++i) {
    inmesh >> pv[i];
    inmesh >> ednr1[pv[i]];
  }
  while(true){
    getline(inmesh,line);
    if (line == "VertexOnGeometricEdge") break;
  }
  int nvg; 
  inmesh >> nvg;
  vector<int> vgnr(nvg);

  for (int i = 0; i < nvg; ++i) {
    inmesh >> vgnr[i];
    inmesh >> ednr1[vgnr[i]] >> dist1[vgnr[i]];
    dist2[vgnr[i]] = dist1[vgnr[i]];
  }
  for (int i = 0; i < nvv; ++i)
    dist2[pv[i]] = 1.0;

  while(true){
    getline(inmesh,line);
    if (line == "EdgeOnGeometricEdge") break;
  }
  int gedges_;
  inmesh >> gedges_;
  vector<int> edge(gedges_);
  vector<int> ednr2(gedges_);
  for (int i = 0; i < gedges_; ++i)
    inmesh >> edge[i] >> ednr2[i];

  inmesh.close();

  //creating netgen mesh file
  outmesh.open(netgen.c_str(), ios::out);
  outmesh.precision(16);
  outmesh.setf(ios::scientific, ios::floatfield);

  outmesh << "mesh3d" << endl << "dimension" << endl << "2" << endl << "geomtype" << endl << "0" << endl;
  outmesh << endl;
  outmesh << "#" << " " << "surfnr" << "    " << "bcnr" << "    " << "domin" << "    " << "domout" << "    " << "np" << "      " << "p1" << "      " << "p2" << "      " << "p3" << endl;
  outmesh << "surfaceelements" << endl;
  outmesh << ntris << endl;
  for (int j = 0; j < ntris; ++j) {

    outmesh << "       " <<  "2" << "       " << "1" << "       " << "0" << "       " << "0" << "       " << "3";
    outmesh << setw(8) << p1[j] << setw(8) << p2[j] << setw(8) << p3[j] << endl;
  }
  outmesh << endl << endl;
  outmesh << "#" << " " << "matnr" << "    " << "np" << "    " << "p1" << "    " << "p2" << "    " << "p3" << "    " << "p4" << endl;
  outmesh << "volumeelements" << endl;
  outmesh << "0" << endl << endl;  

  outmesh << "#" << " " << "surfid" << "       " << "0" << "      " << "p1" << "       " << "p2" << "   " << "trinum1" << "  " << "trinum2" << " " << "domin/sf1" << " " << "domout/sf2" << " " << "ednr1" << "    " << "dist1" << "     " <<  "ednr2" << "     " << "dist2" << endl;
  outmesh << "edgesegmentsgi2" << endl;
  outmesh << gedges << endl;
  for (int j = 0; j < gedges; ++j) {
    outmesh.width(8);
    outmesh << bcnr[j] << "       " << 0;
    outmesh.width(8);
    outmesh << seg1[j];
    outmesh.width(8);
    outmesh << seg2[j] << "       " << -1 << "       " << -1 << "        " << 1 << "        " << 0;
    outmesh.width(8);
    outmesh << ednr1[seg1[j]];
    outmesh.width(24);
    outmesh << dist1[seg1[j]];
    outmesh.width(8);
    outmesh << ednr2[j];
    outmesh.width(24);
    outmesh << dist2[seg2[j]]; 
    outmesh << endl;
  }
  outmesh << endl << endl;
  outmesh << "#" << "     " << "X" << "     " << "Y" << "     " << "Z" << endl;
  outmesh << "points" << endl;
  outmesh << nv << endl;

  // Hack for specific domains
  int scalarBL_square = 0;
  int square_large = 0;
  int l_shape = 0;
  int rectangle = 0;
  int square_m1 = 0;

  double tol = 1e-6;
  double zero=0.0;
  for (int j = 0; j < nv; ++j) {
    outmesh.width(24);
    double vert_1;
    vert_1 = vert1[j];
    double vert_2;
    vert_2 = vert2[j];
    if(scalarBL_square){  // For the scalar BL test case,making sure 0 <= x,y <= 1.0
      if (vert_1 > 1.0) vert_1 = 1.0;
      if (vert_1 > 0.9999999) vert_1 = 1.0;
      if (vert_1 < 0.0) vert_1 = 0.0;
      if (vert_2 > 1.0) vert_2 = 1.0;
      if (vert_2 > 0.9999999) vert_2 = 1.0;
      if (vert_2 < 0.0) vert_2 = 0.0;
    }
    if(square_large){  // For the scalar BL test case,making sure 0 <= x,y <= 1.0
      if (vert_1 > 4.0) vert_1 = 4.0;
      if (vert_1 > 3.9999999) vert_1 = 4.0;
      if (vert_1 < 0.0) vert_1 = 0.0;
      if (vert_2 > 4.0) vert_2 = 4.0;
      if (vert_2 > 3.9999999) vert_2 = 4.0;
      if (vert_2 < 0.0) vert_2 = 0.0;
    }

    if(square_m1){  // For the scalar BL test case,making sure 0 <= x,y <= 1.0
      if (vert_1 > 1.0) vert_1 = 1.0;
      if (vert_1 > 0.9999999) vert_1 = 1.0;
      if (vert_1 < -0.9999999) vert_1 = -1.0;
      if (vert_1 < -1.0) vert_1 = -1.0;
      if (vert_2 > 1.0) vert_2 = 1.0;
      if (vert_2 > 0.9999999) vert_2 = 1.0;
      if (vert_2 < -0.9999999) vert_2 = -1.0;
      if (vert_2 < -1.0) vert_2 = -1.0;
    }
    if(l_shape){  // For the scalar BL test case,making sure 0 <= x,y <= 1.0
      if (vert_1 > 4.0) vert_1 = 4.0;
      if(fabs(vert_1-4.0)<tol) vert_1 = 4.0;
      if(fabs(vert_1-0.0)<tol) vert_1 = 0.0;
      if (vert_1 < 0.0) vert_1 = 0.0;

      if (vert_2 > 4.0) vert_2 = 4.0;
      if(fabs(vert_2-4.0)<tol) vert_2 = 4.0;
      if(fabs(vert_2-0.0)<tol) vert_2 = 0.0;
      if (vert_2 < 0.0) vert_2 = 0.0;
    }
    if(rectangle){  // For the scalar BL test case,making sure 0 <= x,y <= 1.0
      if (vert_1 > 2.0) vert_1 = 2.0;
      if (vert_1 > 1.9999999) vert_1 = 2.0;
      if (vert_1 < 0.0) vert_1 = 0.0;
      if (vert_2 > 1.0) vert_2 = 1.0;
      if (vert_2 > 0.9999999) vert_2 = 1.0;
      if (vert_2 < 0.0) vert_2 = 0.0;
    }

    outmesh << vert_1 << "     ";
    outmesh.width(24);
    outmesh << vert_2 << "     ";
    outmesh.width(24);
    outmesh << zero  << endl;
  }


  outmesh << "materials" << endl; //hardcoded region
  outmesh <<  "1" << endl;
  outmesh << "1" << " " << "domain1" << endl;

  outmesh << endl << "endmesh" << endl;
  
  outmesh.close();
}
