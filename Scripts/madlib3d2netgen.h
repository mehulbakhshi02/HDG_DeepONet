using namespace std;
void Madlib3D2Netgen(string mmg, string netgen) {
  double dummy;
  int gedges;
  string dummy_bc;

  fstream inmesh, outmesh;
  int nv;
  int ntris;
  int nelem;
  string line;

  //opening bamg mesh file
  inmesh.open(mmg.c_str(), ios::in);
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

  while(true){
    getline(inmesh,line);
    if (line == "Edges") break;
  }
  inmesh >> gedges;

  vector<int> seg1(gedges), seg2(gedges), bc_edg(gedges);
  for (int i = 0; i < gedges; ++i)
    inmesh >> seg1[i] >> seg2[i] >> bc_edg[i];

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

  int j=0;
  outmesh.open(netgen.c_str(), ios::out);
  outmesh.precision(16);
  outmesh.setf(ios::scientific, ios::floatfield);

outmesh << "mesh3d" << endl << "dimension" << endl << "3" << endl << "geomtype" << endl << "0" << endl;
outmesh << endl;
outmesh << "#" << " " << "surfnr" << "    " << "bcnr" << "    " << "domin" << "    " << "domout" << "    " << "np" << "      " << "p1" << "      " << "p2" << "      " << "p3" << endl;
outmesh << "surfaceelements" << endl;
outmesh << ntris << endl;
while(j < ntris){
 outmesh << "       " <<  "2" << "       " << bcnr[j] << "       " << "1" << "       " << "0" << "       " << "3";
 outmesh.width(8);
 outmesh << p1[j];
 outmesh.width(8);
 outmesh << p2[j];
 outmesh.width(8);
 outmesh << p3[j] << endl;
 j++;
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

  outmesh << "#" << " " << "surfid" << "       " << "0" << "      " << "p1" << "       " << "p2" << "   " << "trinum1" << "  " << "trinum2" << " " << "domin/sf1" << " " << "domout/sf2" << " " << "ednr1" << "    " << "dist1" << "     " <<  "ednr2" << "     " << "dist2" << endl;
 outmesh << "edgesegmentsgi2" << endl;
 outmesh << gedges << endl;
 j=0;
 while (j < gedges){
   outmesh.width(8);
   outmesh << bc_edg[j] << "       " << 0;
   outmesh.width(8);
   outmesh << seg1[j];
   outmesh.width(8);
   outmesh << seg2[j] << "       " << -1 << "       " << -1 << "        " << 1 << "        " << 0;
   outmesh.width(8);
   outmesh << bc_edg[j];
   outmesh.width(24);
   outmesh << 0;
   outmesh.width(8);
   outmesh << 1;
   outmesh.width(24);
   outmesh << 1;
   outmesh << endl;
   j++;
 }
 outmesh << endl << endl;
 outmesh << "#" << "     " << "X" << "     " << "Y" << "     " << "Z" << endl;
 outmesh << "points" << endl;
 outmesh << nv << endl;
 j=0;
 double zero=0.0;
 int scalarBL_square = 1;
 while(j < nv){
   outmesh.width(22);
   double vert_1;
   vert_1 = vert1[j];
   outmesh << vert_1 << "     ";
   outmesh.width(22);
   double vert_2;
   vert_2 = vert2[j];
   outmesh << vert_2 << "     ";
   outmesh.width(22);
   double vert_3;
   vert_3 = vert3[j];
   outmesh << vert_3 << endl;

   j++;
 }

 outmesh << "materials" << endl; //hardcoded region
 outmesh <<  "1" << endl;
 outmesh << "1" << " " << "domain1" << endl;


 outmesh << endl << "endmesh" << endl;


 outmesh.close();
}
