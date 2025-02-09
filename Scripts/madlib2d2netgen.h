using namespace std;
void Mmg2D2Netgen(string mmg, string netgen) {
  // system("gmsh adapted.o.mesh -3 -o adapted.mesh -format mesh");
  // system("gmsh adapted.o.mesh -3 -o output.msh -format msh");

  double dummy;
  int gedges;
  string dummy_bc;

  fstream inmesh, outmesh;
  int nv;
  string line;

  //opening bamg mesh file
  inmesh.open(mmg.c_str(), ios::in);
  while(true){
    getline(inmesh,line);
    if(line=="$Nodes") break;
  }
  inmesh >> nv;
 vector<vector<double> > vert;
 vert.resize(nv,vector<double> (3,0.0));
 for(int i=0;i<nv;i++)
 {
   inmesh>>dummy>>vert[i][0]>>vert[i][1]>>vert[i][2];
 }

 while(true)
 {
   getline(inmesh,line);
   if(line=="$Elements") break;
 }
 int nelements;
 inmesh>>nelements;

 vector<int> edges1, edges2, edge_bc;
 vector<int> connect1, connect2, connect3;
 int ntriang =0 , nedge = 0;
 for(int i=0;i<nelements;i++)
 {
  int el_type;
  inmesh>>dummy>>el_type;
  if(el_type==2)
  {
    int c1, c2, c3;
    inmesh>>dummy>>dummy>>dummy>>dummy>>c1>>c2>>c3;
    connect1.push_back(c1);
    connect2.push_back(c2);
    connect3.push_back(c3);
    ntriang++;
  }
  else if(el_type==1)
  {
    int e1, e2, bc_old, bc_new;
    inmesh>>dummy>>dummy>>bc_old>>dummy>>e1>>e2;
    edges1.push_back(e1);
    edges2.push_back(e2);
    bc_new = bc_old;//correct_bc(bc_old);
    // bc_new = correct_bc(bc_old);
    edge_bc.push_back(bc_new);
    nedge++;
  }
 }
 vector<vector<int> >connect;
 connect.resize(ntriang,vector<int> (3,0.0));
 for(int i=0;i<ntriang;i++)
 {
    connect[i][0] = connect1[i];
    connect[i][1] = connect2[i];
    connect[i][2] = connect3[i];
 }
vector<vector<int> > edges;
 edges.resize(nedge,vector<int>(3,0.0));
 for(int i=0;i<nedge;i++)
 {
   inmesh>>edges[i][0]>>edges[i][1]>>edges[i][2];
   edges[i][0] = edges1[i];
   edges[i][1] = edges2[i];
   edges[i][2] = edge_bc[i];
 }

  int j=0;
  outmesh.open(netgen.c_str(), ios::out);
  outmesh.precision(16);
  outmesh.setf(ios::scientific, ios::floatfield);

outmesh << "mesh3d" << endl << "dimension" << endl << "2" << endl << "geomtype" << endl << "0" << endl;
outmesh << endl;
outmesh << "#" << " " << "surfnr" << "    " << "bcnr" << "    " << "domin" << "    " << "domout" << "    " << "np" << "      " << "p1" << "      " << "p2" << "      " << "p3" << endl;
outmesh << "surfaceelements" << endl;
outmesh << ntriang << endl;
while(j < ntriang){
 outmesh << "       " <<  "2" << "       " << "1" << "       " << "0" << "       " << "0" << "       " << "3";
 outmesh.width(8);
 outmesh << connect[j][0];
 outmesh.width(8);
 outmesh << connect[j][1];
 outmesh.width(8);
 outmesh << connect[j][2] << endl;
 j++;
}
outmesh << endl << endl;
outmesh << "#" << " " << "matnr" << "    " << "np" << "    " << "p1" << "    " << "p2" << "    " << "p3" << "    " << "p4" << endl;
outmesh << "volumeelements" << endl;
outmesh << "0" << endl << endl;
outmesh << "#" << " " << "surfid" << "       " << "0" << "      " << "p1" << "       " << "p2" << "   " << "trinum1" << "  " << "trinum2" << " " << "domin/sf1" << " " << "domout/sf2" << " " << "ednr1" << "    " << "dist1" << "     " <<  "ednr2" << "     " << "dist2" << endl;
 outmesh << "edgesegmentsgi2" << endl;
 outmesh << nedge << endl;
 j=0;
 while (j < nedge){
   outmesh.width(8);
   outmesh << edges[j][2] << "       " << 0;
   outmesh.width(8);
   outmesh << edges[j][0];
   outmesh.width(8);
   outmesh << edges[j][1] << "       " << -1 << "       " << -1 << "        " << 1 << "        " << 0;
   outmesh.width(8);
   outmesh << edges[j][2];
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
   vert_1 = vert[j][0];
   outmesh << vert_1 << "     ";
   outmesh.width(22);
   double vert_2;
   vert_2 = vert[j][1];

   outmesh << vert_2 << "     ";
   outmesh.width(22);
   outmesh << zero  << endl;
   j++;
 }

 outmesh << "materials" << endl; //hardcoded region
 outmesh <<  "1" << endl;
 outmesh << "1" << " " << "domain1" << endl;


 outmesh << endl << "endmesh" << endl;


 outmesh.close();
}
