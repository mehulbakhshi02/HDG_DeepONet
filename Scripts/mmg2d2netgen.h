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
    if(line=="Vertices") break;
  }
  inmesh >> nv;
 vector<vector<double> > vert;
 vert.resize(nv,vector<double> (3,0.0));
 for(int i=0;i<nv;i++)
 {
   inmesh>>vert[i][0]>>vert[i][1]>>vert[i][2];
 }

 while(true)
 {
   getline(inmesh,line);
   if(line=="Edges") break;
 }
 int nedge;
 inmesh>>nedge;
 vector<vector<int> > edges;
 edges.resize(nedge,vector<int>(3,0.0));
 for(int i=0;i<nedge;i++)
 {
   inmesh>>edges[i][0]>>edges[i][1]>>edges[i][2];
 }

 while(true)
 {
   getline(inmesh,line);
   if(line=="Triangles") break;
 }
 int ntriang;
 inmesh>>ntriang;
 vector<vector<int> >connect;
 connect.resize(ntriang,vector<int> (4,0.0));
 for(int i=0;i<ntriang;i++)
 {
   inmesh>>connect[i][0]>>connect[i][1]>>connect[i][2]>>connect[i][3];
 }
//  vector<int> ednr1(nv+1,0.0),pv(4,0.0);
//  vector<double> dist1(nv+1,0.0),dist2(nv+1,0.0);
//   inmesh.close();
//  vector<vector<double> > VGE;
//  double d1,d2;
//  vector<double> temp(3,0.0);
//  int nvv=4;
//  for(int i=0;i<nvv;i++)
//  {
//    pv[i]=i+1;
//    ednr1[pv[i]]=i+1;
//  }
//  for(int i=4;i<nv;i++)
//  {
//    if(vert[i][1]==0)
//     {
//       temp[0]=i+1;
//       temp[1]=1;
//       d1=fabs(1-vert[i][0]);
//       d2=fabs(0-vert[i][0]);
//       temp[2]=d2;
//       VGE.push_back(temp);
//       ednr1[temp[0]]=1;
//       dist1[temp[0]]=temp[2];
//       dist2[temp[0]]=dist1[temp[0]];

//     }
//     else if(vert[i][0]==1)
//     {
//       temp[0]=i+1;
//       temp[1]=2;
//       d1=fabs(1-vert[i][1]);
//       d2=fabs(0-vert[i][1]);
//       temp[2]=d2;
//       VGE.push_back(temp);
//       ednr1[temp[0]]=2;
//       dist1[temp[0]]=temp[2];
//       dist2[temp[0]]=dist1[temp[0]];

//     }
//     else if(vert[i][1]==1)
//     {
//       temp[0]=i+1;
//       temp[1]=3;
//       d1=fabs(1-vert[i][0]);
//       d2=fabs(0-vert[i][0]);
//       temp[2]=d1;
//       VGE.push_back(temp);
//       ednr1[temp[0]]=3;
//       dist1[temp[0]]=temp[2];
//       dist2[temp[0]]=dist1[temp[0]];

//     }
//     else if(vert[i][0]==0)
//     {
//      temp[0]=i+1;
//      temp[1]=4;
//      d1=fabs(1-vert[i][1]);
//      d2=fabs(0-vert[i][1]);
//      temp[2]=d1;
//      VGE.push_back(temp);
//      ednr1[temp[0]]=4;
//      dist1[temp[0]]=temp[2];
//      dist2[temp[0]]=dist1[temp[0]];

//     }

//  }
//  int m=0;
//  while(m < nvv){
//    dist2[pv[m]] = 1.0;
//    m++;
//  }

//  vector<vector<int> >EGE;
//  vector<int> temp2;
//  temp2.resize(2,0.0);

//   for(int i=0;i<nedge;i++)
//  {

//    if((vert[edges[i][0]-1][1]==0.0)&&(vert[edges[i][1]-1][1]==0.0))
//    {

//      temp2[0]=i+1;
//      temp2[1]=1;
//      EGE.push_back(temp2);

//    }
//    else if((vert[edges[i][0]-1][0]==1.0)&&(vert[edges[i][1]-1][0]==1.0))
//    {
//      temp2[0]=i+1;
//      temp2[1]=2;
//      EGE.push_back(temp2);

//    }
//    else if((vert[edges[i][0]-1][1]==1.0)&&(vert[edges[i][1]-1][1]==1.0))
//    {
//      temp2[0]=i+1;
//      temp2[1]=3;
//      EGE.push_back(temp2);

//    }
//    else if((vert[edges[i][0]-1][0]==0.0)&&(vert[edges[i][1]-1][0]==0.0))
//    {
//      temp2[0]=i+1;
//      temp2[1]=4;
//      EGE.push_back(temp2);
//    }
//  }
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
