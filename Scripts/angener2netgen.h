using namespace std;
void Angener2Netgen (string angener, string netgen) {
  
  
  int i,j, np;
  double dummy;
  int gedges;
  string dummy_bc;

  ifstream inmesh;
  fstream outmesh;
  int nv;
  int ntris;
  string line;
  
  //opening angener mesh file
  inmesh.open(angener.c_str(),ios::in);
  getline(inmesh,line);
  stringstream stream(line);
  int k = 0;
  int mesh_data[4] = {0};//First line contains 3 or 4 elements depending on the boundary conditions
  //In our case it should always contain 4 elements. npoin, nelem, nblem and nbc. This does not
  // depend on the type of mesh being generated.

  while(k<4) {
    int n;
    stream >> n;
    if(!stream)
      break;
    mesh_data[k] = n;
    k++;
  }
  nv = mesh_data[0];//Gives the number of points
  ntris = mesh_data[1];//Gives the number of triangles in the mesh
  gedges = mesh_data[2];//Gives the number of mesh edges that are on geometric edges
  int nbc = mesh_data[3];//Gives the number of bounday segments
  
  // Reading the second line to get information about periodic boundary but we do not use this informatin.
  // This get line is just to skip the line reading
  getline(inmesh,line);

  //Start reading information about the location of vertices
  i = 0;
  double vert1[nv], vert2[nv];
  while(i<nv){
    inmesh >> vert1[i] >> vert2[i];
    i++;
  }

  //Start reading information about the connectivity of triangles
  int p1[ntris], p2[ntris], p3[ntris];
  i=0;
  while(i < ntris){
    inmesh >> p1[i] >> p2[i] >> p3[i]; // might hav to be changed for non-square meshes.
    i++;
  }

  //Start reading information about the connectivity of edges that lie on geometric edges
  int seg1[gedges], seg2[gedges], bcnr[gedges], ednr2[gedges];


  int ednr1[nv+1]; // since vertices numbers are from 1 to nv+1;
  double dist1[nv+1], dist2[nv+1];
  for (i = 0; i < nv+1; ++i) {
    ednr1[i] = 0;
    dist1[i] = 0.0;
    dist2[i] = 0.0;    
  }

//This section is hard coded for square domain

  int com_vertex = 0;//Counting vertices that lie on more than one geometric edge
  i=0;
  while (i < gedges){
    inmesh >> seg1[i] >> seg2[i] >> ednr2[i];//Read in all mesh edges that lie on geometric edges
    bcnr[i] = 1;
    if((ednr1[seg1[i]] != 0 && ednr1[seg1[i]] != ednr2[i]) || (ednr1[seg2[i]] != 0 && ednr1[seg2[i]] != ednr2[i]))//Check for mesh edges that have not been assigned or are on multiple geometric edges
    {
      if((ednr1[seg1[i]] != 0 && ednr1[seg1[i]] != ednr2[i]))//If mesh vertex 1 lies on multiple edges then change the distance
      {
        dist2[seg1[i]] = 1.0;
      }
      if((ednr1[seg2[i]] != 0 && ednr1[seg2[i]] != ednr2[i]))//If mesh vertex 2 lies on multiple edges then change the distance
      {
        dist2[seg2[i]] = 1.0;       
      }
      com_vertex++;
    }
    else
    {
      ednr1[seg1[i]] = ednr2[i];//seg1 and seg2 give the vertex number which lies on a geometric edge. ednr2 gives the boundary number
      ednr1[seg2[i]] = ednr2[i];
      switch(ednr2[i])//Depending on which edge the vertex lies we calculate its distance from the geometric vertex on that edge
      {
        case 1:
          dist1[seg1[i]] = sqrt(pow(vert1[seg1[i]], 2) + pow(vert2[seg1[i]],2));
          dist1[seg2[i]] = sqrt(pow(vert1[seg2[i]], 2) + pow(vert2[seg2[i]],2));
          break;
        case 2:
          dist1[seg1[i]] = sqrt(pow(vert1[seg1[i]] - 1, 2) + pow(vert2[seg1[i]],2));
          dist1[seg2[i]] = sqrt(pow(vert1[seg2[i]] - 1, 2) + pow(vert2[seg2[i]],2));
          break;
        case 3:
          dist1[seg1[i]] = sqrt(pow(vert1[seg1[i]] - 1, 2) + pow(vert2[seg1[i]] - 1,2));
          dist1[seg2[i]] = sqrt(pow(vert1[seg2[i]] - 1, 2) + pow(vert2[seg2[i]] - 1,2));
          break;
        case 4:
          dist1[seg1[i]] = sqrt(pow(vert1[seg1[i]], 2) + pow(vert2[seg1[i]] - 1,2));
          dist1[seg2[i]] = sqrt(pow(vert1[seg2[i]], 2) + pow(vert2[seg2[i]] - 1,2));
          break;
        default:
          cout<<"Error!"<<endl;
      }
      dist2[seg1[i]] = dist1[seg1[i]];
      dist2[seg2[i]] = dist1[seg2[i]];
    }

    i++;
  }

  //Section hard coded for square domain ends here

  inmesh.close();

  
  //creating netgen mesh file
  outmesh.open(netgen.c_str(), ios::out);
  outmesh.precision(16);
  outmesh.setf (ios::fixed, ios::floatfield);
  outmesh.setf (ios::showpoint);
  outmesh << "mesh3d" << endl << "dimension" << endl << "2" << endl << "geomtype" << endl << "0" << endl;
  outmesh << endl;
  outmesh << "#" << " " << "surfnr" << "    " << "bcnr" << "    " << "domin" << "    " << "domout" << "    " << "np" << "      " << "p1" << "      " << "p2" << "      " << "p3" << endl;
  outmesh << "surfaceelements" << endl;
  outmesh << ntris << endl;
  j=0;
  while(j < ntris){
    outmesh << "       " <<  "2" << "       " << "1" << "       " << "0" << "       " << "0" << "       " << "3";
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
  outmesh << "0" << endl << endl;  

  outmesh << "#" << " " << "surfid" << "       " << "0" << "      " << "p1" << "       " << "p2" << "   " << "trinum1" << "  " << "trinum2" << " " << "domin/sf1" << " " << "domout/sf2" << " " << "ednr1" << "    " << "dist1" << "     " <<  "ednr2" << "     " << "dist2" << endl;
  outmesh << "edgesegmentsgi2" << endl;
  outmesh << gedges << endl;
  j=0;
  while (j < gedges){
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
    j++;
  }
  outmesh << endl << endl;
  outmesh << "#" << "     " << "X" << "     " << "Y" << "     " << "Z" << endl;
  outmesh << "points" << endl;
  outmesh << nv << endl;
  j=0;
  double zero=0.0;
  int scalarBL_square = 1.0;
  while(j < nv){
    outmesh.width(22);
    double vert_1;
    vert_1 = vert1[j];
    if(scalarBL_square){  // For the scalar BL test case,making sure 0 <= x,y <= 1.0
      if (vert_1 > 1.0) vert_1 = 1.0;
      if (vert_1 > 0.9999999) vert_1 = 1.0;
      if (vert_1 < 0.0) vert_1 = 0.0;
    }
    outmesh << vert_1 << "     ";
    outmesh.width(22);
    double vert_2;
    vert_2 = vert2[j];
    if(scalarBL_square){  // For the scalar BL test case,making sure 0 <= x,y <= 1.0
      if (vert_2 > 1.0) vert_2 = 1.0;
      if (vert_2 > 0.9999999) vert_2 = 1.0;
      if (vert_2 < 0.0) vert_2 = 0.0;
    }
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
