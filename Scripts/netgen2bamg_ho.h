using namespace std;

void Netgen2Bamg(string netgen_geo, string bamg_geo, string netgen_mesh, string bamg_mesh) 
{

  fstream ingeo,outgeo;
  int i,j,k, np;
  double dummy;
  string dummy_bc;
  fstream inmesh, outmesh;
  int nv;
  //opening netgen geometry file
  ingeo.open(netgen_geo.c_str(),ios::in);
  double val1, val2;
  int v1, v2, v3;
  vector<double> vert1, vert2;  
  string line;
  while(true){
    getline(ingeo,line);
    if (line == "points") break;
  }
  while(true){
    getline(ingeo,line);
    if (line.empty()) break; 
    stringstream parse(line);
    parse >> np >> val1 >> val2;
    vert1.push_back(val1);
    vert2.push_back(val2);
  }
  vector<int> bc_val(np, 0);
  while(true){
    getline(ingeo,line);
    if (line == "segments") break;

  }

  vector<int> seg1, seg2, seg3, bcnr, np_seg, nv_num;
  string bcstr;
  char dumc;
  int bc; 
  int gedges, edges;
  gedges=0;
  edges=0;
  vector<int> vert_remove(np, 0);
  while(true){
    int np_loc;
    getline(ingeo,line);
    if (line.empty()) break; 
    // stringstream parse(line);
    istringstream parse(line);
    parse >> dummy >> dummy >> np_loc;
    np_seg.push_back(np_loc);

    if(np_loc == 2)
    {
      parse >> v1 >> v2;  
      seg1.push_back(v1);
      seg2.push_back(v2);
      seg3.push_back(0);
      parse >> bcstr;
      stringstream parsechar(bcstr);
      parsechar >> dumc >> dumc >> dumc >> dumc >> bc;
      bcnr.push_back(bc);
      bc_val[v1-1] = bc;
      bc_val[v2-1] = bc;
      edges++;

    }
    else if(np_loc == 3)
    {
      parse >> v1 >> v2 >> v3;  
      seg1.push_back(v1);
      seg2.push_back(v2);
      seg3.push_back(v3);
      // zero indexing
      int p_v2 = v2-1;
      vert_remove[p_v2] = -1;// -1 indicates removal of this node if zero then the node remains
      parse >> bcstr;
      stringstream parsechar(bcstr);
      parsechar >> dumc >> dumc >> dumc >> dumc >> bc;
      bcnr.push_back(bc);
      bc_val[v1-1] = bc;
      bc_val[v2-1] = bc;
      bc_val[v3-1] = bc;

      // bcnr.push_back(bc);
      // gedges = gedges + 2;
      gedges++;
      edges++;
      nv_num.push_back(gedges);
    }
    else
    {
      cout<<"Netgen2Bamg converter written only upto 3rd order geometries!"<<endl;
      exit(1);
    }
    // edges++;
  }
  ingeo.close();

  vector<int> ind_shift(np, 0);// the new number of vertices in the linear mesh
  int nv_fin = 0;
  int count = 0;
  cout<<np<<" "<<edges<<endl;
  for(int i=0;i<np;i++)
  {
    // check for removal of vertex
    int check = vert_remove[i];//already zero indexed
    if(check==-1)// vertex is to be removed
    {
      count++;
      ind_shift[i] = -1;
    }
    else
    {
      ind_shift[i] = count;
      nv_fin++;
    }
    // cout<<i<<" "<<count<<endl;
  }

  if(nv_fin!=(np-gedges))
  {
    cout<<"Something is wrong"<<endl;
  }
  //opening netgen mesh file
  inmesh.open(netgen_mesh.c_str(),ios::in);
  while(true){
    getline(inmesh,line);
    if (line == "surfaceelements") break;
  }

  int mtris, nvmax;
  inmesh >> mtris;
  int tripoints[mtris][3];
  nvmax=1;
  i = 0;
  while (i < mtris){
    inmesh >> dummy >> dummy >> dummy >> dummy >> dummy >> tripoints[i][0] >> tripoints[i][1] >> tripoints[i][2]; 
    nvmax=max(max(tripoints[i][0], tripoints[i][1]), max(tripoints[i][2],nvmax));
    i++;
  }

  while(true){
    getline(inmesh,line);
    if (line == "edgesegmentsgi2") break;
  }
  int medges, vertex_geom = 0;
  inmesh >> medges;
  int epoints[medges][2], ednr1[medges], ednr2[medges], surfid[medges];
  double dist1[medges], dist2[medges]; 
  int v_surfid[nvmax];
  for (i = 0; i < nvmax; ++i) v_surfid[i] = 0;
  i=0;
  int nv_geo = 0;
  while (i < medges){
    inmesh >> surfid[i] >> dummy >> epoints[i][0] >> epoints[i][1] >> dummy >> dummy >> dummy >> dummy >> ednr1[i] >> dist1[i] >> ednr2[i] >> dist2[i];
    nv_geo = max(epoints[i][0], max(epoints[i][1], nv_geo));
    if (dist1[i] == 0 ) {
       v_surfid[epoints[i][0] - 1] = epoints[i][0]; 
       vertex_geom++;
    }
    else {
       v_surfid[epoints[i][0] - 1] = surfid[i];
    }
    i++;
  }

  while(true){
    getline(inmesh,line);
    if (line == "points") break;
  }
  inmesh >> nv;
  double mpoints[nv][2];
  i=0;
  while(i < nv){
    inmesh >> mpoints[i][0] >> mpoints[i][1] >> dummy;
    i++;
  }
  inmesh.close();

  //  //creating bamg geometry file
  outgeo.open(bamg_geo.c_str(),ios::out); 
  outgeo << "MeshVersionFormatted 0" << endl;
  outgeo << "Dimension 2" << endl;
  outgeo << "Vertices " << np-gedges << endl;   

  outgeo.precision(16);
  outgeo.setf(ios::fixed,ios::floatfield);
  outgeo.setf(ios::showpoint);

  j=0 ;
  while(j < np){
      if(ind_shift[j]!=-1)
      {     
        outgeo << vert1[j] << " " << vert2[j] << " " << bc_val[j] << endl;
      }
    j++;
  }


  outgeo << "Edges " << edges << endl;
  j=0;
  while (j < edges){
    if(np_seg[j] == 2)
    {
      int v1_seg = seg1[j];// the value is one indexed
      int p_v1_seg = v1_seg-1;// zero indexed position
      int shifted_pos_v1 = ind_shift[p_v1_seg];

      int v2_seg = seg2[j];// the value is one indexed
      int p_v2_seg = v2_seg-1;// zero indexed position
      int shifted_pos_v2 = ind_shift[p_v2_seg];
      // cout<<ind_shift[p_v1_seg]<<" "<<v1_seg<<endl;
      if(shifted_pos_v1==-1 || shifted_pos_v2==-1)
      {
        cout<<"Something is wrong! The shifted index cannot be -1"<<endl;
      }
      int new_v1_ind = v1_seg-shifted_pos_v1;
      int new_v2_ind = v2_seg-shifted_pos_v2;

      // outgeo << seg1[j] << " " << seg2[j] << " " << bcnr[j] << endl;
      outgeo << new_v1_ind << " " << new_v2_ind << " " << bcnr[j] << endl;
    }
    else if(np_seg[j] == 3)
    {
      // outgeo << seg1[j] << " " << seg2[j] << " " << bcnr[j] << endl;
      // outgeo << seg1[j]-(j)%(np-gedges) << " " << seg3[j]-(j+1)%(np-gedges) << " " << bcnr[j] << endl;
      int v1_seg = seg1[j];// the value is one indexed
      int p_v1_seg = v1_seg-1;// zero indexed position
      int shifted_pos_v1 = ind_shift[p_v1_seg];

      int v3_seg = seg3[j];// the value is one indexed
      int p_v3_seg = v3_seg-1;// zero indexed position
      int shifted_pos_v3 = ind_shift[p_v3_seg];

      if(shifted_pos_v1==-1 || shifted_pos_v3==-1)
      {
        cout<<"Something is wrong! The shifted index cannot be -1"<<endl;
      }
      int new_v1_ind = v1_seg-shifted_pos_v1;
      int new_v3_ind = v3_seg-shifted_pos_v3;
      // outgeo << seg1[j] << " " << seg2[j] << " " << bcnr[j] << endl;
      outgeo << new_v1_ind << " " << new_v3_ind << " " << bcnr[j] << endl;
    }
    else
    {
      cout<<"Netgen2Bamg converter written only upto 3rd order geometries!"<<endl;
      exit(1);
    }
    j++;
  } 


  // Might be fix for geometry requiring many points to define the geometry (curved edges only)
  outgeo << "RequiredVertices " << np-gedges<< endl;
  j=0 ;
  while(j < np-gedges){
    outgeo << j+1 << endl;
    j++;
  }
  outgeo << "SubDomain 1\n2 1 1 0\n"<<endl;
  
  outgeo << "Corners " << np-gedges << endl;
  j=0;
  while(j < np-gedges){
    outgeo << j+1 << endl;
    j++;
  }
    // hack for tangents to be tested
   outgeo << "TangentAtEdges"<<endl; 
   outgeo << 2*(edges)<<endl;
  j=0;
  while (j < edges){
    if(np_seg[j] == 2)
    {
      int p1 = seg1[j] - 1;
      int p2 = seg2[j] - 1;
      double p1_x = vert1[p1];
      double p1_y = vert2[p1];
      double p2_x = vert1[p2];
      double p2_y = vert2[p2];
      double x_diff = p2_x-p1_x;
      double y_diff = p2_y-p1_y;
      double mag = sqrt(x_diff*x_diff+(y_diff*y_diff));
      double v1_x = x_diff/mag;
      double v1_y = y_diff/mag;
      x_diff = p1_x-p2_x;
      y_diff = p1_y-p2_y;
      mag = sqrt(x_diff*x_diff+(y_diff*y_diff));
      double v2_x = x_diff/mag;
      double v2_y = y_diff/mag;
      outgeo << j+1 <<" "<< 1 << " " << v1_x << " " << v1_y <<endl;
      outgeo << j+1 <<" "<< 2 << " " << v2_x << " " << v2_y <<endl;

    }
    else if(np_seg[j] == 3)
    {
      int p1 = seg1[j] - 1;
      int p2 = seg2[j] - 1;
      int p3 = seg3[j] - 1;
      double p1_x = vert1[p1];
      double p1_y = vert2[p1];
      double p2_x = vert1[p2];
      double p2_y = vert2[p2];
      double p3_x = vert1[p3];
      double p3_y = vert2[p3];


      double x_diff = p2_x-p1_x;
      double y_diff = p2_y-p1_y;
      double mag = sqrt(x_diff*x_diff+(y_diff*y_diff));
      double v1_x = x_diff/mag;
      double v1_y = y_diff/mag;
      x_diff = p3_x-p2_x;
      y_diff = p3_y-p2_y;
      mag = sqrt(x_diff*x_diff+(y_diff*y_diff));
      double v2_x = x_diff/mag;
      double v2_y = y_diff/mag;
      outgeo << j+1 <<" "<< 1 << " " << v1_x << " " << v1_y <<endl;
      outgeo << j+1 <<" "<< 2 << " " << v2_x << " " << v2_y <<endl;
    }
    else
    {
      cout<<"Netgen2Bamg converter written only upto 3rd order geometries!"<<endl;
      exit(1);
    }
    j++;
  } 
  outgeo.close();    

  // //creating bamg mesh file
  outmesh.open(bamg_mesh.c_str(), ios::out);
  outmesh << "MeshVersionFormatted 0" << endl;
  outmesh << "Dimension 2" << endl;
  outmesh << "Geometry" << endl << "\"bamg.geo\"" << endl;
  outmesh << endl << "Vertices" << endl << nv << endl; 

  outmesh.precision(16);
  outmesh.setf(ios::fixed,ios::floatfield);
  outmesh.setf(ios::showpoint);


  j=0;
  while (j < nv){
    outmesh << mpoints[j][0] << " " << mpoints[j][1] << " " << v_surfid[j] << endl;
    j++;
  }  
  outmesh << endl << "Edges" << endl << medges << endl;
  j=0;
  while (j < medges){
    outmesh << epoints[j][0] << " " << epoints[j][1] << " " << surfid[j] << endl;
    j++;
  }
  outmesh << endl << "Triangles" << endl << mtris << endl;
  j=0;
  while(j < mtris){
    outmesh << tripoints[j][0] << " " << tripoints[j][1] << " " << tripoints[j][2]  << " " << 1 << endl;
    j++;
  }
  outmesh << endl;  // not sure why there are so
  // outmesh << "SubDomainFromMesh" << endl << "1" << endl << "3" << " " << "1" << " " << "1" << " " << "1" << endl << endl; 
  // outgeo << "SubDomainFromGeom 1\n2 1 1 0\n"<<endl;
  outmesh << "VertexOnGeometricVertex" << endl << vertex_geom << endl; 
  j=0,k=1;      
  while (j < medges){
    if(dist1[j] == 0) {
      outmesh << epoints[j][0] << " " << k << endl;
      k++;
    }
    j++;
    }
  outmesh << endl;
  outmesh << "VertexOnGeometricEdge" << endl <<  medges-vertex_geom   << endl;
  j=0;      
  while (j < medges){
    if(dist1[j] != 0) outmesh << epoints[j][0] << " " << ednr1[j] << " " << dist1[j] << endl; 
    j++;
  }
  outmesh << endl;
  outmesh << "EdgeOnGeometricEdge" << endl << medges << endl;
  j= 0;
  while (j < medges){
    outmesh << j+1 << " " << ednr2[j] << endl; 
    j++;
  }   
  outmesh << endl << "End";
  outmesh.close();

}