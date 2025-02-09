/** @file
 * @brief This function will write the order and make it a
 * real number. The computed integers can become
 * real numbers as we do a volume averaging.
 * @param[in] - filename - The file name which we give for writing out the order.
 * @param[in] - eleorder - The element order for the new mesh. Size = ne double
*/
template <int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::WriteOrder(const string & filename, const vector<int> & eleorder) {
	// Mesh curving options
	int mesh_points = ma->GetNV();
	if(AnisotropyData::curved_mesh)
	{
	  mesh_points = mesh_points - nf;
	}
	vector<double>nodeorder(mesh_points,0.0);

	Array<int> elns;
	for (int j = 0; j < mesh_points; ++j) 
	{
	    ma->GetVertexElements(j, elns);
		double vol = 0.0;
		double order_loc = 0.0;
		for (int k = 0; k < elns.Size(); ++k) 
		{
			int elnr = elns[k];
			ElementData<D,COMP> & ed = *eldata[elnr];
			double vol_loc = ed.vol;
			order_loc += vol_loc * (double)eleorder[elnr];
			vol += vol_loc;
		}

		nodeorder[j] = order_loc/vol;
	}
// Open the node order file and set the header appropriately for the mesh generator being used
	fstream outnodeorder;
	outnodeorder.open(filename.c_str(), ios::out);
	if((AnisotropyData::mesh_generator == AnisotropyData::refine || AnisotropyData::mesh_generator == AnisotropyData::mmg3d)&& D==3)
	{
		outnodeorder << "MeshVersionFormatted 2" << endl;
		outnodeorder << "Dimension 3" << endl;
		outnodeorder << "SolAtVertices" << endl;
		outnodeorder << mesh_points << endl;
		outnodeorder << "1 1"<<endl<<endl;
	}
	if(D==2)
	{
	    outnodeorder << D << " " << "1" << " " << mesh_points << " " << "2" << endl;
	}
	outnodeorder.precision(16);
	outnodeorder.setf(ios::scientific, ios::floatfield);

	for (int j = 0; j < mesh_points; j++)
	{
		double elemorder = nodeorder[j];
		outnodeorder << elemorder << endl;
	}
	
	if((AnisotropyData::mesh_generator == AnisotropyData::refine || AnisotropyData::mesh_generator == AnisotropyData::mmg3d)&& D==3)
	{
		outnodeorder << "End" << endl;
	}
	outnodeorder.close();
	// // Write out average p
	// double p_avg = 0.0;
	// for(int i=0;i<ne;i++)
	// {
	// 	p_avg += (double)order_array[i];
	// }
	// p_avg = p_avg/(double)ne;
	// fstream outpavg;
	// outpavg.open("pavg.csv", ios::app);
	// outpavg << ne << "," << p_avg << endl;
	// outpavg.close();
}