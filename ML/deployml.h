template<int D, int COMP, class Model>
void UnifyingFramework<D, COMP, Model>
::DeployML() {

// KUNAL: I need the nip to know how many values are missing
// To store the nip values at the faces
int nip_universal;
// A marker to check that the value is acquired
int nip_acquired = 0; 

for (int i = 0; i<nf && nip_acquired == 0; i++)
{
	FacetData<D, COMP> & fd = *fadata[i];
	if (fd.ndof_l != 0)
	{
		nip_universal = fd.nip;
		nip_acquired = 1;
	}
}

cout<<"nip_universal: "<<nip_universal<<endl;

  int ne_int = 0;
  for(int i=0;i<ne;i++)
  {
    ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
    if(ml_elidata.nf==ml_elidata.nf_bc)
        ne_int++;
  }
  
  Vec<Model::NumParam> param;
  Model::GetParameters(param);
  

  double q_norm = AnisotropyData::norm;

 // Some python initialization
    Py_Initialize();
    
    int PyCheckFlag = 0;
    PyCheckFlag = Py_IsInitialized();
    printf("PyCheckFlag: %d\n",PyCheckFlag);
    
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\"./ML/\")");
   //  // initialize numpy array library
    _import_array();
    std::cout << "Initializing numpy library" << std::endl;
    // initialize numpy array library
   //  // import_array1(-1);
    
    std::cout << "Loading python module" << std::endl;
    PyObject* pName = PyUnicode_DecodeFSDefault("python_module"); // Python filename
    
    // KUNAL
    if (pName == NULL)
    {
    	printf("NULL RETURNED: pName\n");
    	PyErr_Print();
    }
    PyObject* pModule = PyImport_Import(pName);
    
    // KUNAL
    if (pModule == NULL)
    {
    	printf("NULL RETURNED: pModule \n");
    	PyErr_Print();
    }
    Py_DECREF(pName); // finished with this string so release reference
    std::cout << "Loaded python module" << std::endl; 

    std::cout << "Loading functions from module" << std::endl;
    // PyObject* pcollection_func = PyObject_GetAttrString(pModule, "collection_func");
    PyObject* pml_predict = PyObject_GetAttrString(pModule, "output_ml_error_prediction");


      // stringstream ml_pred(" ");
      // ml_pred<< "ml_predict.dat";
      
      // ofstream ml_pred_log(ml_pred.str().c_str(), ios::app);
      // ml_pred_log.precision(16);
      // ml_pred_log.setf(ios::scientific, ios::floatfield);






    Py_DECREF(pModule); // finished with this module so release reference
    std::cout << "Loaded functions" << std::endl;
 
    // std::cout << "Called python data collection function successfully"<<std::endl;
    // Py_Finalize();
   //  int NX = 50;
   // double u[NX+2];
   //  double x;

   //    for (int i = 1; i < NX+1; i++)
   //    {
   //        x = (double) (i-1)/(NX) * 2.0 * M_PI;
   //        u[i] = sin(x);
   //    }

   //    // Handle the ghost points: periodic boundary conditions
   //    u[0] = u[NX];
   //    u[NX+1] = u[1];

   //  collect_data(pml_predict,u, NX);
   
   
  // KUNAL
  // Batch-predictor 
  // vector<vector<double>> Double2DVector(rows,vector<double>(cols, 0.0));
  vector<vector<double>> Double2DVector;
  /*for(int i = 0;i < ne;i++)
  {
    Double2DVector.push_back(input_data);
  }
  */
  

  for(int i = 0; i < ne; i++)
  {
  // int bc_marker = 0;
    ML_ElementData<D, COMP> & ml_elidata = *ml_eldata[i];
    // KUNAL
    // if(ml_elidata.nf!=ml_elidata.nf_bc) continue;
    int ndof_w = ml_elidata.ndof_w;
    // Taking the quadrature nodes from the first face
    int fcnr = ml_elidata.faces[0];
    ML_FacetData<D, COMP> & ml_fd_1 = *ml_fadata[fcnr];
    int nip_1 = ml_fd_1.nip;

    int input_size =  (ml_elidata.ndof_w*(D+1) + nip_1*(D+1)*ml_elidata.nf)*COMP + (D*(D+1))/2 + Model::NumParam;
    vector<double> input_data(input_size);
    // setting the element parameters // Hard coded for 2D
    input_data[0] = ml_elidata.vol;
    input_data[1] = ml_elidata.beta;
    input_data[2] = ml_elidata.theta;
    
    int offset = 3;
    
    for(int pr = 0; pr<Model::NumParam;pr++)
    {
      input_data[offset+pr] = param(pr);
    }
    
    offset = offset+Model::NumParam;
    // Generating the input data for solution and gradient
    for (int ll = 0; ll < COMP; ++ll)
    {
      for (int pp = 0; pp < ndof_w; ++pp)
      {
        input_data[offset+pp+ll*ndof_w] = ml_elidata.vecW[pp+ll*ndof_w];
      }
    }

    offset = offset + ndof_w*(COMP);
      // Generating the input data for solution and gradient
    for (int ll = 0; ll < COMP; ++ll)
    {
      for (int pp = 0; pp < ndof_w; ++pp)
      {
        for(int dd = 0; dd < D; ++dd)
        {
          input_data[offset+pp+ndof_w*(dd+D*ll)] = ml_elidata.vecQ[pp+ndof_w*(dd+D*ll)];
        }
      }
    }

    offset = offset + ndof_w*D*COMP;
    for(int j = 0 ; j < ml_elidata.nf ; j ++)
    {

      int fcnr = ml_elidata.faces[j];
      ML_FacetData<D, COMP> & ml_fd = *ml_fadata[fcnr];
      int nip = ml_fd.nip;
        
        if (!fadata[fcnr]) continue;
        FacetData<D, COMP> & fd = *fadata[fcnr];
        if (fd.ndof_l == 0) continue; // skip boundary faces
        if (fd.bcnr == -2) continue; // zero-measure face

      // err_log << ml_fd.nip << endl;
      for (int qp = 0; qp < nip; ++qp) 
      {
        // Jump in w on quarature points
        double *jump_w = &ml_fd.w_jump[qp*COMP];
        for(int ll=0;ll<COMP;ll++)
        {
          input_data[offset+qp*COMP+ll] = jump_w[ll];
        }
      }

      offset = offset + nip*COMP;

      for (int qp = 0; qp < nip; ++qp) 
      {
            // Jump in q on quadrature points
            double *jump_q = &ml_fd.q_jump[qp * COMP * D];

            for(int ll=0;ll<COMP;ll++)
            {
              for(int dd=0;dd<D;dd++)
              {
                input_data[offset + qp * COMP * D + dd+D*ll] = jump_q[dd+D*ll];
              }
            }
      }
      offset = offset + nip*D*COMP;
    }

    // Call python here
    vector<double> vec = input_data;
    /*for (int m = 0; m < 60;m++)
    {
      cout << "vec["<<m<<"]: "<<vec[m]<< endl;
    }*/
    Double2DVector.push_back(vec);
  }
  
  vector<double> flattenedData;
  // This is a range-based for loop.
  for (const auto& row : Double2DVector) 
  {
    flattenedData.insert(flattenedData.end(), row.begin(), row.end());
  }
  
  npy_intp dims[2] = {static_cast<npy_intp>(Double2DVector.size()), static_cast<npy_intp>(Double2DVector[0].size())};
  
      // Get the dimensions of the 2D vector
    // int nRows = Double2DVector.size();
    // int nCols = (nRows > 0) ? Double2DVector[0].size() : 0;
    // Create a NumPy array with the same dimensions
    // npy_intp dims[2] = {nRows, nCols};
    
    PyObject* array_2d = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, flattenedData.data());

    // PyObject* array_2d = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, (double*)Double2DVector.data());
    // Construct the argument tuple
    PyObject* pArgs = PyTuple_New(1);
    // placing array_2d in first position of the tuple
    PyTuple_SetItem(pArgs, 0, array_2d);
    // pml_predict (a pointer) values with pArgs argument is being casted to pValue 
    PyArrayObject* pValue = (PyArrayObject*)PyObject_CallObject(pml_predict, pArgs);

    PyErr_Print();
    int len{PyArray_SHAPE(pValue)[0]};
    int len2{PyArray_SHAPE(pValue)[1]};
    
    cout<<"len: " <<len <<endl;
    cout<<"len2: " <<len2 <<endl;
  
  cout << "Double2DVector: Filled" << endl;

  vector<double> testing(ne, 0.0);

  for(int i = 0; i < ne; i++)
  {
    // cout<<"Element Number: " <<i <<endl;
    
    ML_DeployData<D, COMP> & ml_depldata = *ml_deploydata[i];

    // Mehul
    // int nip = ml_depldata.nip;
    int nip = 1; // there is no nip for adjoint based error

      // double scal = pow(ml_elidata.h, q_norm*(max_order+1)); //h^r*(p+1)
      
      for(int qq=0;qq<nip;qq++)
      {
        // double* current = (double*) PyArray_GETPTR2(pValue, i, qq);
        double* current = (double*) PyArray_GETPTR2(pValue, i, qq);
        double tempcurrent = *current;
        double err_exp_loc = exp(tempcurrent) ;//* scal;
        ml_depldata.w[qq] = err_exp_loc;
        // ml_depldata.w[qq] = pow(err_exp_loc, 1.0/q_norm);
      }
  }//end of loop over elements

  Py_Finalize();
}
