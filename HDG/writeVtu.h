extern"C" {
  void table_delaunay_wrapper_(int*, double*, int*, int*);
}

// So far only 2d!

template<int D, int COMP, class Model>
  void UnifyingFramework<D, COMP, Model>
::SaveVtu(const string & filename, int order, const Solution & sol, const Space & sp, LocalHeap & lh) const
{
      string fname_dat(filename);
      fname_dat.append(".vtk");
      fstream output;
      output.open(fname_dat.c_str(),ios::out);

      //char res_out[160];
      //output << "TITLE = DG2D" << endl;
      //output << "VARIABLES = X Y ";
      output << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << endl;
      output << "\t<UnstructuredGrid>" << endl;

      /*
      if (D == 3)
          output << "Z ";

      for (int ll = 0; ll < COMP; ++ll)
      {
          sprintf(res_out, "W%i ", ll+1);
          output << res_out;
      }

      output << "Ma P ";
      */

      // Calculate a triangulation of the unit triangle
      int nl = 2 * (max_order+1);
      int np = 0.5 * (nl + 2) * (nl + 1);

      vector<double> xll(2*np);

      int l = 0;
      for (int j = 0; j <= nl; j++)
      {
          double rl1 = 1.*j / (1. * nl);
          for (int k = 0; k <= nl; k++)
          {
              double rl2 = 1. * k / (1. * nl);
              if ( (rl2 + rl1) < 1. + 1e-9)
              {
                  double rl3 = 1. - rl2 - rl1;
                  xll[2*l] = rl2;
                  xll[2*l+1] = rl3;
                  l++;
              }
          }
      }

      vector<int> nlc(9*np);
      int ntr = 0;

      table_delaunay_wrapper_(&np, &xll[0], &ntr, &nlc[0]);

      // Now get the shape functions evaluated at these help nodes
      FlatVector<> shape(ndof_w_max, lh);

      // Loop over the boundary faces and write data out. Each triangle is a zone
      for (int i = 0; i < ne; i++) {

          HeapReset hr(lh);
          ElementData<D, COMP> & elidata = *eldata[i];

          const ScalarFiniteElement<D> & fel_w = dynamic_cast<const ScalarFiniteElement<D>&> (sp.fes_w -> GetFE(i, lh));

          ElementTransformation & eltrans = ma->GetTrafo(i, lh);

          Array<int> vnums;
          vnums = ma->GetElVertices(ElementId(VOL,i));

          ELEMENT_TYPE eltype = fel_w.ElementType();
          bool admissible = true;

          for (int j = 0; j < np; j++)
	  {
              // x-coordinate: xll[2*j], y-coordinate: xll[2*j + 1]
              IntegrationPoint xi(xll[2*j], xll[2*j+1]);
              MappedIntegrationPoint<D,D> sip (xi, eltrans);

              if (abs(sip.GetPoint()[0]) > tecplot_x || abs(sip.GetPoint()[1]) > tecplot_y)
              {
                  admissible = false;
                  break;
              }

              if (D == 3) {
                  if (abs(sip.GetPoint()[2]) > tecplot_z) {
                      admissible = false;
                      break;
                  }
              }
          }

          if (!admissible) continue;

          char res_out[160];
          //sprintf(res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", np, ntr);
          sprintf( res_out, "\t\t<Piece NumberOfPoints=\"%i\" NumberOfCells=\"%i\">", np, ntr);
          output << res_out << endl;


          // necessary for vtu output in XML structure
          stringstream  vtuCoords;
          stringstream  vtuDensity;
          stringstream  vtuEnergy;
          stringstream  vtuMomentun;
          stringstream  vtuPressure;
          stringstream  vtuMach;

          for (int j = 0; j < np; j++)
          {

              // x-coordinate: xll[2*j], y-coordinate: xll[2*j + 1]
              IntegrationPoint xi(xll[2*j], xll[2*j+1]);
              MappedIntegrationPoint<D,D> sip (xi, eltrans);

              fel_w.CalcShape (xi, shape);

              SpatialParams<D> sparam;

              for (int dd = 0; dd < D; ++dd)
                  sparam.pos(dd) = sip.GetPoint()[dd];

              Vec<COMP> w = 0.;

              for (int ll = 0; ll < COMP; ++ll)
              {
                  for (int pp = 0; pp < elidata.ndof_w; ++pp)
                  {
                      int new_pp = ndof_converter_tet[elidata.order][pp];
                      w(ll) += shape(new_pp) * sol.vecW[COMP*elidata.offset_w+pp+ll*elidata.ndof_w];
                  }
              }

              if (D == 2)
              {
                  //sprintf(res_out, "%10.10f %10.10f ", pos(0), pos(1));
                  sprintf(res_out, "%10.10f %10.10f %1.1f", sparam.pos(0), sparam.pos(1), 0.0);
              }else
              {
                  sprintf(res_out, "%10.10f %10.10f %10.10f ", sparam.pos(0), sparam.pos(1), sparam.pos(2));
              }

              //output << res_out;
              vtuCoords << "\t\t\t\t" << res_out << endl;

              vtuDensity << "\t\t\t\t" << w(0) << endl;
              vtuMomentun << "\t\t\t\t";
              for( int ll = 0; ll < D; ll++ )
              {
                  sprintf( res_out, "%10.10f ", w(ll+1) );
                  vtuMomentun << res_out;
              }
              vtuMomentun << endl;
              /*
              for (int ll = 0; ll < COMP; ++ll)
              {
                  sprintf(res_out, "%10.10f ", w(ll));
                  output << res_out;
              }
              */
              vtuEnergy << "\t\t\t\t" << w( COMP - 1 ) << endl;

              Vec<D> m;
              for (int dd = 0; dd < D; ++dd)
                  m(dd) = w(dd+1);

              double gm1 = 1.4 - 1.;

              Vec<D> U = 1. / w(0) * m;

              double U2 = 0.;
              for (int dd = 0; dd < D; ++dd)
                  U2 += U(dd) * U(dd);

              double p = gm1 * (w(D+1) - 0.5 * w(0) * U2);
              double c2 = 1.4 * p / w(0);

              double ma = sqrt(U2 / c2);

              //sprintf(res_out, "%10.10f %10.10f ", ma, p);
              //output << res_out;
              //output << endl;
              vtuPressure << "\t\t\t\t" << p << endl;
              vtuMach << "\t\t\t\t" << ma << endl;
          }

          output << "\t\t<Points>" << endl;
          output << "\t\t\t<DataArray type=\"Float64\" NumberOfComponents=\"3\" Format=\"ascii\">" << endl;
          output << vtuCoords.str();
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t</Points>" << endl;

          output << "\t\t<PointData Scalars=\"scalars\">" << endl;
          output << "\t\t\t<DataArray type=\"Float64\" Name=\"density\" Format=\"ascii\">" << endl;
          output << vtuDensity.str();
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t\t<DataArray type=\"Float64\" Name=\"momentum\" NumberOfComponents=\""<<D<<"\" Format=\"ascii\">" << endl;
          output << vtuMomentun.str();
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t\t<DataArray type=\"Float64\" Name=\"energy\" Format=\"ascii\">" << endl;
          output << vtuEnergy.str();
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t\t<DataArray type=\"Float64\" Name=\"pressure\" Format=\"ascii\">" << endl;
          output << vtuPressure.str();
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t\t<DataArray type=\"Float64\" Name=\"Mach\" Format=\"ascii\">" << endl;
          output << vtuMach.str();
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t</PointData>" << endl;

          output << "\t\t<Cells>" << endl;
          stringstream  vtuConn;
          stringstream  vtuOffset;
          stringstream  vtuType;
          int   vtuElementType = 5;
          for (int j = 0; j < ntr; j++)
          {
              //sprintf(res_out, "%i %i %i", nlc[3*j], nlc[3*j+1], nlc[3*j+2]);
              sprintf(res_out, "\t\t\t\t\t%i %i %i\n", nlc[3*j] - 1 , nlc[3*j+1] - 1, nlc[3*j+2] - 1);
              vtuConn << res_out;
              sprintf(res_out, "%i ", j*3 + 3);
              vtuOffset << res_out;
              sprintf(res_out, "%i ", vtuElementType);
              vtuType << res_out;
              //output << res_out << endl;
          }
          output << "\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << endl;
          output << vtuConn.str();
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" << endl;
          output << "\t\t\t\t" << vtuOffset.str() << endl;
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t\t<DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">" << endl;
          output << "\t\t\t\t" << vtuType.str() << endl;
          output << "\t\t\t</DataArray>" << endl;
          output << "\t\t</Cells>" << endl;
          output << "\t\t</Piece>" << endl << endl;

      }
      output << "\t</UnstructuredGrid>" << endl;
      output << "</VTKFile>" << endl;
      output.close();
  }
