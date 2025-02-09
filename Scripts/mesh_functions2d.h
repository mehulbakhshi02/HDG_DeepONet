// double det(double x1, double y1, double z1, double x2, double y2, double z2, double x3, double y3, double z3, double x4, double y4, double z4)
// {
// 	double val = x1*y2*z3-x1*y3*z2-x2*y1*z3+x2*y3*z1+x3*y1*z2-x3*y2*z1-x1*y2*z4+x1*y4*z2+x2*y1*z4-x2*y4*z1-x4*y1*z2+x4*y2*z1+x1*y3*z4-x1*y4*z3-x3*y1*z4+x3*y4*z1+x4*y1*z3-x4*y3*z1-x2*y3*z4+x2*y4*z3+x3*y2*z4-x3*y4*z2-x4*y2*z3+x4*y3*z2;
// 	// if(fabs(val)<1e-15)
// 	// 	val = 0.0;
// 	return val;
// }

// double area(double x1, double y1, double x2, double y2, double x3, double y3)
// {
// 	double Area = 0.5 *(-y2*x3 + y1*(-x2 + x3) + x1*(y2 - y3) + x2*y3);
// 	return Area;
// }
// double sgn(double x) {
// 	if(x>0)
// 		return 1;
// 	if(x<0)
// 		return -1;
// 	if(x==0)
// 		return 0;
//     // return (x > 0) - (x < 0);
// }

double is_inside(vector<double> pos_tri, vector<double> pos_loc, vector<double>tri_order)
{
	double x1 = pos_tri[0];
	double y1 = pos_tri[1];

	double x2 = pos_tri[2];
	double y2 = pos_tri[3];

	double x3 = pos_tri[4];
	double y3 = pos_tri[5];

	double x = pos_loc[0];
	double y = pos_loc[1];

	double v1 = tri_order[0];
	double v2 = tri_order[1];
	double v3 = tri_order[2];

	double denom = (y2-y3)*(x1-x3)+(x3-x2)*(y1-y3);
	double w1 = ((y2-y3)*(x-x3)+(x3-x2)*(y-y3))/denom;
	double w2 = ((y3-y1)*(x-x3)+(x1-x3)*(y-y3))/denom;
	double w3 = 1. - w1 - w2;
	double val = 0;
	// cout<<x1<<" "<<y1<<endl<<x2<<" "<<y2<<endl<<x3<<" "<<y3<<endl<<endl;
	// cout<<x<<" "<<y<<endl<<endl;
	// cout<<w1<<" "<<w2<<" "<<w3<<endl<<endl;
	if(abs(w1)<1e-15)	
	{
		w1 = 0;
	}
	if(abs(w2)<1e-15)	
	{
		w2 = 0;
	}
	if(abs(w3)<1e-15)	
	{
		w3 = 0;
	}

	if(w1 < 0 || w2 < 0 || w3 < 0) 
		val = 0;
	else
		val = (v1*w1+v2*w2+v3*w3);
	return val;

	// double v1 = tet_order[0];
	// double v2 = tet_order[1];
	// double v3 = tet_order[2];
	// double v4 = tet_order[3];

	// double d0 = det(x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4);
	// double d1 = det(x, y, z, x2, y2, z2, x3, y3, z3, x4, y4, z4);
	// double d2 = det(x1, y1, z1, x, y, z, x3, y3, z3, x4, y4, z4);
	// double d3 = det(x1, y1, z1, x2, y2, z2, x, y, z, x4, y4, z4);
	// double d4 = det(x1, y1, z1, x2, y2, z2, x3, y3, z3, x, y, z);
	// // cout<<d1<<" "<<d2<<" "<<d3<<" "<<d4<<endl;
	// if(fabs(d0-(d1+d2+d3+d4))>1e-13)
	// {
	// 	cout<<"Something is wrong "<< fabs(d0-(d1+d2+d3+d4))<<endl;
	// }
	// double sgn0 = sgn(d0);
	// double sgn1 = sgn(d1);
	// double sgn2 = sgn(d2);
	// double sgn3 = sgn(d3);
	// double sgn4 = sgn(d4);


	// double value = 0.0;
	// // cout<<sgn0<<" "<<sgn1<<" "<<sgn2<<" "<<sgn3<<" "<<sgn4<<endl;
	// if(sgn0 == sgn1 && sgn1 == sgn2 && sgn2 == sgn3 && sgn3 == sgn4)
	// {	
	// 	value = (d1*v1+d2*v2+d3*v3+d4*v4)/d0;
	// 	return value;
	// }
	// else if(sgn1 == 0)
	// {	
	// 	double p0 = area(x2, y2, z2, x3, y3, z3, x4, y4, z4);
	// 	double p1 = area(x, y, z, x2, y2, z2, x3, y3, z3);
	// 	double p2 = area(x, y, z, x2, y2, z2, x4, y4, z4);
	// 	double p3 = area(x, y, z, x4, y4, z4, x3, y3, z3);
	// 	if(fabs((p1+p2+p3)-p0)<1e-13)
	// 	{		
	// 		value = (p1*v4+p2*v3+p3*v2)/p0;
	// 	}
	// 	// else
	// 	// 	cout<<"Area is inconsistent 1 " <<(p1+p2+p3)-p0 <<endl;
	// 	return value;
	// }
	// else if(sgn2 == 0)
	// {	
	// 	double p0 = area(x1, y1, z1, x3, y3, z3, x4, y4, z4);
	// 	double p1 = area(x, y, z, x1, y1, z1, x3, y3, z3);
	// 	double p2 = area(x, y, z, x1, y1, z1, x4, y4, z4);
	// 	double p3 = area(x, y, z, x4, y4, z4, x3, y3, z3);
	// 	if(fabs((p1+p2+p3)-p0)<1e-13)
	// 	{
	// 		value = (p1*v4+p2*v3+p3*v1)/p0;
	// 	}
	// 	// else
	// 	// 	cout<<"Area is inconsistent 2 " <<(p1+p2+p3)-p0 <<endl;
	// 	return value;
	// }
	// else if(sgn3 == 0)
	// {	
	// 	double p0 = area(x1, y1, z1, x2, y2, z2, x4, y4, z4);
	// 	double p1 = area(x, y, z, x1, y1, z1, x2, y2, z2);
	// 	double p2 = area(x, y, z, x1, y1, z1, x4, y4, z4);
	// 	double p3 = area(x, y, z, x4, y4, z4, x2, y2, z2);
	// 	if(fabs((p1+p2+p3)-p0)<1e-13)
	// 	{		
	// 		value = (p1*v4+p2*v2+p3*v1)/p0;
	// 	}
	// 	// else
	// 	// 	cout<<"Area is inconsistent 3 " <<((p1+p2+p3)-p0) <<endl;
	// 	return value;
	
	// }
	// else if(sgn4 == 0)
	// {	
	// 	double p0 = area(x1, y1, z1, x2, y2, z2, x3, y3, z3);
	// 	double p1 = area(x, y, z, x1, y1, z1, x2, y2, z2);
	// 	double p2 = area(x, y, z, x1, y1, z1, x3, y3, z3);
	// 	double p3 = area(x, y, z, x3, y3, z3, x2, y2, z2);
	// 	if(fabs((p1+p2+p3)-p0)<1e-13)
	// 	{		
	// 		value = (p1*v3+p2*v2+p3*v1)/p0;
	// 	}
	// 	// else
	// 	// 	cout<<"Area is inconsistent 4 " <<(p1+p2+p3)-p0 <<endl;
	// 	return value;

	// }
	// else
	// {	
	// 	value = -1.0;
	// 	return value;
	// }
}