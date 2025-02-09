algebraic3d
#
# Example with two sub-domains: 
#
solid sph = sphere (0.0, 0.0, 0.0; 1.) -bc=2; # -maxh=0.5;
solid far = sphere (0.0, 0.0, 0.0; 100.) -bc=1;
solid sym = plane (0, 0, 0; 0, -1, 0) -bc=3;
solid rest = far and not sph and not sym;

tlo rest -col=[0,0,1];
#tlo sph -col=[1,0,0];
#tlo far -col=[0,1,0] -transparent;

