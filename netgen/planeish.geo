algebraic3d

solid myplane = (cylinder(0,0,0; 0,0,10 ; 5)
  and cone (0,0,0; 10; 0,-0,-3; 2))
  or cylinder (-10,0,5; 10,-0,5; 3);

solid bplane = orthobrick(-12,-12,-3;12,12,10) and myplane;
# cube with hole
#algebraic3d
#solid cubehole = orthobrick (0, 0, 0; 1, 1, 1)
#and not cylinder (0.5, 0.5, 0; 0.5, 0.5, 1; 0.1);
#tlo cubehole;

tlo bplane;
