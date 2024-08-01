basin_x = .5;
basin_y = .3;

headland_x_scale = 0.2;
headland_y = 0.1;

element_size_coarse = 0.02;

Point(1) = {0, 0, 0, element_size_coarse};
Point(2) = {basin_x, 0, 0, element_size_coarse};


// Generate nodes for the headland
res = 400;
For k In {0:res:1}
    x = basin_x/res*k;
    b = 0.01;
    y = basin_y - headland_y*Exp(-0.5*((headland_x_scale*(x-basin_x/2))/b)^2);
	Point(10+k) = {x, y, 0, element_size_coarse*(y-(basin_y-headland_y))/headland_y + 0.1*element_size_coarse*(basin_y-y)/headland_y};
EndFor

// Generate lines for the headland

BSpline(100) = { 10 : res+10 };

Line(101) = {10, 1};
Line(102) = {1, 2};
Line(103) = {2, res+10};
Line Loop(104) = {100, -103, -102, -101};

Plane Surface(111) = {104};
Physical Surface(112) = {111};
Physical Line(1) = {101};
Physical Line(2) = {103};
Physical Line(3) = {102};
Physical Line(4) = {100};
