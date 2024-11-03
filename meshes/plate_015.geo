// far field resolution
dx1 = 0.015;
// resolution at cylinder
dx2 = 0.015;

// 4 corners of rectangle
Point(1) = {0, 0, 0, dx1};
Point(2) = {2.2, 0, 0, dx1};
Point(3) = {0, 0.41, 0, dx1};
Point(4) = {2.2, 0.41, 0, dx1};

// define the rectange
Line(1) = {3, 4};
Line(2) = {4, 2};
Line(3) = {2, 1};
Line(4) = {1, 3};

Line Loop(9) = {1, 2, 3, 4};
Plane Surface(9) = {9};

// top(1) and bottom(3)
Physical Line(1) = {1, 3};
// inflow  
Physical Line(2) = {4};
// outflow
Physical Line(3) = {2};

// Whole domain id
Physical Surface(9) = {9};