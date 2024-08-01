dx = 0.05;
l3 = 0.2;  // length of channel
// remaining distance on either side of channel:
l1 = (1- l3)/2;
l2 = l1;
w3 = 0.2; // width of channel
w1 = (1-w3)/2;
w2 = w1;
Point(1) = {0.0, 0.0, 0, dx};
Point(2) = {l1,  0.0, 0, dx};
Point(3) = {l1,  w1,  0, dx};
Point(4) = {l1+l3, w1, 0, dx};
Point(5) = {l1+l3, 0, 0, dx};
Point(6) = {1, 0, 0, dx};
Point(7) = {1, 1, 0, dx};
Point(8) = {l1+l3, 1, 0, dx};
Point(9) = {l1+l3, w1+w3, 0, dx};
Point(10) = {l1, w1+w3, 0, dx};
Point(11) = {l1, 1, 0, dx};
Point(12) = {0, 1, 0, dx};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 1};



Curve Loop(1) = {12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

Plane Surface(1) = {1};
// Left and Right:
Physical Curve(1) = {12};
Physical Curve(2) = {6};
// Bottom left, right and middle:
Physical Curve(3) = {1};
Physical Curve(4) = {5};
Physical Curve(5) = {3};
// Top left, right and middle:
Physical Curve(6) = {11};
Physical Curve(7) = {7};
Physical Curve(8) = {9};
// Middle vertical sections: bottom left and right and top left and right resp.
Physical Curve(9) = {2};
Physical Curve(10) = {4};
Physical Curve(11) = {10};
Physical Curve(12) = {8};

Physical Surface(1) = {1};
