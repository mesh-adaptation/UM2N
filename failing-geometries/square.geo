Point(1) = {0.0, 0.0, 0, 0.1};
Extrude {0, 1, 0} {
  Point{1};
}
Delete {
  Line{1};
}
Delete {
  Point{2};
}
Extrude {1, 0, 0} {
  Point{1};
}
Extrude {0, 1, 0} {
  Line{1};
}
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Physical Surface(6) = {5};
