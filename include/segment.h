#ifndef SCALPEL_H
#define SCALPEL_H

#include "point.h"
#include <vector>

namespace segment {

std::vector<float> descend(std::vector<float> axisVector);

std::vector<Point> partition(std::vector<Point>& points);

    std::pair<std::vector<float>, std::vector<float>> split( std::vector<float> axisVector);
}
#endif /* SCALPEL_H */
