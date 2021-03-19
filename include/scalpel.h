#ifndef SCALPEL_H
#define SCALPEL_H

#include "point.h"
#include <vector>

namespace scalpel {

std::vector<float> descend(std::vector<float> axisVector);
std::pair<std::vector<float>, std::vector<float>> split(
    std::vector<float> axisVector);
std::vector<Point> segment(const std::vector<Point>& points);

}
#endif /* SCALPEL_H */
