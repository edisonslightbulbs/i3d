#ifndef SEGMENT_H
#define SEGMENT_H

#include "point.h"
#include <vector>

namespace segment {
std::vector<Point> partition(std::vector<Point>& points);
}
#endif /* SEGMENT_H */
