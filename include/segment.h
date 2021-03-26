#ifndef SEGMENT_H
#define SEGMENT_H

#include "point.h"
#include <vector>

namespace segment {
/**
 * cut
 *   Identifies and isolates tabletop interaction context segment.
 *
 * @param points
 *   Unorganized 3D point-cloud representation of tabletop environment.
 *
 * @retval
 *    Segment of tabletop interaction context
 */
std::vector<Point> cut(std::vector<Point>& points);
}
#endif /* SEGMENT_H */
