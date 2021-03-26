#ifndef EDGE_H
#define EDGE_H

#include "point.h"
#include <vector>

namespace edge {

/**
 * detect
 *   Detects the edge of a tabletop surface.
 *
 * @param  points
 *   Coarse segment of tabletop interaction context.
 *
 * @retval
 *   Final segment of tabletop interaction context.
 */
std::vector<Point> detect(std::vector<Point>& points);
}
#endif /* EDGE_H */
