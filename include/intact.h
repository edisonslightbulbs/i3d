#ifndef INTACT_H
#define INTACT_H

#include <memory>
#include <mutex>
#include <vector>

#include "kinect.h"
#include "point.h"

namespace intact {
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
void segment(Kinect& kinect);

void render(Kinect& kinect);
}
#endif /* INTACT_H */
