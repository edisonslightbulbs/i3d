#ifndef IQR_H
#define IQR_H

#include <vector>
#include "point.h"

namespace iqr {

std::vector<Point> denoise(std::vector<Point> points);

}
#endif /* IQR_H */
