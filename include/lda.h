#ifndef LDA_H
#define LDA_H

#include <vector>

#include "point.h"
#include "logger.h"
#include "timer.h"

namespace lda {

    std::vector<Point> analyze(std::vector<Point>& points);

}
#endif /* LDA_H */
