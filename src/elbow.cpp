#include <algorithm>
#include <cmath>

#include "point.h"
#include "elbow.h"
#include "logger.h"
#include "timer.h"

float elbow::find(std::vector<float>& x)
{
    Timer timer;
    /** sort in descending order */
    std::sort(x.begin(), x.end(), std::greater<>());

    /** evaluate successive angular distances */
    std::vector<std::pair<float, float>> angles;
    for (int i = 5; i < x.size() - 5; i++) {
        float angle = std::asin(x[i + 5] - x[i]);
        angles.emplace_back(x[i], angle);
    }

    /** find the max second derivative */
    float sd = 0;
    for (int i = 1; i < angles.size() - 1; i++) {
        float d1 = angles[i - 1].second - angles[i].second;
        float d2 = angles[i + 1].second - angles[i].second;

        /** evaluate second derivative-based central difference */
        //float candidate = d1 + d2 - 2 * angles[i].second;
        float candidate = d1 + d2 /2;

        /** elbow corresponds to distance with the max second derivative */
        if (candidate > sd) {
            sd = angles[i].first;
        }
    }
    LOG(INFO) << timer.getDuration() << " ms: epsilon estimation runtime";

    return sd;
}
