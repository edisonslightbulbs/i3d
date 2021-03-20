#ifndef PROBE_H
#define PROBE_H

#include <vector>

#include "elbow.h"
#include "point.h"

namespace edge {

    std::vector<float> descend(std::vector<float> axisVector)
    {
        std::sort(axisVector.begin(), axisVector.end(), std::greater<>());
        return axisVector;
    }

    std::pair<std::vector<float>, std::vector<float>> split(
            std::vector<float> axisVector)
    {
        /** secondDerivative axis center */
        int index = (int)axisVector.size() / 2;

        /** split at center */
        std::vector<float> minVector(
                axisVector.begin(), axisVector.begin() + index);
        std::vector<float> maxVector(axisVector.begin() + index, axisVector.end());

        return { minVector, maxVector };
    }

    std::vector<Point> find(const std::vector<Point>& points) {
        Timer timer;
        std::vector<float> xVector;
        std::vector<float> yVector;

        for (const auto &point : points) {
            xVector.push_back(point.m_x);
            yVector.push_back(point.m_y);
        }

        /** sort in ascending order */
        std::sort(xVector.begin(), xVector.end());
        std::sort(yVector.begin(), yVector.end());

        /** split at axis means*/
        std::pair<std::vector<float>, std::vector<float>> xVectors = split(xVector);
        std::pair<std::vector<float>, std::vector<float>> yVectors = split(yVector);

        /** define surface edges */
        float xMinEdge = elbow::analyze(xVectors.first);
        float xMaxEdge = elbow::analyze(xVectors.second);
        float yMinEdge = elbow::analyze(yVectors.first);
        float yMaxEdge = elbow::analyze(yVectors.second);

        float maxXEdge = std::max(xMaxEdge, xMinEdge);
        float maxYEdge = std::max(yMaxEdge, yMinEdge);
        float maxEdge = std::max(maxXEdge, maxYEdge);

        /** partition context proposal */
        std::vector<Point> proposal;
        for (auto &point : points) {
            if (point.m_x < xMaxEdge && point.m_x > xMinEdge) {
                if (point.m_y < yMaxEdge && point.m_y > yMinEdge) {
                    proposal.push_back(point);
                }
            }
        }
        return proposal;
    }
}
#endif /* PROBE_H */
