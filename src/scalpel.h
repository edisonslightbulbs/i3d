#ifndef EDGE_H
#define EDGE_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "point.h"
#include "io.h"
#include "elbow.h"

namespace scalpel {

    std::vector<float> descend (std::vector<float> axisVector){
        std::sort(axisVector.begin(), axisVector.end(), std::greater<>());
        return axisVector;
    }

    std::pair<std::vector<float>, std::vector<float>> split(std::vector<float> axisVector){
        /** secondDerivative axis center */
        int index = (int)axisVector.size()/2;

        /** split at center */
        std::vector<float> minVector(axisVector.begin(), axisVector.begin() + index);
        std::vector<float> maxVector(axisVector.begin() + index, axisVector.end());

        return { minVector, maxVector };
    }

    std::vector<Point> segment(const std::vector<Point> &points) {
        Timer timer;
        std::vector<float> xVector;
        std::vector<float> yVector;

        for (const auto& point : points){
            xVector.push_back(point.m_x);
            yVector.push_back(point.m_y);
        }

        /** sort in ascending order */
        std::sort(xVector.begin(), xVector.end());
        std::sort(yVector.begin(), yVector.end());

        /** split at axis means*/
        std::pair<std::vector<float>, std::vector<float>> xVectors = split(xVector);
        std::pair<std::vector<float>, std::vector<float>> yVectors = split(yVector);

        /** edge boundaries */
        float xMinEdge = elbow::analyzeAxis(xVectors.first);
        float xMaxEdge = elbow::analyzeAxis(xVectors.second);
        float yMinEdge = elbow::analyzeAxis(yVectors.first);
        float yMaxEdge = elbow::analyzeAxis(yVectors.second);


        std::vector<Point> proposal;
        for (auto& point : points){
            if (point.m_x < xMaxEdge && point.m_x > xMinEdge){
                if (point.m_y < yMaxEdge && point.m_y > yMinEdge) {
                    proposal.push_back(point);
                }
            }
        }
        std::vector<Point> context = lda::analyze(proposal);
        io::write_ply(context);


        /** use the distance boundary as clipping criteria */

        return context;
    }
}
#endif /* EDGE_H */
