#ifndef EDGE_H
#define EDGE_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "point.h"
#include "io.h"
#include "slope.h"

namespace scalpel {

    std::vector<float> descend (std::vector<float> axisVector){
        std::sort(axisVector.begin(), axisVector.end(), std::greater<>());
        return axisVector;
    }

    std::pair<std::vector<float>, std::vector<float>> split(std::vector<float> axisVector){
        /** find axis center */
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

        /** find edges */
        //float xMinEdge = slope::largest(xVectors.first);
         float xMaxEdge = slope::largest(xVectors.second);

        // float yMinEdge = slope::largest(yVectors.first);
        // float yMaxEdge = slope::largest(yVectors.second);

        /** edge boundaries */
        //std::cout << "x max: " << xMaxEdge << std::endl;
        //std::cout << "x min: " << xMinEdge << " --- " << "x max: " << xMaxEdge << std::endl;
        // std::cout << "y min: " << yMinEdge << " --- " << "y max: " << yMaxEdge << std::endl;

        //io::write_ply(points);

        std::vector<Point> context;
        for (auto& point : points){
            if (point.m_x < xMaxEdge){
                context.push_back(point);
            }
        }
        io::write_ply(context);


        /** use the distance boundary as clipping criteria */

        return context;
    }
}
#endif /* EDGE_H */
