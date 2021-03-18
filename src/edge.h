#ifndef EDGE_H
#define EDGE_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "point.h"
#include "io.h"
#include "slope.h"

namespace edge {

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

    std::vector<Point> segment(std::vector<Point> &points) {
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
        float xMaxEdge = slope::largest(xVectors.second);
        //float xMinEdge = slope::largest(xVectors.first);
        // float yMaxEdge = slope::largest(yVectors.second);
        // float yMinEdge = slope::largest(yVectors.first);

        /** edge boundaries */
        std::cout << "x max: " << xMaxEdge << std::endl;
        //std::cout << "x min: " << xMinEdge << " --- " << "x max: " << xMaxEdge << std::endl;
        // std::cout << "y min: " << yMinEdge << " --- " << "y max: " << yMaxEdge << std::endl;

        /** use the distance boundary as clipping criteria */
        std::vector<Point> context;

        return context;
    }
}
#endif /* EDGE_H */

// std::vector<float> xMinVector = descend(xVectors.first);
// std::vector<float> xMaxVector = descend(xVectors.second);
// std::vector<float> yMinVector = descend(yVectors.first);
// std::vector<float> yMaxVector = descend(yVectors.second);
// const std::string xMin = io::pwd() + "/build/bin/xMin.csv";
// const std::string xMax = io::pwd() + "/build/bin/xMax.csv";
// const std::string yMin = io::pwd() + "/build/bin/yMin.csv";
// const std::string yMax = io::pwd() + "/build/bin/yMax.csv";
// io::write_val(xMinVector, xMin);
// io::write_val(xMaxVector, xMax);
// io::write_val(yMinVector, yMin);
// io::write_val(yMaxVector, yMax);
