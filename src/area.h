#ifndef AREA_H
#define AREA_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "point.h"
#include "io.h"

namespace area {

    int queryIndex(std::vector<float>& axis, const float& key){
        auto itr = std::find(axis.begin(), axis.end(), key);
        return std::distance(axis.begin(), itr);
    }

    std::pair<std::vector<float>, std::vector<float>> splitAxis(std::vector<float> axis){
        /** find axis center */
        float sum = std::accumulate(axis.begin(), axis.end(), 0.0);
        float center = sum / axis.size();

        /** add center to axis */
        axis.push_back(center);

        /** sort axis */
        std::sort(axis.begin(), axis.end());

        /** query index of emplaced center */
        int splitIndex = queryIndex(axis, center);

        /** split at center */
        std::vector<float> low(axis.begin(), axis.begin() + splitIndex);
        std::vector<float> high(axis.begin() + splitIndex, axis.end());

        return {low, high};
    }

    std::vector<Point> estimate(std::vector<Point> &points) {
        Timer timer;
        std::vector<Point> mut_points = points;

        /** find within-point variance using the centroid */
        Point centroid = Point::centroid(mut_points);
        for (auto &point : mut_points) {
            float distance = centroid.distance(point);
            point.m_distance.second = distance;
        }
        Point::sort(mut_points);

        std::vector<float> x;
        std::vector<float> y;
        std::vector<float> dist;
        for (const auto& point : mut_points){
            x.push_back(point.m_x);
            y.push_back(point.m_y);
            dist.push_back(point.m_distance.second);
        }
        /** Clip at axis cliffs */
        std::pair<std::vector<float>, std::vector<float>> xSplit = splitAxis(x);
        std::pair<std::vector<float>, std::vector<float>> ySplit = splitAxis(y);
        float xMinClip = elbow::find(xSplit.first);
        float xMaxClip = elbow::find(xSplit.second);
        float yMinClip = elbow::find(ySplit.first);
        float yMaxClip = elbow::find(ySplit.second);
        float distClip = elbow::find(y);

        /** region of interest boundaries */
        std::cout << "x min: " << xMinClip << " --- " << "x max: " << xMaxClip << std::endl;
        std::cout << "y min: " << yMinClip << " --- " << "y max: " << yMaxClip << std::endl;
        std::cout << "max dist: " << distClip << yMaxClip << std::endl;

        /** use the distance boundary as clipping criteria */
        std::vector<Point> area;
        for (const auto& point : mut_points){
            if(point.m_distance.second < distClip){
                area.push_back(point);
            }
        }

        /** resources for external LDA analysis */
        io::write_points_2d(mut_points);
        io::write_axis(xSplit.first);
        io::write_ply_2d(area);

        return area;
    }
}
#endif /* AREA_H */
