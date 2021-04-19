#ifndef CONTEXT_H
#define CONTEXT_H

#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "point.h"

class Context {
public:
    std::mutex m_mutex;
    std::shared_mutex s_mutex;

    std::vector<uint8_t> m_rgb;
    std::vector<float> m_points;

    int m_numPoints {};
    std::shared_ptr<int> sptr_numClusters = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_points = nullptr;

    explicit Context(std::vector<Point>& points, int& numPoints)
    {
        /** block threads from accessing
         *  resources during recording */
        std::lock_guard<std::mutex> lck(m_mutex);
        sptr_points = std::make_shared<std::vector<Point>>(numPoints * 3);
        *sptr_points = points;
        m_rgb = std::vector<uint8_t>(numPoints * 3);
        m_points = std::vector<float>(numPoints * 3);
    }

    Context() = default;

    Context(const Context& other)
    {
        m_numPoints = other.m_numPoints;
        sptr_points = other.sptr_points;
    }

    Context& operator=(Context other)
    {
        /** block threads from accessing
         *  resources during recording */
        std::lock_guard<std::mutex> lck(m_mutex);
        std::swap(sptr_points, other.sptr_points);
        return *this;
    }

    [[nodiscard]] std::shared_ptr<std::vector<Point>> getContext()
    {
        /** allow multiple threads to read pcl color */
        std::shared_lock lock(s_mutex);
        return sptr_points;
    }

    void updateContext(std::vector<Point>& points)
    {
        /** block threads from accessing
         *  resources during recording */
        std::lock_guard<std::mutex> lck(m_mutex);
        sptr_points = std::make_shared<std::vector<Point>>(points);
    }

    void updateNumClusters(const int& clusters)
    {
        /** block threads from accessing
         *  resources during recording */
        std::lock_guard<std::mutex> lck(m_mutex);
        sptr_numClusters = std::make_shared<int>(clusters);
    }

    void castPcl()
    { // todo: put me in a dedicated thread
        std::vector<Point> clone(m_numPoints * 3);

        {
            /** block threads from accessing
             *  resources during recording */
            std::lock_guard<std::mutex> lck(m_mutex);
            clone = *getContext(); // todo this size is small than numPoints
        }

        std::string delimiter = " ";
        for (auto& point : clone) {
            m_points.push_back(point.m_xyz[0]);
            m_points.push_back(point.m_xyz[1]);
            m_points.push_back(point.m_xyz[2]);

            /** parse to color-string to color-int */
            std::string s = point.m_clusterColor;
            s.erase(0, 1);
            size_t last = 0;
            size_t next = 0;
            int rgb[3];
            int index = 0;
            while ((next = s.find(delimiter, last)) != std::string::npos) {
                int colorVal = std::stoi(s.substr(last, next - last));
                rgb[index] = colorVal;
                last = next + 1;
                index++;
            }
            int colorVal = std::stoi(s.substr(last));
            rgb[index] = colorVal;

            m_rgb.push_back(rgb[0]);
            m_rgb.push_back(rgb[1]);
            m_rgb.push_back(rgb[2]);
        }
    }

    std::pair<std::vector<float>, std::vector<uint8_t>> getCastPcl()
    {
        /** allow multiple threads to read pcl color */
        std::shared_lock lock(s_mutex);
        return { m_points, m_rgb };
    }
};
#endif /* CONTEXT_H */
