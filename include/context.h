#ifndef CONTEXT_H
#define CONTEXT_H

#include <iostream>
#include <memory>
#include <vector>

#include "point.h"

class Context {
public:
    std::shared_ptr<std::vector<Point>> sptr_points = nullptr;
    std::shared_ptr<int> sptr_numClusters = nullptr;

    explicit Context(std::vector<Point>& points)
    {
        sptr_points = std::make_shared<std::vector<Point>>(points);
    }

    Context() = default;

    Context(const Context& other) { sptr_points = other.sptr_points; }

    Context& operator=(Context other)
    {
        std::swap(sptr_points, other.sptr_points);
        return *this;
    }

    [[nodiscard]] std::shared_ptr<std::vector<Point>> getContext() const
    {
        return sptr_points;
    }

    void updateContext(std::vector<Point>& points)
    {
        sptr_points = std::make_shared<std::vector<Point>>(points);
    }

    void updateNumClusters(const int& clusters)
    {
        sptr_numClusters = std::make_shared<int>(clusters);
    }
};
#endif /* CONTEXT_H */
