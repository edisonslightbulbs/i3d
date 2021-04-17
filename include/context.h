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
    std::vector<Point> m_points;

    explicit Context(std::vector<Point>& points)
        : m_points(points)
    {
    }

    Context() = default;

    Context(const Context& other) { m_points = other.m_points; }

    Context& operator=(Context other)
    {
        std::swap(m_points, other.m_points);
        return *this;
    }

    [[nodiscard]] std::vector<Point> getContext() const { return m_points; }
};
#endif /* CONTEXT_H */
