#ifndef INTACT_H
#define INTACT_H

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "kinect.h"
#include "point.h"

class Intact {

public:
    /** number of raw point cloud points */
    int m_numPoints;

    /** raw point cloud */
    std::shared_ptr<std::vector<float>> sptr_raw = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_rawColor = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_rawPoints = nullptr;

    /** raw point cloud segment */
    std::shared_ptr<std::vector<float>> sptr_segment = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_segmentColor = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_segmentPoints = nullptr;

    /** segment region */
    std::shared_ptr<std::vector<float>> sptr_region = nullptr;
    std::shared_ptr<std::vector<uint8_t>> sptr_regionColor = nullptr;
    std::shared_ptr<std::vector<Point>> sptr_clusterPoints = nullptr;

    /** mutual exclusion */
    std::mutex m_mutex;
    std::shared_mutex s_mutex;

    /** flow-control semaphores */
    std::shared_ptr<bool> sptr_run;
    std::shared_ptr<bool> sptr_stop;
    std::shared_ptr<bool> sptr_isIntactReady;
    std::shared_ptr<bool> sptr_isContextClustered;
    std::shared_ptr<bool> sptr_isEpsilonComputed;
    std::shared_ptr<bool> sptr_isContextSegmented;

    std::pair<Point, Point> m_boundary {};

    void setBoundary(std::pair<Point, Point>& boundary)
    {
        std::lock_guard<std::mutex> lck(m_mutex);
        m_boundary = boundary;
    }

    /** initialize API */
    explicit Intact(int& numPoints)
        : m_numPoints(numPoints)
    {
        /** initialize infinite boundary */
        Point lower(__FLT_MIN__, __FLT_MIN__, __FLT_MIN__);
        Point upper(__FLT_MAX__, __FLT_MAX__, __FLT_MAX__);
        m_boundary = { lower, upper };

        sptr_run = std::make_shared<bool>(false);
        sptr_stop = std::make_shared<bool>(false);
        sptr_isEpsilonComputed = std::make_shared<bool>(false);
        sptr_isContextClustered = std::make_shared<bool>(false);
        sptr_isContextSegmented = std::make_shared<bool>(false);
        sptr_isIntactReady = std::make_shared<bool>(false);

        sptr_raw = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_rawColor = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);

        sptr_segment = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_segmentColor
            = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);

        sptr_region = std::make_shared<std::vector<float>>(m_numPoints * 3);
        sptr_regionColor
            = std::make_shared<std::vector<uint8_t>>(m_numPoints * 3);

        sptr_rawPoints = std::make_shared<std::vector<Point>>(m_numPoints * 3);
        sptr_segmentPoints
            = std::make_shared<std::vector<Point>>(m_numPoints * 3);
        sptr_clusterPoints
            = std::make_shared<std::vector<Point>>(m_numPoints * 3);
    }

    /**
     * segment
     *   Segments context.
     *
     * @param sptr_kinect
     *   Kinect device.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    void segment(std::shared_ptr<Kinect>& sptr_kinect,
        std::shared_ptr<Intact>& sptr_intact);
    /**
     * render
     *   Renders point cloud.
     *
     * @param sptr_kinect
     *   Kinect device.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void render(std::shared_ptr<Kinect>& sptr_kinect,
        std::shared_ptr<Intact>& sptr_intact);
    /**
     * cluster
     *   Clusters segmented context.
     *
     * @param E
     *   Epsilon parameter.
     *
     * @param N
     *   Number of epsilon-neighbourhood neighbours.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void cluster(
        const float& E, const int& N, std::shared_ptr<Intact>& sptr_intact);

    /**
     * estimateEpsilon
     *   Estimates size of epsilon neighbourhood using knn.
     *
     * @param K
     *   K parameter.
     *
     * @param sptr_intact
     *   Instance of API call.
     */
    static void estimateEpsilon(
        const int& K, std::shared_ptr<Intact>& sptr_intact);

    /**
     * adapt
     *  Adapts std::vector<Point> to point cloud.
     */
    void adapt();

    /** Setters. */
    void setRawPoints(const std::vector<Point>& points);

    /** Getters. */
    std::shared_ptr<std::vector<Point>> getPoints();

    std::shared_ptr<std::vector<float>> getRaw();

    std::shared_ptr<std::vector<float>> getSegment();

    std::shared_ptr<std::vector<float>> getRegion();

    std::shared_ptr<std::vector<uint8_t>> getRawColor();

    std::shared_ptr<std::vector<uint8_t>> getSegmentColor();

    std::shared_ptr<std::vector<uint8_t>> getRegionColor();

    /** Thread-safe-semaphores */
    bool isIntactReady();

    bool isSegmented();

    bool isClustered();

    void raiseSegFlag();

    void raiseClustFlag();

    void raiseEpsilonFlag();

    bool isEpsilonComputed();

    // /** initialize boundless interaction context */

    void buildPcl(k4a_image_t pcl, k4a_image_t transformedImage)
    {
        auto* data = (int16_t*)(void*)k4a_image_get_buffer(pcl);
        uint8_t* color = k4a_image_get_buffer(transformedImage);

        std::vector<float> raw(m_numPoints * 3);
        std::vector<float> segment(m_numPoints * 3);
        std::vector<uint8_t> rawColor(m_numPoints * 3);
        std::vector<uint8_t> segmentColor(m_numPoints * 3);

        /** n.b., kinect colors reversed! */
        for (int i = 0; i < m_numPoints; i++) {
            if (data[3 * i + 2] == 0) {
                raw[3 * i + 0] = 0.0f;
                raw[3 * i + 1] = 0.0f;
                raw[3 * i + 2] = 0.0f;
                rawColor[3 * i + 2] = color[4 * i + 0];
                rawColor[3 * i + 1] = color[4 * i + 1];
                rawColor[3 * i + 0] = color[4 * i + 2];
                continue;
            }
            raw[3 * i + 0] = (float)data[3 * i + 0];
            raw[3 * i + 1] = (float)data[3 * i + 1];
            raw[3 * i + 2] = (float)data[3 * i + 2];
            rawColor[3 * i + 2] = color[4 * i + 0];
            rawColor[3 * i + 1] = color[4 * i + 1];
            rawColor[3 * i + 0] = color[4 * i + 2];

            /** filter boundary */
            if (m_boundary.second.m_xyz[2] == __FLT_MAX__
                || m_boundary.first.m_xyz[2] == __FLT_MIN__) {
                continue;
            }

            if ((float)data[3 * i + 0] > m_boundary.second.m_xyz[0]
                || (float)data[3 * i + 0] < m_boundary.first.m_xyz[0]
                || (float)data[3 * i + 1] > m_boundary.second.m_xyz[1]
                || (float)data[3 * i + 1] < m_boundary.first.m_xyz[1]
                || (float)data[3 * i + 2] > m_boundary.second.m_xyz[2]
                || (float)data[3 * i + 2] < m_boundary.first.m_xyz[2]) {
                continue;
            }
            segment[3 * i + 0] = (float)data[3 * i + 0];
            segment[3 * i + 1] = (float)data[3 * i + 1];
            segment[3 * i + 2] = (float)data[3 * i + 2];
            segmentColor[3 * i + 2] = color[4 * i + 0];
            segmentColor[3 * i + 1] = color[4 * i + 1];
            segmentColor[3 * i + 0] = color[4 * i + 2];
        }

        /** thread-safe update */
        std::lock_guard<std::mutex> lck(m_mutex);
        *sptr_raw = raw;
        *sptr_rawColor = rawColor;

        /** Iff context boundaries not set: default to raw point cloud */
        if (m_boundary.second.m_xyz[2] == __FLT_MAX__
            || m_boundary.first.m_xyz[2] == __FLT_MIN__) {
            *sptr_segment = *sptr_raw;
            *sptr_segmentColor = *sptr_rawColor;
        } else {
            *sptr_segment = segment;
            *sptr_segmentColor = segmentColor;
        }
    }

    void raiseIntactReadyFlag();

    int getNumPoints();

    bool isRun();

    void raiseRunFlag();

    void raiseStopFlag();

    bool isStop();

    void stop();

    void setSegment(
        std::pair<std::vector<float>, std::vector<uint8_t>>& segment);

    void setClusterPoints(const std::vector<Point>& points);

    void setSegmentPoints(const std::vector<Point>& points);
};
#endif /* INTACT_H */
