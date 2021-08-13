#include <Eigen/Dense>
#include <chrono>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <thread>
#include <utility>

#include "dbscan.h"
#include "i3d.h"
#include "i3dmacros.hpp"
#include "i3dutils.h"
#include "kinect.h"
#include "outliers.h"
#include "region.h"
#include "svd.h"

i3d::i3d()
{
    Point i3dMaxBound(SHRT_MAX, SHRT_MAX, SHRT_MAX);
    Point i3dMinBound(SHRT_MIN, SHRT_MIN, SHRT_MIN);

    m_segBoundary = { i3dMaxBound, i3dMinBound };

    sptr_run = std::make_shared<bool>(false);
    sptr_stop = std::make_shared<bool>(false);
    sptr_clustered = std::make_shared<bool>(false);
    sptr_segmented = std::make_shared<bool>(false);
    sptr_proposalReady = std::make_shared<bool>(false);
    sptr_pCloudReady = std::make_shared<bool>(false);
    sptr_resourcesReady = std::make_shared<bool>(false);
}

bool i3d::isRun()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_run;
}

void i3d::raiseRunFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_run = true;
}

bool i3d::isSensorReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_resourcesReady;
}

void i3d::raiseSensorReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_resourcesReady = true;
}

bool i3d::isSegmented()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_segmented;
}

bool i3d::isProposalReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_proposalReady;
}

void i3d::raiseProposalReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_proposalReady = true;
}

void i3d::raiseSegmentedFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_segmented = true;
}

bool i3d::isPCloudReady()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_pCloudReady;
}

void i3d::raisePCloudReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_pCloudReady = true;
}

bool i3d::isClustered()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_clustered;
}

void i3d::raiseClusteredFlag()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_clustered = true;
}

bool i3d::isStop()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    return *sptr_stop;
}

void i3d::stop()
{
    std::lock_guard<std::mutex> lck(m_flagMutex);
    *sptr_run = false;
    *sptr_stop = true;
}

void i3d::setRGBAHeight(const int& height)
{
    std::lock_guard<std::mutex> lck(m_dImgMutex);
    m_imgHeight = height;
}

void i3d::setRGBAWidth(const int& width)
{
    std::lock_guard<std::mutex> lck(m_dImgMutex);
    m_imgWidth = width;
}

int i3d::getRGBAWidth()
{
    std::lock_guard<std::mutex> lck(m_dImgMutex);
    return m_imgWidth;
}

int i3d::getRGBAHeight()
{
    std::lock_guard<std::mutex> lck(m_dImgMutex);
    return m_imgHeight;
}

void i3d::setDHeight(const int& height)
{
    std::lock_guard<std::mutex> lck(m_dDepthMutex);
    m_depthHeight = height;
}

void i3d::setDWidth(const int& width)
{
    std::lock_guard<std::mutex> lck(m_dDepthMutex);
    m_depthWidth = width;
}

int i3d::getDWidth()
{
    std::lock_guard<std::mutex> lck(m_dDepthMutex);
    return m_depthWidth;
}

int i3d::getDHeight()
{
    std::lock_guard<std::mutex> lck(m_dDepthMutex);
    return m_depthHeight;
}

void i3d::setPCloud2x2Bin(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloud2x2BinMutex);
    sptr_pCloud2x2Bin = std::make_shared<std::vector<Point>>(points);
}

std::shared_ptr<std::vector<Point>> i3d::getPCloud2x2Bin()
{
    std::lock_guard<std::mutex> lck(m_pCloud2x2BinMutex);
    return sptr_pCloud2x2Bin;
}

void i3d::setPCloudSeg2x2Bin(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudSeg2x2BinMutex);
    sptr_pCloudSeg2x2Bin = std::make_shared<std::vector<Point>>(points);
}

__attribute__((unused)) std::shared_ptr<std::vector<Point>>
i3d::getPCloudSeg2x2Bin()
{
    std::lock_guard<std::mutex> lck(m_pCloudSeg2x2BinMutex);
    return sptr_pCloudSeg2x2Bin;
}

void i3d::setBGRAData(uint8_t* bgra)
{
    std::lock_guard<std::mutex> lck(m_BGRARawMutex);
    sptr_BGRARaw = std::make_shared<uint8_t*>(bgra);
}

std::shared_ptr<uint8_t*> i3d::getBGRAData()
{
    std::lock_guard<std::mutex> lck(m_BGRARawMutex);
    return sptr_BGRARaw;
}

void i3d::setC2DBGRAData(uint8_t* bgra)
{
    std::lock_guard<std::mutex> lck(m_BGRARawMutex);
    sptr_BGRAC2DRaw = std::make_shared<uint8_t*>(bgra);
}

std::shared_ptr<uint8_t*> i3d::getC2DBGRAData()
{
    std::lock_guard<std::mutex> lck(m_BGRARawMutex);
    return sptr_BGRAC2DRaw;
}

void i3d::setXYZData(int16_t* xyz)
{
    std::lock_guard<std::mutex> lck(m_XYZRawMutex);
    sptr_XYZRaw = std::make_shared<int16_t*>(xyz);
}

std::shared_ptr<int16_t*> i3d::getXYZData()
{
    std::lock_guard<std::mutex> lck(m_XYZRawMutex);
    return sptr_XYZRaw;
}

void i3d::setDepthData(uint16_t* xyz) // todo: cross check
{
    std::lock_guard<std::mutex> lck(m_depthMutex);
    sptr_XYZDepth = std::make_shared<uint16_t*>(xyz);
}

std::shared_ptr<uint16_t*> i3d::getDepthData()
{
    std::lock_guard<std::mutex> lck(m_depthMutex);
    return sptr_XYZDepth;
}

void i3d::setXYTableData(k4a_float2_t* ptr_table)
{
    std::lock_guard<std::mutex> lck(m_XYTableMutex);
    ptr_XYTableData = ptr_table;
}

k4a_float2_t* i3d::getXYTableData()
{
    std::lock_guard<std::mutex> lck(m_XYTableMutex);
    return ptr_XYTableData;
}

void i3d::setPCloud(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudMutex);
    sptr_pCloud = std::make_shared<std::vector<Point>>(points);
}

__attribute__((unused)) std::shared_ptr<std::vector<Point>> i3d::getPCloud()
{
    std::lock_guard<std::mutex> lck(m_pCloudMutex);
    return sptr_pCloud;
}

void i3d::setPCloudSeg(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_pCloudSegMutex);
    sptr_pCloudSeg = std::make_shared<std::vector<Point>>(points);
}

std::shared_ptr<std::vector<Point>> i3d::getPCloudSeg()
{
    std::lock_guard<std::mutex> lck(m_pCloudSegMutex);
    return sptr_pCloudSeg;
}

void i3d::setOptimizedPCloudSeg(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_optimizedPCloudSegMutex);
    sptr_optimizedPCloudSeg = std::make_shared<std::vector<Point>>(points);
}

std::shared_ptr<std::vector<Point>> i3d::getOptimizedPCloudSeg()
{
    std::lock_guard<std::mutex> lck(m_optimizedPCloudSegMutex);
    return sptr_optimizedPCloudSeg;
}

void i3d::setRGBA(const std::vector<uint8_t>& rgba)
{
    std::lock_guard<std::mutex> lck(m_RGBAMutex);
    sptr_RGBA = std::make_shared<std::vector<uint8_t>>(rgba);
}

std::shared_ptr<std::vector<uint8_t>> i3d::getRGBA()
{
    std::lock_guard<std::mutex> lck(m_RGBAMutex);
    return sptr_RGBA;
}

void i3d::setSegBoundary(std::pair<Point, Point>& boundary)
{
    std::lock_guard<std::mutex> lck(m_boundaryMutex);
    m_segBoundary = boundary;
}

std::pair<Point, Point> i3d::getSegBoundary()
{
    std::lock_guard<std::mutex> lck(m_boundaryMutex);
    return m_segBoundary;
}

void i3d::setXYZ(const std::vector<int16_t>& xyz)
{
    std::lock_guard<std::mutex> lck(m_XYZMutex);
    sptr_XYZ = std::make_shared<std::vector<int16_t>>(xyz);
}

std::shared_ptr<std::vector<int16_t>> i3d::getXYZ()
{
    std::lock_guard<std::mutex> lck(m_XYZMutex);
    return sptr_XYZ;
}

void i3d::setXYZSeg(const std::vector<int16_t>& xyz)
{
    std::lock_guard<std::mutex> lck(m_XYZSegMutex);
    sptr_XYZSeg = std::make_shared<std::vector<int16_t>>(xyz);
}

std::shared_ptr<std::vector<int16_t>> i3d::getPCloudSegFrame()
{
    std::lock_guard<std::mutex> lck(m_XYZSegMutex);
    return sptr_XYZSeg;
}

void i3d::setRGBASeg(const std::vector<uint8_t>& rgba)
{
    std::lock_guard<std::mutex> lck(m_RGBASegMutex);
    sptr_RGBASeg = std::make_shared<std::vector<uint8_t>>(rgba);
}

std::shared_ptr<std::vector<uint8_t>> i3d::getRGBASeg()
{
    std::lock_guard<std::mutex> lck(m_RGBASegMutex);
    return sptr_RGBASeg;
}

void i3d::setBGRA(const std::vector<uint8_t>& bgra)
{
    std::lock_guard<std::mutex> lck(m_BGRAMutex);
    sptr_BGRA = std::make_shared<std::vector<uint8_t>>(bgra);
}

std::shared_ptr<std::vector<uint8_t>> i3d::getBGRA()
{
    std::lock_guard<std::mutex> lck(m_BGRAMutex);
    return sptr_BGRA;
}

void i3d::setPCloudClusters(const t_clusters& clusters)
{
    std::lock_guard<std::mutex> lck(m_pCloudClusterMutex);
    sptr_pCloudClusters = std::make_shared<t_clusters>(clusters);
}

std::shared_ptr<i3d::t_clusters> i3d::getPCloudClusters()
{
    std::lock_guard<std::mutex> lck(m_pCloudClusterMutex);
    return sptr_pCloudClusters;
}

void i3d::setColorizedClusters(
    const std::pair<int16_t*, uint8_t*>& colorizedClusters)
{
    std::lock_guard<std::mutex> lck(m_colorizedClustersMutex);
    sptr_colorizedClusters
        = std::make_shared<std::pair<int16_t*, uint8_t*>>(colorizedClusters);
}

std::shared_ptr<std::pair<int16_t*, uint8_t*>> i3d::getColorizedClusters()
{
    std::lock_guard<std::mutex> lck(m_colorizedClustersMutex);
    return sptr_colorizedClusters;
}

void i3d::buildPCloud(std::shared_ptr<i3d>& sptr_i3d)
{
#if BUILD_POINTCLOUD == 1
    SLEEP_UNTIL_SENSOR_DATA_READY
    START
    int w = sptr_i3d->getDWidth();
    int h = sptr_i3d->getDHeight();

    uint16_t* ptr_depthData;
    k4a_float2_t* ptr_tableData;

    std::vector<Point> pCloud(w * h);
    while (RUN) {
        START_TIMER
        ptr_depthData = *sptr_i3d->getDepthData();
        ptr_tableData = sptr_i3d->getXYTableData();

        int index = 0;
        for (int i = 0; i < w * h; i++) {
            if (i3dutils::invalid(i, ptr_tableData, ptr_depthData)) {
                continue;
            }
            int16_t x = ptr_tableData[i].xy.x * (float)ptr_depthData[i];
            int16_t y = ptr_tableData[i].xy.y * (float)ptr_depthData[i];
            int16_t z = (float)ptr_depthData[i];
            Point point(x, y, z);
            pCloud[index] = point;
            index++;
        }
        std::vector<Point> optimizedPCloud(
            pCloud.begin(), pCloud.begin() + index);

        sptr_i3d->setPCloud2x2Bin(optimizedPCloud);
        RAISE_POINTCLOUD_READY_FLAG
        STOP_TIMER(" build point cloud thread: runtime @ ")
    }
#endif
}

void i3d::proposeRegion(std::shared_ptr<i3d>& sptr_i3d)
{
#if PROPOSE_REGION == 1
    WAIT_FOR_FAST_POINTCLOUD
    START
    while (RUN) {
        START_TIMER
        std::vector<Point> pCloud = *sptr_i3d->getPCloud2x2Bin();
        std::vector<Point> pCloudSeg = region::segment(pCloud);
        std::pair<Point, Point> boundary = i3dutils::queryBoundary(pCloudSeg);
        sptr_i3d->setPCloudSeg2x2Bin(pCloudSeg);
        sptr_i3d->setSegBoundary(boundary);
        POINTCLOUD_READY
        STOP_TIMER(" propose region thread: runtime @ ")
    }
#endif
}

void i3d::segmentRegion(std::shared_ptr<i3d>& sptr_i3d)
{
#if SEGMENT_REGION == 1
    WAIT_FOR_PROPOSAL
    START
    int w = sptr_i3d->getDWidth();
    int h = sptr_i3d->getDHeight();

    std::vector<Point> pCloud(w * h);
    std::vector<Point> pCloudSeg(w * h);

    std::vector<int16_t> pCloudFrame(w * h * 3);
    std::vector<uint8_t> imgFrame_RGBA(w * h * 4);
    std::vector<uint8_t> imgFrame_BGRA(w * h * 4);

    std::vector<int16_t> pCloudSegFrame(w * h * 3);
    std::vector<uint8_t> imgSegFrame_RGBA(w * h * 4);
    std::vector<uint8_t> imgSegFrame_BGRA(w * h * 4);

    uint8_t* ptr_sensorC2DData;
    int16_t* ptr_sensorPCloudData;

    while (RUN) {
        START_TIMER
        ptr_sensorPCloudData = *sptr_i3d->getXYZData();
        ptr_sensorC2DData = *sptr_i3d->getC2DBGRAData();

        int index = 0;
        for (int i = 0; i < w * h; i++) {
            Point point;

            // create unsegmented assets
            if (i3dutils::invalid(i, ptr_sensorPCloudData, ptr_sensorC2DData)) {
                i3dutils::addXYZ(i, pCloudFrame);
                i3dutils::addPixel_RGBA(i, imgFrame_RGBA);
                i3dutils::addPixel_BGRA(i, imgFrame_BGRA);
            } else {
                i3dutils::addXYZ(i, pCloudFrame, ptr_sensorPCloudData);
                i3dutils::addPixel_RGBA(i, imgFrame_RGBA, ptr_sensorC2DData);
                i3dutils::addPixel_BGRA(i, imgFrame_BGRA, ptr_sensorC2DData);
            }
            i3dutils::adapt(i, point, pCloudFrame, imgFrame_BGRA);
            pCloud[i] = point;

            // create segmented assets
            if (i3dutils::inSegment(i, pCloudFrame,
                    sptr_i3d->getSegBoundary().first,
                    sptr_i3d->getSegBoundary().second)) {
                i3dutils::addXYZ(i, pCloudSegFrame, ptr_sensorPCloudData);
                i3dutils::addPixel_RGBA(i, imgSegFrame_RGBA, ptr_sensorC2DData);
                i3dutils::addPixel_BGRA(i, imgSegFrame_BGRA, ptr_sensorC2DData);
                // point.m_id = index; // test
                pCloudSeg[index] = point;
                index++;
            } else {
                i3dutils::addXYZ(i, pCloudSegFrame);
                i3dutils::addPixel_RGBA(i, imgSegFrame_RGBA);
                i3dutils::addPixel_BGRA(i, imgSegFrame_BGRA);
            }
        }
        std::vector<Point> optimizedPCloudSeg(
            pCloudSeg.begin(), pCloudSeg.begin() + index);

        sptr_i3d->setPCloud(pCloud);
        sptr_i3d->setPCloudSeg(pCloudSeg);
        sptr_i3d->setXYZ(pCloudFrame);
        sptr_i3d->setRGBA(imgFrame_RGBA);
        sptr_i3d->setBGRA(imgFrame_BGRA);
        sptr_i3d->setXYZSeg(pCloudSegFrame);
        sptr_i3d->setRGBASeg(imgSegFrame_RGBA);
        sptr_i3d->setOptimizedPCloudSeg(optimizedPCloudSeg);
        PROPOSAL_READY
        STOP_TIMER(" frame region thread: runtime @ ")
    }
#endif
}

void i3d::clusterRegion(
    const float& epsilon, const int& minPoints, std::shared_ptr<i3d>& sptr_i3d)
{
#if CLUSTER_REGION == 1
    WAIT_FOR_SEGMENT
    START
    while (RUN) {
        START_TIMER
        std::vector<Point> points = *sptr_i3d->getOptimizedPCloudSeg();

        // dbscan::cluster clusters candidate interaction regions
        //   from the segmented tabletop surface. It returns a
        //   collection of clusters, more specifically, a list of
        //   index collections. The indexes correspond to clustered
        //   3D points specified as part of function arguments.
        //
        auto indexClusters = dbscan::cluster(points, epsilon, minPoints);

        // sorting the clusters is one possible approach to
        //   extracting the vacant or non occupied space on
        //   the tabletop surface, this is of cause assuming
        //   non occupied space is prevalent on the tabletop
        //   environment.
        //
        std::sort(indexClusters.begin(), indexClusters.end(),
            [](const std::vector<unsigned long>& a,
                const std::vector<unsigned long>& b) {
                return a.size() > b.size();
            });

        sptr_i3d->setPCloudClusters({ points, indexClusters });
        CLUSTERS_READY
        STOP_TIMER(" cluster region thread: runtime @ ")
    }
#endif
}
