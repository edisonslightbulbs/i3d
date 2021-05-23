#include <chrono>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <utility>

#include "dbscan.h"
#include "intact.h"
#include "kinect.h"
#include "macros.hpp"
#include "region.h"
#include "utils.hpp"
#include "viewer.h"
#include "yolov5.h"

Intact::Intact(int& numPts)
    : m_numPoints(numPts)
    , m_pclsize(numPts * 3)
    , m_imgsize(numPts * 4)
{
    Point intactMaxBound(SHRT_MAX, SHRT_MAX, SHRT_MAX);
    Point intactMinBound(SHRT_MIN, SHRT_MIN, SHRT_MIN);
    m_intactBoundary = { intactMaxBound, intactMinBound };

    sptr_run = std::make_shared<bool>(false);
    sptr_stop = std::make_shared<bool>(false);
    sptr_isClustered = std::make_shared<bool>(false);
    sptr_isSegmented = std::make_shared<bool>(false);
    sptr_isKinectReady = std::make_shared<bool>(false);
    sptr_isIntactReady = std::make_shared<bool>(false);

    sptr_refinedPoints = std::make_shared<std::vector<Point>>(m_pclsize);
    sptr_unrefinedPoints = std::make_shared<std::vector<Point>>(m_pclsize);
    sptr_segmentedPoints = std::make_shared<std::vector<Point>>(m_pclsize);
    sptr_chromaBkgdPoints = std::make_shared<std::vector<Point>>(m_pclsize);
}

///////////////////////////////////////////////////////////////////////////////
//                semaphores for asynchronous threads
///////////////////////////////////////////////////////////////////////////////

void Intact::raiseSegmentedFlag()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    *sptr_isSegmented = true;
}

void Intact::raiseClusteredFlag()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    *sptr_isClustered = true;
}

void Intact::raiseRunFlag()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    *sptr_run = true;
}

void Intact::raiseStopFlag()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    *sptr_stop = true;
}

bool Intact::isRun()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    return *sptr_run;
}

bool Intact::isStop()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    return *sptr_stop;
}

void Intact::raiseKinectReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    *sptr_isKinectReady = true;
}

void Intact::raiseIntactReadyFlag()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    *sptr_isIntactReady = true;
}

bool Intact::isKinectReady()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    return *sptr_isKinectReady;
}

bool Intact::isIntactReady()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    return *sptr_isIntactReady;
}

bool Intact::isSegmented()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    return *sptr_isSegmented;
}

bool Intact::isClustered()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    return *sptr_isClustered;
}

void Intact::stop()
{
    std::lock_guard<std::mutex> lck(m_semaphoreMutex);
    *sptr_run = false;
}

///////////////////////////////////////////////////////////////////////////////
//                     depth image width and height
///////////////////////////////////////////////////////////////////////////////

void Intact::setDepthImgHeight(const int& height)
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    m_depthHeight = height;
}

void Intact::setDepthImgWidth(const int& width)
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    m_depthWidth = width;
}

int Intact::getDepthImgWidth()
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    return m_depthWidth;
}

int Intact::getDepthImgHeight()
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    return m_depthHeight;
}

///////////////////////////////////////////////////////////////////////////////
//                              segment boundaries
///////////////////////////////////////////////////////////////////////////////

void Intact::setIntactBoundary(std::pair<Point, Point>& boundary)
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    m_intactBoundary = boundary;
}

std::pair<Point, Point> Intact::getIntactBoundary()
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    return m_intactBoundary;
}

///////////////////////////////////////////////////////////////////////////////
//                              shared point-cloud points
///////////////////////////////////////////////////////////////////////////////

void Intact::setUnrefinedPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    *sptr_unrefinedPoints = points;
}

std::shared_ptr<std::vector<Point>> Intact::getUnrefinedPoints()
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    return sptr_unrefinedPoints;
}

void Intact::setRefinedPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    *sptr_refinedPoints = points;
}

std::shared_ptr<std::vector<Point>> Intact::getRefinedPoints()
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    return sptr_refinedPoints;
}

///////////////////////////////////////////////////////////////////////////////
//                              shared sensor resources
///////////////////////////////////////////////////////////////////////////////

void Intact::setSensorPcl(int16_t* ptr_pcl)
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    sptr_sensorPcl = std::make_shared<int16_t*>(ptr_pcl);
}

std::shared_ptr<int16_t*> Intact::getSensorPcl()
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    return sptr_sensorPcl;
}

void Intact::setSensorImg_GL(uint8_t* ptr_img)
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    sptr_sensorImg_GL = std::make_shared<uint8_t*>(ptr_img);
}

std::shared_ptr<uint8_t*> Intact::getSensorImg_GL()
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    return sptr_sensorImg_GL;
}

void Intact::setSensorImg_CV(uint8_t* ptr_img)
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    sptr_sensorImg_CV = std::make_shared<uint8_t*>(ptr_img);
}

std::shared_ptr<uint8_t*> Intact::getSensorImg_CV()
{
    std::lock_guard<std::mutex> lck(m_sensorMutex);
    return sptr_sensorImg_CV;
}

///////////////////////////////////////////////////////////////////////////////
//                              shared api resources
///////////////////////////////////////////////////////////////////////////////

void Intact::setSegmentPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    *sptr_segmentedPoints = points;
}

std::shared_ptr<std::vector<Point>> Intact::getSegmentPoints()
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    return sptr_segmentedPoints;
}

void Intact::setIntactPcl(int16_t* ptr_pcl)
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    sptr_intactPcl = std::make_shared<int16_t*>(ptr_pcl);
}

std::shared_ptr<int16_t*> Intact::getIntactPcl()
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    return sptr_intactPcl;
}

void Intact::setIntactImg_GL(uint8_t* ptr_img)
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    sptr_intactImg_GL = std::make_shared<uint8_t*>(ptr_img);
}

std::shared_ptr<uint8_t*> Intact::getIntactImg_GL()
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    return sptr_intactImg_GL;
}

void Intact::setIntactImg_CV(uint8_t* ptr_img)
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    sptr_intactImg_CV = std::make_shared<uint8_t*>(ptr_img);
}

std::shared_ptr<uint8_t*> Intact::getIntactImg_CV()
{
    std::lock_guard<std::mutex> lck(m_intactMutex);
    return sptr_intactImg_CV;
}

///////////////////////////////////////////////////////////////////////////////
//                           chroma background
///////////////////////////////////////////////////////////////////////////////

void Intact::setChromaBkgdPoints(const std::vector<Point>& points)
{
    std::lock_guard<std::mutex> lck(m_bkgdMutex);
    *sptr_chromaBkgdPoints = points;
}

std::shared_ptr<std::vector<Point>> Intact::getChromaBkgdPoints()
{
    std::lock_guard<std::mutex> lck(m_bkgdMutex);
    return sptr_chromaBkgdPoints;
}

void Intact::setChromaBkgdPcl(int16_t* ptr_pcl)
{
    std::lock_guard<std::mutex> lck(m_bkgdMutex);
    sptr_chromaBkgdPcl = std::make_shared<int16_t*>(ptr_pcl);
}

std::shared_ptr<int16_t*> Intact::getChromaBkgdPcl()
{
    std::lock_guard<std::mutex> lck(m_bkgdMutex);
    return sptr_chromaBkgdPcl;
}

void Intact::setChromaBkgdImg_GL(uint8_t* ptr_img)
{
    std::lock_guard<std::mutex> lck(m_bkgdMutex);
    sptr_chromaBkgdImg_GL = std::make_shared<uint8_t*>(ptr_img);
}

std::shared_ptr<uint8_t*> Intact::getChromaBkgdImg_GL()
{
    std::lock_guard<std::mutex> lck(m_bkgdMutex);
    return sptr_chromaBkgdImg_GL;
}

void Intact::setChromaBkgdImg_CV(uint8_t* ptr_img)
{
    std::lock_guard<std::mutex> lck(m_bkgdMutex);
    sptr_chromaBkgdImg_CV = std::make_shared<uint8_t*>(ptr_img);
}

std::shared_ptr<uint8_t*> Intact::getChromaBkgdImg_CV()
{
    std::lock_guard<std::mutex> lck(m_bkgdMutex);
    return sptr_chromaBkgdImg_CV;
}

///////////////////////////////////////////////////////////////////////////////
//                          pipeline operations
///////////////////////////////////////////////////////////////////////////////

void Intact::segment(std::shared_ptr<Intact>& sptr_intact)
{
    WHILE_KINECT_READY
    START
    while (sptr_intact->isRun()) {
        std::vector<Point> points = *sptr_intact->getRefinedPoints();
        std::vector<Point> segmentPoints = region::segment(points);
        std::pair<Point, Point> boundary = region::queryBoundary(segmentPoints);
        sptr_intact->setSegmentPoints(segmentPoints);
        sptr_intact->setIntactBoundary(boundary);
        SEGMENT_READY
    }
}

void Intact::render(std::shared_ptr<Intact>& sptr_intact)
{
    WHILE_CLUSTERS_READY
    viewer::draw(sptr_intact);
}

void Intact::cluster(const float& epsilon, const int& minPoints,
    std::shared_ptr<Intact>& sptr_intact)
{
    WHILE_INTACT_READY
    START
    while (sptr_intact->isRun()) {
        std::vector<Point> points = *sptr_intact->getSegmentPoints();
        std::vector<Point> frame = *sptr_intact->getUnrefinedPoints();

        // dbscan using kd-tree: returns clustered indexes
        auto clusters = dbscan::cluster(points, epsilon, minPoints);

        // sort clusters in descending order
        std::sort(clusters.begin(), clusters.end(),
            [](const std::vector<unsigned long>& a,
                const std::vector<unsigned long>& b) {
                return a.size() > b.size();
            });

        // define chromakey color for tabletop surface
        uint8_t rgb[3] = { chromagreen[0], chromagreen[1], chromagreen[2] };
        uint8_t bgra[4] = { chromagreen[2], chromagreen[1], chromagreen[0], 0 };

        // use clustered indexes to chromakey a selected cluster
        for (const auto& index : clusters[0]) {
            int id = points[index].m_id;
            frame[id].setPixel_GL(rgb);
            frame[id].setPixel_CV(bgra);
        }

        // sizes
        const int numPoints = sptr_intact->m_numPoints;
        const uint32_t pclsize = numPoints * 3;
        const uint32_t imgsize = numPoints * 4;

        int16_t pclBuf[pclsize];
        uint8_t imgBuf_GL[pclsize];
        uint8_t imgBuf_CV[imgsize];

        // stitch data from processed point cloud
        for (int i = 0; i < numPoints; i++) {
            stitch(i, frame[i], pclBuf, imgBuf_GL, imgBuf_CV);
        }

        sptr_intact->setChromaBkgdPoints(frame);
        sptr_intact->setChromaBkgdPcl(pclBuf);
        sptr_intact->setChromaBkgdImg_GL(imgBuf_GL);
        sptr_intact->setChromaBkgdImg_CV(imgBuf_CV);
        CLUSTERS_READY
    }
}

void Intact::showObjects(std::vector<std::string>& classnames,
    torch::jit::script::Module& module, std::shared_ptr<Intact>& sptr_intact)
{
    WHILE_CLUSTERS_READY
    while (sptr_intact->isRun()) {

        // start frame rate clock
        clock_t start = clock();

        // get resources
        int w = sptr_intact->getDepthImgWidth();
        int h = sptr_intact->getDepthImgHeight();

        // uint8_t* ptr_img = *sptr_intact->getSensorImg_CV();
        // uint8_t* ptr_img = *sptr_intact->getIntactImg_CV();
        uint8_t* ptr_img = *sptr_intact->getChromaBkgdImg_CV();

        // create image frame
        cv::Mat img;
        cv::Mat frame
            = cv::Mat(h, w, CV_8UC4, (void*)ptr_img, cv::Mat::AUTO_STEP)
                  .clone();

        // format frame for tensor input
        cv::resize(frame, img, cv::Size(640, 384));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        torch::Tensor imgTensor = torch::from_blob(
            img.data, { img.rows, img.cols, 3 }, torch::kByte);
        imgTensor = imgTensor.permute({ 2, 0, 1 });
        imgTensor = imgTensor.toType(torch::kFloat);
        imgTensor = imgTensor.div(255);
        imgTensor = imgTensor.unsqueeze(0);

        torch::Tensor preds // preds: [?, 15120, 9]
            = module.forward({ imgTensor }).toTuple()->elements()[0].toTensor();
        std::vector<torch::Tensor> dets
            = yolo::non_max_suppression(preds, 0.4, 0.5);

        // show objects
        if (!dets.empty()) {
            for (size_t i = 0; i < dets[0].sizes()[0]; ++i) {
                float left
                    = dets[0][i][0].item().toFloat() * (float)frame.cols / 640;
                float top
                    = dets[0][i][1].item().toFloat() * (float)frame.rows / 384;
                float right
                    = dets[0][i][2].item().toFloat() * (float)frame.cols / 640;
                float bottom
                    = dets[0][i][3].item().toFloat() * (float)frame.rows / 384;
                float score = dets[0][i][4].item().toFloat();
                int classID = dets[0][i][5].item().toInt();

                cv::rectangle(frame,
                    cv::Rect(left, top, (right - left), (bottom - top)),
                    cv::Scalar(0, 255, 0), 2);
                cv::putText(frame,
                    classnames[classID] + ": " + cv::format("%.2f", score),
                    cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX,
                    (right - left) / 200, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::putText(frame,
            "FPS: " + std::to_string(int(1e7 / (double)(clock() - start))),
            cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
            cv::Scalar(0, 255, 0), 2);

        cv::imshow("", frame);
        if (cv::waitKey(1) == 27) {
            sptr_intact->raiseStopFlag();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}
