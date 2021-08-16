#include <cmath>
#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <torch/script.h>

#include "i3dutils.h"
#include "point.h"
#include "yolov5.h"

// colors handy for coloring clusters:
// see@ https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=9
//
__attribute__((unused)) uint8_t red[3] = { 215, 48, 39 };
__attribute__((unused)) uint8_t orange[3] = { 244, 109, 67 };
__attribute__((unused)) uint8_t gold[3] = { 253, 173, 97 };
__attribute__((unused)) uint8_t brown[3] = { 254, 224, 144 };
__attribute__((unused)) uint8_t yellow[3] = { 255, 255, 191 };
__attribute__((unused)) uint8_t skyblue[3] = { 224, 243, 248 };
__attribute__((unused)) uint8_t oceanblue[3] = { 171, 217, 233 };
__attribute__((unused)) uint8_t blue[3] = { 116, 173, 209 };
__attribute__((unused)) uint8_t deepblue[3] = { 69, 117, 180 };

// colors handy for coloring clusters:
// https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=9 */
//
__attribute__((unused)) uint8_t deepbrown[3] = { 140, 81, 10 };
__attribute__((unused)) uint8_t darkbrown[3] = { 191, 129, 45 };
__attribute__((unused)) uint8_t goldenbrown[3] = { 223, 194, 125 };
__attribute__((unused)) uint8_t khaki[3] = { 223, 232, 195 };
__attribute__((unused)) uint8_t lightgrey[3] = { 245, 245, 245 };
__attribute__((unused)) uint8_t lightgreen[3] = { 199, 234, 229 };
__attribute__((unused)) uint8_t green[3] = { 128, 205, 193 };
__attribute__((unused)) uint8_t chromagreen[3] = { 120, 198, 121 };
__attribute__((unused)) uint8_t deepgreen[3] = { 53, 151, 143 };
__attribute__((unused)) uint8_t othergreen[3] = { 1, 102, 94 };
__attribute__((unused)) uint8_t black[3] = { 0, 0, 0 };

void i3dutils::configTorch(
    std::vector<std::string>& classNames, torch::jit::script::Module& module)
{
    const std::string scriptName = io::pwd() + "/resources/torchscript.pt";
    const std::string cocoNames = io::pwd() + "/resources/coco.names";
    module = torch::jit::load(scriptName);
    std::ifstream f(cocoNames);
    std::string name;
    while (std::getline(f, name)) {
        classNames.push_back(name);
    }
}

void i3dutils::adapt(const int& index, Point& point,
    const std::vector<int16_t>& pCloud, const std::vector<uint8_t>& image)
{
    int16_t x = pCloud[3 * index + 0];
    int16_t y = pCloud[3 * index + 1];
    int16_t z = pCloud[3 * index + 2];

    uint8_t b = image[4 * index + 0];
    uint8_t g = image[4 * index + 1];
    uint8_t r = image[4 * index + 2];
    uint8_t a = image[4 * index + 3];

    int16_t xyz[3] = { x, y, z };
    uint8_t rgba[4] = { r, g, b, a };
    uint8_t bgra[4] = { b, g, r, a };

    point.m_id = index;
    point.setPoint(xyz);
    point.setRGBA(rgba);
    point.setBGRA(bgra);
}

bool i3dutils::null(
    const int& index, std::vector<int16_t>& pCloud, std::vector<uint8_t>& image)
{
    if (pCloud[3 * index + 2] == 0 && image[4 * index + 3] == 0) {
        return true;
    }
    return false;
}

bool i3dutils::invalid(const int& index, const k4a_float2_t* ptr_xyTable,
    const uint16_t* depthData)
{
    if (depthData[index] == 0 && std::isnan(ptr_xyTable[index].xy.x)
        && std::isnan(ptr_xyTable[index].xy.y)) {
        return true;
    }

    auto x = (int16_t)(ptr_xyTable[index].xy.x * (float)depthData[index]);
    auto y = (int16_t)(ptr_xyTable[index].xy.y * (float)depthData[index]);
    auto z = (int16_t)depthData[index];
    if (x == 0 && y == 0 && z == 0) {
        return true;
    }

    return false;
}

bool i3dutils::invalid(
    const int& index, const int16_t* xyzData, const uint8_t* rgbaData)
{
    if (xyzData[3 * index + 2] == 0 || std::isnan(xyzData[3 * index + 2])) {
        return true;
    }
    if (std::isnan(rgbaData[index])) {
        return true;
    }
    return false;
}

void i3dutils::addXYZ(
    const int& index, std::vector<int16_t>& pCloud, const int16_t* xyzData)
{
    pCloud[3 * index + 0] = xyzData[3 * index + 0];
    pCloud[3 * index + 1] = xyzData[3 * index + 1];
    pCloud[3 * index + 2] = xyzData[3 * index + 2];
}

void i3dutils::addXYZ(const int& index, std::vector<int16_t>& pCloud)
{
    pCloud[3 * index + 0] = 0;
    pCloud[3 * index + 1] = 0;
    pCloud[3 * index + 2] = 0;
}

void i3dutils::addBGRA(const int& index, std::vector<uint8_t>& bgraImage,
    const uint8_t* ptr_imgData)
{
    bgraImage[4 * index + 0] = ptr_imgData[4 * index + 0]; // blue
    bgraImage[4 * index + 1] = ptr_imgData[4 * index + 1]; // green
    bgraImage[4 * index + 2] = ptr_imgData[4 * index + 2]; // red
    bgraImage[4 * index + 3] = ptr_imgData[4 * index + 3]; // alpha
}

void i3dutils::addRGBA(const int& index, std::vector<uint8_t>& rgbaImage,
    const uint8_t* ptr_imgData)
{
    rgbaImage[4 * index + 2] = ptr_imgData[4 * index + 0]; // blue
    rgbaImage[4 * index + 1] = ptr_imgData[4 * index + 1]; // green
    rgbaImage[4 * index + 0] = ptr_imgData[4 * index + 2]; // red
    rgbaImage[4 * index + 3] = ptr_imgData[4 * index + 3]; // alpha
}

void i3dutils::addRGBA(const int& index, std::vector<uint8_t>& rgbaImage)
{
    rgbaImage[4 * index + 0] = 0; // red
    rgbaImage[4 * index + 1] = 0; // green
    rgbaImage[4 * index + 2] = 0; // blue
    rgbaImage[4 * index + 3] = 0; // alpha
}

void i3dutils::addBGRA(const int& index, std::vector<uint8_t>& bgraImage)
{
    bgraImage[4 * index + 0] = 0; // red
    bgraImage[4 * index + 1] = 0; // green
    bgraImage[4 * index + 2] = 0; // blue
    bgraImage[4 * index + 3] = 0; // alpha
}

bool i3dutils::inSegment(const int& index, const std::vector<int16_t>& pCloud,
    const Point& minPoint, const Point& maxPoint)
{
    if (pCloud[3 * index + 2] == 0) {
        return false;
    }

    if ((int16_t)pCloud[3 * index + 0] > maxPoint.m_xyz[0]
        || (int16_t)pCloud[3 * index + 0] < minPoint.m_xyz[0]
        || (int16_t)pCloud[3 * index + 1] > maxPoint.m_xyz[1]
        || (int16_t)pCloud[3 * index + 1] < minPoint.m_xyz[1]
        || (int16_t)pCloud[3 * index + 2] > maxPoint.m_xyz[2]
        || (int16_t)pCloud[3 * index + 2] < minPoint.m_xyz[2]) {
        return false;
    }
    return true;
}

void i3dutils::stitch(const int& index, Point& point, int16_t* xyzData,
    uint8_t* rgbaData, uint8_t* bgraData)
{
    xyzData[3 * index + 0] = point.m_xyz[0]; // x
    xyzData[3 * index + 1] = point.m_xyz[1]; // y
    xyzData[3 * index + 2] = point.m_xyz[2]; // z

    bgraData[4 * index + 0] = point.m_bgra[0]; // blue
    bgraData[4 * index + 1] = point.m_bgra[1]; // green
    bgraData[4 * index + 2] = point.m_bgra[2]; // red
    bgraData[4 * index + 3] = point.m_bgra[3]; // alpha

    rgbaData[4 * index + 0] = point.m_rgba[0]; // red
    rgbaData[4 * index + 1] = point.m_rgba[1]; // green
    rgbaData[4 * index + 2] = point.m_rgba[2]; // blue
    rgbaData[4 * index + 3] = point.m_rgba[3]; // alpha
}

void i3dutils::stitch(const int& index, Point& point,
    std::vector<int16_t>& pCloud, std::vector<uint8_t> rgba)
{
    pCloud[3 * index + 0] = point.m_xyz[0]; // x
    pCloud[3 * index + 1] = point.m_xyz[1]; // y
    pCloud[3 * index + 2] = point.m_xyz[2]; // z

    rgba[4 * index + 0] = point.m_rgba[0]; // red
    rgba[4 * index + 1] = point.m_rgba[1]; // green
    rgba[4 * index + 2] = point.m_rgba[2]; // blue
    rgba[4 * index + 3] = point.m_rgba[3]; // alpha
}

void i3dutils::stitch(const int& index, Point& point, uint8_t* bgra)
{
    bgra[4 * index + 0] = point.m_bgra[0]; // blue
    bgra[4 * index + 1] = point.m_bgra[1]; // green
    bgra[4 * index + 2] = point.m_bgra[2]; // red
    bgra[4 * index + 3] = point.m_bgra[3]; // alpha
}

std::pair<Point, Point> i3dutils::queryBoundary(std::vector<Point>& points)
{
    std::vector<int16_t> X(points.size());
    std::vector<int16_t> Y(points.size());
    std::vector<int16_t> Z(points.size());
    for (int i = 0; i < points.size(); i++) {
        X[i] = points[i].m_xyz[0];
        Y[i] = points[i].m_xyz[1];
        Z[i] = points[i].m_xyz[2];
    }

    int16_t xMax = (int16_t)*std::max_element(X.begin(), X.end());
    int16_t xMin = (int16_t)*std::min_element(X.begin(), X.end());
    int16_t yMax = (int16_t)*std::max_element(Y.begin(), Y.end());
    int16_t yMin = (int16_t)*std::min_element(Y.begin(), Y.end());
    int16_t zMax = (int16_t)*std::max_element(Z.begin(), Z.end());
    int16_t zMin = (int16_t)*std::min_element(Z.begin(), Z.end());

    Point min(xMin, yMin, zMin);
    Point max(xMax, yMax, zMax);
    return { min, max };
}

void i3dutils::add(std::vector<uint8_t*>& colors)
{
    colors.emplace_back(red);
    colors.emplace_back(orange);
    colors.emplace_back(gold);
    colors.emplace_back(brown);
    colors.emplace_back(yellow);
    colors.emplace_back(skyblue);
    colors.emplace_back(oceanblue);
    colors.emplace_back(blue);
    colors.emplace_back(deepblue);
    colors.emplace_back(deepbrown);
    colors.emplace_back(goldenbrown);
    colors.emplace_back(khaki);
    colors.emplace_back(lightgreen);
    colors.emplace_back(lightgrey);
    colors.emplace_back(green);
    colors.emplace_back(chromagreen);
    colors.emplace_back(deepgreen);
    colors.emplace_back(othergreen);
    colors.emplace_back(black);
}

int i3dutils::randNum(const int& max)
{
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0, max);
    return (int)dist(mt);
}

void i3dutils::show(const int& h, const int& w, uint8_t* bgraData,
    std::shared_ptr<i3d>& sptr_i3d)
{
    cv::Mat img
        = cv::Mat(h, w, CV_8UC4, (void*)bgraData, cv::Mat::AUTO_STEP).clone();
    cv::imshow("i3d", img);
    if (cv::waitKey(1) == 27) {
        sptr_i3d->stop();
    }
}

void i3dutils::findObjects(const int& h, const int& w, uint8_t* bgraData,
    std::vector<std::string>& classnames, torch::jit::script::Module& module,
    std::shared_ptr<i3d>& sptr_i3d)
{
    clock_t start = clock();

    cv::Mat frame, frameResized;
    frame = cv::Mat(h, w, CV_8UC4, (void*)bgraData, cv::Mat::AUTO_STEP).clone();

    // format frame for tensor input
    cv::resize(frame, frameResized, cv::Size(640, 384));
    cv::cvtColor(frameResized, frameResized, cv::COLOR_BGR2RGB);
    torch::Tensor imgTensor = torch::from_blob(frameResized.data,
        { frameResized.rows, frameResized.cols, 3 }, torch::kByte);
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
        for (int64_t i = 0; i < dets[0].sizes()[0]; ++i) {
            auto left = (int)(dets[0][i][0].item().toFloat() * (float)frame.cols
                / 640);
            auto top = (int)(dets[0][i][1].item().toFloat() * (float)frame.rows
                / 384);
            auto right = (int)(dets[0][i][2].item().toFloat()
                * (float)frame.cols / 640);
            auto bottom = (int)(dets[0][i][3].item().toFloat()
                * (float)frame.rows / 384);
            float score = dets[0][i][4].item().toFloat();
            int classID = dets[0][i][5].item().toInt();

            cv::rectangle(frame,
                cv::Rect(left, top, (right - left), (bottom - top)),
                cv::Scalar(0, 255, 0), 1);
            // todo:: we can introduce a mapping function here that
            //   evaluates classnames
            //  1. find closest free space in vacant surface
            //  2. allocate coordinates for the projection
            //  3. project .... cause bob is your uncle
            //
            cv::putText(frame,
                classnames[classID] + ": " + cv::format("%.2f", score),
                cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX,
                (double)(right - left) / 200, cv::Scalar(0, 255, 0), 1);
        }
    }

    cv::putText(frame,
        "FPS: " + std::to_string(int(1e7 / (double)(clock() - start))),
        cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0),
        2);

    cv::imshow("i3d", frame);
    if (cv::waitKey(1) == 27) {
        sptr_i3d->stop();
    }
}
