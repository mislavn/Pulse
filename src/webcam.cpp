#include <iostream>
#include <numeric>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/objdetect/objdetect.hpp>

#include "config.hpp"

int main()
{
    std::string fn_haar = HAARCASCADES_PATH;
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    cv::CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Capture Device ID 0 cannot be opened.\n";
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        const cv::Mat original = frame.clone();
        cv::Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        std::vector<cv::Rect_<int>> faces;
        haar_cascade.detectMultiScale(gray, faces);

        /* show only the biggest face in the vector of detected faces */
        const auto &face_i = std::max_element(
            std::begin(faces), std::end(faces), [](const cv::Rect_<int> &a, const cv::Rect_<int> &b) { return a.area() < b.area(); });
        if (face_i == std::end(faces)) { continue; }

        const cv::Mat face = gray(*face_i);
        const cv::Mat faceMat(original, *face_i);
        const cv::Mat face_resized;
        rectangle(original, *face_i, CV_RGB(0, 255, 0), 1);
        const int pos_x = std::max(face_i->tl().x - 10, 0);
        const int pos_y = std::max(face_i->tl().y - 10, 0);

        cv::SimpleBlobDetector::Params params;
        params.filterByArea  = true;
        params.minArea       = 400;
        params.maxArea       = 100000;
        params.filterByColor = true;
        params.blobColor     = 255;

        cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
        std::vector<cv::KeyPoint> keypoints;
        detector->detect(faceMat, keypoints);

        if (keypoints.empty()) { continue; }
        const auto sum_red = std::accumulate(keypoints.begin(), keypoints.end(), 0, [&original](unsigned int prev, cv::KeyPoint kp) {
            return prev + original.at<cv::Vec3b>(kp.pt)[2];
        });

        int av_red = static_cast<int>(static_cast<float>(sum_red) / static_cast<float>(keypoints.size()));

        const std::string box_text = std::string("Face pulse ") + std::to_string(av_red);
        cv::putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2);

        cv::Mat im_with_keypoints;
        cv::drawKeypoints(face, keypoints, im_with_keypoints, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        cv::imshow("face_recognizer", original);
        if (cv::waitKey(30) >= 0) { break; }
    }
    return 0;
}
