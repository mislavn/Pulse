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
    const std::string fn_haar = HAARCASCADES_PATH;
    cv::CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    float prev_av_red = -1;

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Capture Device ID 0 cannot be opened.\n";
        return -1;
    }

    cv::Mat frame, face_HSV, face_threshold;
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
        cv::rectangle(original, *face_i, CV_RGB(0, 255, 0), 1);
        const int pos_x = std::max(face_i->tl().x - 10, 0);
        const int pos_y = std::max(face_i->tl().y - 10, 0);

        // Convert from BGR to HSV colorspace
        cv::cvtColor(faceMat, face_HSV, cv::COLOR_BGR2HSV, 3);
        // Detect the object based on HSV Range Values
        cv::inRange(face_HSV, cv::Scalar(0, 100, 30), cv::Scalar(5, 255, 255), face_threshold);
        cv::Mat face_masked;
        cv::copyTo(faceMat, face_masked, face_threshold);

        int size_red = std::accumulate(face_threshold.begin<cv::Vec3b>(), face_threshold.end<cv::Vec3b>(), 0, [](unsigned int prev, cv::Vec3b kp) {
            return (kp[0] > 0) ? prev + 1 : prev;
        });
        int sum_red  = std::accumulate(
            face_masked.begin<cv::Vec3b>(), face_masked.end<cv::Vec3b>(), 0, [](unsigned int prev, cv::Vec3b kp) { return prev + kp[2]; });

        float av_red = static_cast<float>(sum_red) / static_cast<float>(size_red);

        if (prev_av_red == -1) {
            prev_av_red = av_red;
            continue;
        }
        const std::string box_text = std::string("Red pixel change ") + std::to_string(av_red - prev_av_red) + std::string(" on ") + std::to_string(size_red) + std::string(" pixels.");
        cv::putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255, 0), 2);
        prev_av_red = av_red;

        cv::imshow("test2", original);
        if (cv::waitKey(30) >= 0) { break; }
    }
    return 0;
}
