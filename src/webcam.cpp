#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/objdetect/objdetect.hpp>

#include "config.hpp"

int main() {
    std::string fn_haar = HAARCASCADES_PATH;
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    cv::CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);

    cv::VideoCapture cap(0);

    if(!cap.isOpened()) {
        std::cerr << "Capture Device ID 0 cannot be opened." << '\n';
        return -1;
    }

    cv::Mat frame;
    while(true) {
        cap >> frame;
        cv::Mat original = frame.clone();
        cv::Mat gray;
        cvtColor(original, gray, CV_BGR2GRAY);
        std::vector<cv::Rect_<int>> faces;
        haar_cascade.detectMultiScale(gray, faces);

        for(unsigned int i = 0; i < faces.size(); i++) {
            cv::Rect face_i = faces[i];
            cv::Mat face = gray(face_i);
            cv::Mat face_resized;
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            std::string box_text = cv::format("Face");
            cv::putText(original, box_text, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

            cv::SimpleBlobDetector::Params params;

            cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
            std::vector<cv::KeyPoint> keypoints;
            detector->detect(face, keypoints);

            cv::Mat im_with_keypoints;
            cv::drawKeypoints(face, keypoints, im_with_keypoints, cv::Scalar(0,0,255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
            cv::imshow("keypoints", im_with_keypoints);
        }

        //cv::imshow("face_recognizer", original);
        if(cv::waitKey(30) >= 0) {
            break;
        }
    }
    return 0;
}
