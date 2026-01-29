#include <iostream>
#include <opencv2/opencv.hpp>
#include "blocks.hpp"

using namespace std;

int main(){
    const int cam_idx = 0;
    const int hsv_thresh = 250;
    const int minArea = 9000;
    const int maxArea =15000;
    const double scale = 0.65;

    getVideo cam(cam_idx);
    if (!cam.isOpened()){
        cout << "error in opening camera , check id !" << endl;
        return -1;
    }
    lightDetector detector(hsv_thresh,minArea,maxArea);
    imageOverlay img;
    cv::Mat patronous = cv::imread("data/unicorn.png");
    if (patronous.empty()){
        cout << "error in opening patronus imagae !" << endl;
        return -1;
    }
    while(true){
        cv::Mat frame = cam.getFrame();
        if(frame.empty()){
            cout << "video not getting captured !" << endl;
            return -1;
        }
        cv::Point pos = detector.get_bp(frame);
        if(pos.x != -1 && pos.y != -1){
            frame = img.putImage(frame , patronous , pos , scale);

        }
        cv::imshow("expecto patronous" , frame);
        if(cv::waitKey(1) == 'q'){
            break;
        }
    }
    cam.release();
    cv::destroyAllWindows();
    return 0;
}