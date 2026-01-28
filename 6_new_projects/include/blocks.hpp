#ifndef BLOCKS_HEADER
#define BLOCKS_HEADER

#include <opencv4/opencv2/opencv.hpp>
#include <vector>

//Live Video Block
class getVideo{
    private:
        cv::VideoCapture capture;
        bool isOpen;
    public:
        getVideo(int cam_idx=0);
        ~getVideo();
        cv::Mat getFrame();
        void release();
        void isOpened();

};
//Light Detection Block
class lightDetector{
    private:
        int minVal , minArea , maxArea;
        cv::Scalar lowerBound , upperBound;
    public:
        lightDetector(int min_thresh=245 , int min_area =8500,int max_area= 10000);
        cv::Point get_bp(const cv::Mat &frame);
};
//Image Overlay Block
class imageOverlay{
    public:
        imageOverlay() = default;
        cv::Mat putImage(const cv::Mat &bg ,const cv::Mat &fg, cv::Point position, double scale=1.0 );
};
#endif