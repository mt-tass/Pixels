#include "blocks.hpp"
#include <iostream>
#include <algorithm>

// block 1 : video capture

getVideo::getVideo(int cameraIdx) : isOpen(false){
    capture.open(cameraIdx , cv::CAP_V4L2);
    if (!capture.isOpened()) {
        std::cout << "error : camera not opened " << std::endl;
        return;
    }
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1080);
    isOpen = true;
}
getVideo::~getVideo(){
    capture.release();
    isOpen = false;
}
cv::Mat getVideo::getFrame(){
    cv::Mat frame;
    if (!isOpen){
        return cv::Mat();
    }
    
    bool success = capture.read(frame);

    if (!success || frame.empty()){
        return cv::Mat();
    }
    cv::flip(frame, frame, 1);
    return frame;
}
void getVideo::release() {
    if (capture.isOpened()) {
        capture.release();
    }
    isOpen = false;
}
bool getVideo::isOpened(){
    return isOpen;
}

// block 2 : light detector

lightDetector::lightDetector(int min_thresh, int min_area, int max_area){
    minVal = min_thresh;
    minArea = min_area;
    maxArea = max_area;
    lowerBound = cv::Scalar(0, 0, minVal);
    upperBound = cv::Scalar(180, 255, 255);
}

cv::Point lightDetector::getBrightestPoint(const cv::Mat &frame){
    if (frame.empty()){
        return cv::Point(-1, -1);
    }
    
    cv::Mat hsvFrame;
    cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
    
    cv::Mat mask;
    cv::inRange(hsvFrame, lowerBound, upperBound, mask);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<cv::Point> largestContour;
    double largestArea = 0;
    
    for (const auto &contour : contours){
        double area = cv::contourArea(contour);
        if (area >= minArea && area <= maxArea){
            if(area > largestArea){
                largestArea = area;
                largestContour = contour;
            }
        }
    }
    
    if (!largestContour.empty()){
        cv::Moments m = cv::moments(largestContour);
        if (m.m00 != 0) {
            int cx = static_cast<int>(m.m10 / m.m00);
            int cy = static_cast<int>(m.m01 / m.m00);
            return cv::Point(cx, cy);
        }
    }
    
    return cv::Point(-1, -1);
}

// block 3 : image overlaying

cv::Mat imageOverlay::putImage(const cv::Mat &background, const cv::Mat &foreground,cv::Point position, double scale, bool removeBlack){
    cv::Mat result = background.clone();
    cv::Mat fg = foreground.clone();
    
    int fgH = fg.rows;
    int fgW = fg.cols;
    
    if (scale != 1.0) {
        fgW = static_cast<int>(fgW * scale);
        fgH = static_cast<int>(fgH * scale);
        cv::resize(fg, fg, cv::Size(fgW, fgH), 0, 0, cv::INTER_AREA);
    }
    
    int cx = position.x;
    int cy = position.y;
    
    int x1 = cx - (fgW / 2);
    int y1 = cy - (fgH / 2);
    int x2 = x1 + fgW;
    int y2 = y1 + fgH;
    
    int bgH = background.rows;
    int bgW = background.cols;
    
    int bx1 = std::max(0, x1);
    int by1 = std::max(0, y1);
    int bx2 = std::min(bgW, x2);
    int by2 = std::min(bgH, y2);
    
    int fx1 = std::max(0, -x1);
    int fy1 = std::max(0, -y1);
    int fx2 = fgW - std::max(0, x2 - bgW);
    int fy2 = fgH - std::max(0, y2 - bgH);
    
    if (bx1 >= bx2 || by1 >= by2){
        return result;
    }
    
    cv::Mat bgROI = result(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1));
    cv::Mat fgROI = fg(cv::Rect(fx1, fy1, fx2 - fx1, fy2 - fy1));
    
    if (fg.channels() == 4) {
        std::vector<cv::Mat> channels;
        cv::split(fgROI, channels);
        
        std::vector<cv::Mat> rgbChannels = {channels[0], channels[1], channels[2]};
        cv::Mat rgb;
        cv::merge(rgbChannels, rgb);
        
        cv::Mat alpha;
        channels[3].convertTo(alpha, CV_32F, 1.0 / 255.0);
        
        cv::Mat rgbF, bgROIF;
        rgb.convertTo(rgbF, CV_32FC3);
        bgROI.convertTo(bgROIF, CV_32FC3);
        
        std::vector<cv::Mat> alphaChannels = {alpha, alpha, alpha};
        cv::Mat alpha3;
        cv::merge(alphaChannels, alpha3);
        
        cv::Mat blend;
        cv::multiply(alpha3, rgbF, rgbF);
        cv::multiply(cv::Scalar(1.0, 1.0, 1.0) - alpha3, bgROIF, bgROIF);
        cv::add(rgbF, bgROIF, blend);
        
        blend.convertTo(bgROI, CV_8UC3);
    }
    else if (removeBlack && fg.channels() == 3){
        cv::Mat gray;
        cv::cvtColor(fgROI, gray, cv::COLOR_BGR2GRAY);
        cv::Mat mask = gray > 20;
        
        fgROI.copyTo(bgROI, mask);
    }
    else{
        fgROI.copyTo(bgROI);
    }
    return result;
}

// block 4 : calibaration rectangle

calibrationRectangle::calibrationRectangle(int x, int y, int width, int height){
        rect = cv::Rect(x,y,width,height);
        calibrated = false;
}
void  calibrationRectangle::draw(cv::Mat &frame){
    cv::Scalar color;
    if(calibrated){
         color = cv::Scalar(0,255,0);
    }
    else{
        color=cv::Scalar(0,165,255);
    }
    cv::rectangle(frame, rect, color, 3);
    std::string text = calibrated ? "CALIBRATED" : "Place cloth here - Press SPACE";
    cv::Size text_size = cv::getTextSize(text,cv::FONT_HERSHEY_SIMPLEX,0.6,2,0);
    int x_pos = frame.cols/2-text_size.width/2;
    cv::Point textPos(x_pos, rect.y - 10);
    cv::putText(frame, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
}

bool calibrationRectangle::calibrate(const cv::Mat& frame) {
    if (frame.empty()){
        return false;
    }
    
    cv::Mat roi = frame(rect);
    cv::Mat hsvROI;
    cv::cvtColor(roi, hsvROI, cv::COLOR_BGR2HSV);
    
    cv::Scalar mean = cv::mean(hsvROI);
    cv::Scalar stddev;
    cv::meanStdDev(hsvROI, mean, stddev);
    
    int hueMargin = 15;
    int satMargin = 50;
    int valMargin = 50;
    
    lowerBound = cv::Scalar(
        std::max(0.0, mean[0] - hueMargin),
        std::max(0.0, mean[1] - satMargin),
        std::max(0.0, mean[2] - valMargin)
    );
    
    upperBound = cv::Scalar(
        std::min(180.0, mean[0] + hueMargin),
        std::min(255.0, mean[1] + satMargin),
        std::min(255.0, mean[2] + valMargin)
    );
    
    calibrated = true;
    return true;
}

cv::Mat calibrationRectangle::getMask(const cv::Mat &frame){
    if (!calibrated || frame.empty()){
        return cv::Mat();
    }
    
    cv::Mat hsvFrame;
    cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
    
    cv::Mat mask;
    cv::inRange(hsvFrame, lowerBound, upperBound, mask);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);
    
    return mask;
}

// block 5 : menu UI

menuSystem::menuSystem(){
    selectedIndex = 0;
    normalColor = cv::Scalar(180, 180, 180);
    selectedColor = cv::Scalar(0, 215, 255);
    bgColor = cv::Scalar(10,10,10);
}

void menuSystem::addItem(const std::string &item) {
    items.push_back(item);
}

void menuSystem::moveUp() {
    if (selectedIndex > 0) {
        selectedIndex--;
    }
    else {
        selectedIndex = static_cast<int>(items.size())-1;
    }
}

void menuSystem::moveDown() {
    if (selectedIndex < static_cast<int>(items.size()) - 1) {
        selectedIndex++;
    }
    else{
        selectedIndex = 0;
    }
}

void menuSystem::draw(cv::Mat &frame, int startY) {
    int itemHeight = 60;
    int padding = 15;
    cv::Size text_size;
    
    for (int i = 0; i < static_cast<int>(items.size()); i++) {
        int y = startY + (i * itemHeight);
        bool isSelected = (i==selectedIndex);
        text_size = cv::getTextSize(items[i],cv::FONT_HERSHEY_COMPLEX,0.8,isSelected?2:1,0);
        int startX = (frame.cols/4)-text_size.width/2;
        if (isSelected) {
            cv::rectangle(frame, cv::Point(startX - 10, y - padding),cv::Point(startX + text_size.width+10, y + 35),cv::Scalar(40, 40, 40), -1);
            cv::rectangle(frame, cv::Point(startX - 10, y - padding),cv::Point(startX + text_size.width+10, y + 35),selectedColor, 2);
        }
        cv::Scalar textColor ;
        if(isSelected){
            textColor = selectedColor;
        }
        else{
            textColor = normalColor;
        }
        cv::putText(frame, items[i], cv::Point(startX, y + 20),cv::FONT_HERSHEY_COMPLEX, 0.8, textColor,isSelected ? 2 : 1);
    }
}

// block 6 : main UI

const cv::Scalar MagicalUI::GOLD = cv::Scalar(0, 215, 255);
const cv::Scalar MagicalUI::DARK_BG = cv::Scalar(15, 15, 15);
const cv::Scalar MagicalUI::SELECTED = cv::Scalar(0, 215, 255);
const cv::Scalar MagicalUI::CREAM = cv::Scalar(200, 220, 230);

void MagicalUI::drawDarkBackground(cv::Mat& frame) {
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay,cv::Point(frame.cols/6,0),cv::Point(frame.cols/3,frame.rows),cv::Scalar(10,10,10),-1,cv::LINE_AA);
    cv::addWeighted(frame, 0.3, overlay, 0.7, 0, frame);
}

void MagicalUI::drawTitle(cv::Mat& frame, const std::string& title , const std::string &subtitle , const std::string &footer){
    cv::Size text_size = cv::getTextSize(title,cv::FONT_HERSHEY_COMPLEX,1.5,3,0);
    cv::Size text2_size = cv::getTextSize(subtitle,cv::FONT_HERSHEY_COMPLEX,0.7,1,0);
    cv::Size text3_size = cv::getTextSize(footer,cv::FONT_HERSHEY_SIMPLEX,0.5,1,0);
    cv::Point pos(frame.cols / 4 - text_size.width/2, 80);
    cv::Point pos2(frame.cols / 4 - text2_size.width/2, 120);
    cv::Point pos3((frame.cols/4 - text3_size.width/2),frame.rows-30);
    
    cv::putText(frame, title, cv::Point(pos.x + 2, pos.y + 2),cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 0), 5);
    cv::putText(frame, title, pos,cv::FONT_HERSHEY_COMPLEX, 1.5, GOLD, 3);
    cv::putText(frame, subtitle, pos2 ,cv::FONT_HERSHEY_COMPLEX, 0.7,CREAM, 1);
    cv::putText(frame,footer,pos3,cv::FONT_HERSHEY_SIMPLEX,0.5,CREAM,1);
}

void MagicalUI::drawStatus(cv::Mat& frame, const std::string& status, cv::Scalar color) {
    cv::putText(frame, status, cv::Point(20, 40),cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
}
void MagicalUI::drawScore(cv::Mat& frame, int score, const std::string& label) {
    std::string text = label +":" +std::to_string(score);
    cv::putText(frame, text, cv::Point(20, frame.rows - 20),cv::FONT_HERSHEY_SIMPLEX, 0.8, GOLD, 2);
}

// block 7 : Game Implementation : Snitch

GoldenSnitch::GoldenSnitch(){
    position = cv::Point(0,0);
    velocity = cv::Point(0,0);
    radius = 15;
    active = true;
    srand(time(0));
}
GoldenSnitch::GoldenSnitch(int frameWidth, int y){
    radius = 15;
    active = true;
    position.x = rand() % frameWidth;
    position.y = y;
    velocity.x = 0;
    velocity.y = 2 + rand() % 3;
}
void GoldenSnitch::update(int frameHeight){
    if (!active){
        return;
    }
    position.y += velocity.y;
    if (position.y > frameHeight + radius){
        active = false;
    }
}
void GoldenSnitch::draw(cv::Mat& frame){
    if (!active){
        return;
    }
    
    cv::circle(frame, position, radius, cv::Scalar(0, 215, 255), -1);
    cv::circle(frame, position, radius, cv::Scalar(0, 180, 220), 2);
    
    cv::Point leftWing1(position.x - radius - 5, position.y - 3);
    cv::Point leftWing2(position.x - radius - 12, position.y);
    cv::Point rightWing1(position.x + radius + 5, position.y - 3);
    cv::Point rightWing2(position.x + radius + 12, position.y);
    
    cv::line(frame, leftWing1, leftWing2, cv::Scalar(0, 215, 255), 2);
    cv::line(frame, rightWing1, rightWing2, cv::Scalar(0, 215, 255), 2);
}

bool GoldenSnitch::checkCollision(cv::Point ringPosition, int ringRadius){
    if (!active){
        return false;
    }
    int dx = position.x - ringPosition.x;
    int dy = position.y - ringPosition.y;
    int distance = sqrt(dx * dx + dy * dy);
    
    return distance < (radius + ringRadius);
}

void GoldenSnitch::reset(int frameWidth) {
    position.x = rand() % frameWidth;
    position.y = 0;
    velocity.y = 2 + rand() % 3;
    active = true;
}

// block 8 : Game Implementatio : Quidditch

QuidditchGame::QuidditchGame(int width, int height){
    score = 0;
    missed = 0;
    ringRadius = 50;
    frameWidth = width;
    frameHeight = height;
    spawnTimer = 0;
    spawnInterval = 60;
    ringX = ringRadius;
    srand(time(0));
}
void QuidditchGame::updateRingPosition(cv::Point lightPosition) {
    if (lightPosition.x != -1) {
        ringX = lightPosition.x;
    }
}
void QuidditchGame::update(){
    spawnTimer++;
    if (spawnTimer >= spawnInterval){
        snitches.push_back(GoldenSnitch(frameWidth, -15));
        spawnTimer = 0;
        if (spawnInterval > 30){
            spawnInterval -= 1;
        }
    }
    
    for (auto &snitch : snitches){
        snitch.update(frameHeight);
        
        if (snitch.active && snitch.checkCollision(cv::Point(ringX,frameHeight - 60),ringRadius)){
            score += 10;
            snitch.active = false;
        }
        if (!snitch.active && snitch.position.y > frameHeight) {
            missed++;
        }
    }
    for (auto it = snitches.begin() ; it != snitches.end() ; ){
        if(!(it->active) && it->position.y > frameHeight+50){
            it = snitches.erase(it);
        }
        else{
            it++;
        }

    }
}

void QuidditchGame::draw(cv::Mat &frame){
    for (auto &snitch : snitches){
        snitch.draw(frame);
    }
    cv::circle(frame, cv::Point(ringX, frameHeight - 60), ringRadius, cv::Scalar(0, 215, 255), 3);
    cv::circle(frame, cv::Point(ringX, frameHeight - 60), ringRadius - 5, cv::Scalar(0, 180, 220), 2);
    cv::circle(frame, cv::Point(ringX, frameHeight - 60), 5, cv::Scalar(0, 255, 255), -1);
}
void QuidditchGame::reset(){
    score = 0;
    missed = 0;
    snitches.clear();
    spawnTimer = 0;
    spawnInterval = 60;
}