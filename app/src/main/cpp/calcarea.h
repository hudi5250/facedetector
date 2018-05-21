//
// Created by yjd1 on 2018/5/3.
//

#ifndef AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_CALCAREA_H
#define AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_CALCAREA_H

#include<opencv2/core.hpp>
#include<memory>
#include<iostream>
#include<vector>
using namespace std;
using namespace cv;
class mRect {
public:
    Point2f point1;
    Point2f point2;
    Point2f point3;
    Point2f point4;
    mRect(Point2f _point1, Point2f _point2, Point2f _point3, Point2f _point4) :point1(_point1), point2(_point2), point3(_point3), point4(_point4) {}
};
class calcarea
{
public:
    vector<Point2f> rect1;
    vector<Point2f> rect2;
    //y=ax*b <a,b>
    vector<pair<float, float>> line1;
    vector<pair<float, float>> line2;
    calcarea();
    ~calcarea();
    int filter;
    vector<Point2f> box;
    calcarea(Point2f _pointa1, Point2f _pointa2, Point2f _pointa3, Point2f _pointa4,
             Point2f _pointb1, Point2f _pointb2, Point2f _pointb3, Point2f _pointb4) :rect1(), rect2(), box(), line1(), line2(), filter(0) {
        rect1.push_back(_pointa1);
        rect1.push_back(_pointa2);
        rect1.push_back(_pointa3);
        rect1.push_back(_pointa4);
        rect2.push_back(_pointb1);
        rect2.push_back(_pointb2);
        rect2.push_back(_pointb3);
        rect2.push_back(_pointb4);
    }
    void getline();
    int inside(Point2f p, const vector<pair<float, float>>& line);
    pair<int, Point2f> findcross(pair<float, float> line1, pair<float, float> line2, pair<float, float> range);
    void findcandidata();
    void order();
    float getarea();
private:


};

class moffset {
public:
    moffset(int _dim1, int _dim2, int _dim3, int _dim4) :dim1(_dim1), dim2(_dim2), dim3(_dim3), dim4(_dim4) {}
    int offset(int d1, int d2, int d3, int d4) {
        if (d1 >= dim1 || d2 >= dim2 || d3 >= dim3 || d4 >= dim4)
        {
            cout << d4 << "   " << dim4 << endl;
            return 0;
        }
        return d1*dim2*dim3*dim4 + d2*dim3*dim4 + d3*dim4 + d4;
    }
private:
    int dim1;
    int dim2;
    int dim3;
    int dim4;
};
vector<Point2f> getboundingbox(const Point2f& bottom, const Point2f& left, const Point2f& right);
vector<Point2f> getrectbox(const Point2f& bottom, const Point2f& left, const Point2f& right);
float rectarea(const vector<Point2f>& rect);
vector<Point2f> getsquare(float* data, float col, float row);
pair<int, vector<float>> bounding_box_reg(const float* score, const float* box, int* data_size);
pair<vector<float>, vector<float>> net24_output_rearrange(int size, float* rate, float* data);


#endif //AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_CALCAREA_H
