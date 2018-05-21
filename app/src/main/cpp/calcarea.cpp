//
// Created by yjd1 on 2018/5/3.
//

#include "calcarea.h"



calcarea::calcarea()
{
}


calcarea::~calcarea()
{
}
const float MINF = 999999999999;
bool locationsort(const Point2f & a, const Point2f & b)
{
    if (a.y < b.y) {
        return true;
    }
    else if (a.y == b.y) {
        if (a.x < b.y)
            return true;
    }
    return false;
}

bool anglesort(const pair<float, Point2f>& a, const pair<float, Point2f>& b)
{
    if (a.first > b.first)
        return true;
    return false;
}


void calcarea::getline()
{
    float tmpk;
    vector<Point2f> tmprect1(rect1);
    tmprect1.push_back(rect1[0]);
    for (int i = 0; i < 4; i++) {
        if (tmprect1[i].x - tmprect1[i + 1].x == 0) {
            line1.push_back(make_pair(MINF, tmprect1[i].x));
        }
        else {
            tmpk = (tmprect1[i].y - tmprect1[i + 1].y) / (tmprect1[i].x - tmprect1[i + 1].x);
            line1.push_back(make_pair(tmpk, tmprect1[i].y - tmpk*tmprect1[i].x));
        }
    }
    vector<Point2f> tmprect2(rect2);
    tmprect2.push_back(rect2[0]);
    for (int i = 0; i < 4; i++) {
        if (tmprect2[i].x - tmprect2[i + 1].x == 0) {
            line2.push_back(make_pair(MINF, tmprect2[i].x));
        }
        else {
            tmpk = (tmprect2[i].y - tmprect2[i + 1].y) / (tmprect2[i].x - tmprect2[i + 1].x);
            line2.push_back(make_pair(tmpk, tmprect2[i].y - tmpk*tmprect2[i].x));
        }
    }
}

int calcarea::inside(Point2f p, const vector<pair<float, float>>& line)
{
    float tmp1, tmp2, tmp3, tmp4;
    if (line[0].first != MINF)
        tmp1 = line[0].first*p.x + line[0].second - p.y;
    else
        tmp1 = line[0].second - p.x;
    if (line[1].first != MINF)
        tmp2 = line[1].first*p.x + line[1].second - p.y;
    else
        tmp2 = line[1].second - p.x;
    if (line[2].first != MINF)
        tmp3 = line[2].first*p.x + line[2].second - p.y;
    else
        tmp3 = line[2].second - p.x;
    if (line[3].first != MINF)
        tmp4 = line[3].first*p.x + line[3].second - p.y;
    else
        tmp4 = line[3].second - p.x;
    if (tmp1*tmp3 < 0 && tmp2*tmp4 < 0)
        return 1;
    else
        return 0;
}

pair<int, Point2f> calcarea::findcross(pair<float, float> line1, pair<float, float> line2, pair<float, float> range)
{
    int aaa = 0;
    aaa++;
    if (line1.first == MINF&&line2.first == MINF)
        return make_pair(0, Point2f());
    else if (line1.first == MINF) {
        if (line1.second >= range.first&&line1.second <= range.second) {
            return make_pair(1, Point2f(line1.second, line1.second*line2.first + line2.second));
        }
    }
    else if (line2.first == MINF) {
        if (line2.second >= range.first&&line2.second <= range.second) {
            return make_pair(1, Point2f(line2.second, line2.second*line1.first + line1.second));
        }
    }
    else
    {
        float result = (line1.second - line2.second) / (line2.first - line1.first);
        if (result >= range.first&&result <= range.second) {
            return make_pair(1, Point2f(result, result*line1.first + line1.second));
        }
    }
    return make_pair(0, Point2f());
}

void calcarea::findcandidata()
{
    getline();
    for (int i = 0; i < 4; i++) {
        if (inside(rect1[i], line2) == 1)
            box.push_back(rect1[i]);
        if (inside(rect2[i], line1) == 1)
            box.push_back(rect2[i]);
    }
    //get the range
    vector<Point2f> tmprect1(rect1);
    tmprect1.push_back(rect1[0]);
    vector<Point2f> tmprect2(rect2);
    tmprect2.push_back(rect2[0]);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            vector<float> tmp;
            tmp.push_back(tmprect1[i].x);
            tmp.push_back(tmprect1[i + 1].x);
            tmp.push_back(tmprect2[j].x);
            tmp.push_back(tmprect2[j + 1].x);
            if (tmp[2] > tmp[0] && tmp[2] > tmp[1] && tmp[3] > tmp[0] && tmp[3] > tmp[1])
                continue;
            else if (tmp[2] < tmp[0] && tmp[2] < tmp[1] && tmp[3] < tmp[0] && tmp[3] < tmp[1])
                continue;
            vector<float> tmp1;
            tmp1.push_back(tmprect1[i].y);
            tmp1.push_back(tmprect1[i + 1].y);
            tmp1.push_back(tmprect2[j].y);
            tmp1.push_back(tmprect2[j + 1].y);
            if (tmp1[2] > tmp1[0] && tmp1[2] > tmp1[1] && tmp1[3] > tmp1[0] && tmp1[3] > tmp1[1])
                continue;
            else if (tmp1[2] < tmp1[0] && tmp1[2] < tmp1[1] && tmp1[3] < tmp1[0] && tmp1[3] < tmp1[1])
                continue;
            sort(tmp.begin(), tmp.end());
            auto result = findcross(line1[i], line2[j], make_pair(tmp[1], tmp[2]));
            if (result.first == 1)
                box.push_back(result.second);
        }
    }
}

void calcarea::order()
{
    findcandidata();
    if (box.size() < 2)
        return;
    sort(box.begin(), box.end(), locationsort);
    Point2f p0 = box[0];
    //<angle,point>
    vector<pair<float, Point2f>> angle_point;
    for (int i = 1; i < box.size(); i++) {
        float tmpangle = (box[i] - p0).dot(Point2f(1, 0)) / sqrtf((box[i] - p0).dot(box[i] - p0));
        angle_point.push_back(make_pair(tmpangle, box[i]));
    }
    sort(angle_point.begin(), angle_point.end(), anglesort);
    for (int i = 1; i < box.size(); i++) {
        box[i] = angle_point[i - 1].second;
    }
}

float calcarea::getarea()
{
    order();
    float result = 0;
    if (box.size() < 3)
        return result;
    for (int i = 0; i < box.size() - 1; i++) {
        result += 0.5*(box[i] - box[0]).cross(box[i + 1] - box[0]);
    }
    return result;
}

vector<Point2f> getboundingbox(const Point2f& bottom, const Point2f& left, const Point2f& right) {
    vector<Point2f> result;
    if (left.x - right.x == 0) {
        result.push_back(Point2f(bottom.x, left.y));
        result.push_back(Point2f(bottom.x, right.y));
        result.push_back(Point2f(bottom.x + (left.x - bottom.x)*1.5, right.y));
        result.push_back(Point2f(bottom.x + (right.x - bottom.x)*1.5, left.y));
        return result;
    }
    else if (left.y - right.y == 0) {
        result.push_back(Point2f(left.x, bottom.y));
        result.push_back(Point2f(right.x, bottom.y));
        result.push_back(Point2f(right.x, bottom.y + (right.y - bottom.y)*1.5));
        result.push_back(Point2f(left.x, bottom.y + (left.y - bottom.y)*1.5));
        return result;
    }
    else {
        float k1 = (left.y - right.y) / (left.x - right.x);
        float b1 = bottom.y - k1*bottom.x;
        float k_side = -(1 / k1);
        float b_bottom_left = left.y - k_side*left.x;
        float b_bottom_right = right.y - k_side*right.x;
        float bottom_left_x = (b_bottom_left - b1) / (k1 - k_side);
        float bottom_left_y = bottom_left_x*k_side + b_bottom_left;
        float bottom_right_x = (b_bottom_right - b1) / (k1 - k_side);
        float bottom_right_y = bottom_right_x*k_side + b_bottom_right;
        float top_left_x = bottom_left_x + (left.x - bottom_left_x)*1.5;
        float top_left_y = bottom_left_y + (left.y - bottom_left_y)*1.5;
        float top_right_x = bottom_right_x + (right.x - bottom_right_x)*1.5;
        float top_right_y = bottom_right_y + (right.y - bottom_right_y)*1.5;
        result.push_back(Point2f(bottom_left_x, bottom_left_y));
        result.push_back(Point2f(bottom_right_x, bottom_right_y));
        result.push_back(Point2f(top_right_x, top_right_y));
        result.push_back(Point2f(top_left_x, top_left_y));
        return result;
    }
}

vector<Point2f> getrectbox(const Point2f& bottom, const Point2f& left, const Point2f& right) {
    Point2f tmp;
    vector<Point2f> result;
    if (left.y > right.y)
        tmp = left;
    else
        tmp = right;
    result.push_back(Point2f(left.x, bottom.y));
    result.push_back(Point2f(left.x, bottom.y + (tmp.y - bottom.y)*1.5));
    result.push_back(Point2f(right.x, bottom.y + (tmp.y - bottom.y)*1.5));
    result.push_back(Point2f(right.x, bottom.y));
    return result;
}

float rectarea(const vector<Point2f>& rect) {
    assert(rect.size() == 4);
    float result = 0;
    for (int i = 1; i < 3; i++) {
        result += 0.5*(rect[i] - rect[0]).cross(rect[i + 1] - rect[0]);
    }
    if (result < 0)
        result = -result;
    return result;
}
vector<Point2f> getsquare(float* data, float col, float row) {
    Point2f max = Point2f(col - data[0], row - data[1]);
    Point2f min = Point2f(col - data[0], row - data[1]);
    for (int i = 0; i < 73; i++) {
        if (col - data[2 * i] > max.x) {
            max.x = col - data[2 * i];
        }
        if (col - data[2 * i] < min.x) {
            min.x = col - data[2 * i];
        }
        if (row - data[2 * i + 1] > max.y) {
            max.y = row - data[2 * i + 1];
        }
        if (row - data[2 * i + 1] < min.y) {
            min.y = row - data[2 * i + 1];
        }
    }

    float sizex = max.x - min.x;
    float sizey = max.y - min.y;
    max.x += sizex / 20;
    min.x -= sizex / 20;
    max.y += sizey / 20;
    min.y -= sizey / 20;

    vector<Point2f> result;
    result.push_back(Point2f(max.x, max.y));
    result.push_back(Point2f(max.x, min.y));
    result.push_back(Point2f(min.x, min.y));
    result.push_back(Point2f(min.x, max.y));
    return result;
}

pair<int, vector<float>> bounding_box_reg(const float* score, const float* box, int* data_size) {
    const float positive_thresh = 0.6;
    int count = 0;
    vector<float> result;
    if (data_size[2] % 2 == 1)
        data_size[2]++;
    if (data_size[3] % 2 == 1)
        data_size[3]++;
    moffset score_off(data_size[0], 2, (data_size[2] - 2) / 2 - 4, (data_size[3] - 2) / 2 - 4);
    moffset box_off(data_size[0], 4, (data_size[2] - 2) / 2 - 4, (data_size[3] - 2) / 2 - 4);
    for (int n = 0; n < data_size[0]; n++) {
        for (int x = 0; x < (data_size[3] - 2) / 2 - 4; x++) {
            for (int y = 0; y < (data_size[2] - 2) / 2 - 4; y++) {
                //cout << box[box_off.offset(n, 0, y, x)] << "  " << box[box_off.offset(n, 1, y, x)] << "  " << box[box_off.offset(n, 2, y, x)] << "  " << box[box_off.offset(n, 3, y, x)] << endl;
                //cout << score_off.offset(n, 1, y, x) << "   ";
                //cout << score[score_off.offset(n, 1, y, x)] <<"   "<< score[score_off.offset(n, 0, y, x)] << endl;
                if (score[score_off.offset(n, 1, y, x)] / (score[score_off.offset(n, 1, y, x)] + score[score_off.offset(n, 0, y, x)]) > positive_thresh) {
                    count++;
                    float tmp_x = static_cast<float>(x) * 2;
                    float tmp_y = static_cast<float>(y) * 2;
                    float tmp_h = 12.0;
                    float tmp_w = 12.0;
                    tmp_x += box[box_off.offset(n, 1, y, x)] * 12.0;
                    tmp_y += box[box_off.offset(n, 0, y, x)] * 12.0;
                    tmp_h += (box[box_off.offset(n, 3, y, x)] - box[box_off.offset(n, 1, y, x)])*12.0;
                    tmp_w += (box[box_off.offset(n, 2, y, x)] - box[box_off.offset(n, 0, y, x)])*12.0;
                    result.push_back(tmp_x);
                    result.push_back(tmp_y);
                    result.push_back(tmp_h);
                    result.push_back(tmp_w);
                    result.push_back(score[score_off.offset(n, 1, y, x)] / (score[score_off.offset(n, 1, y, x)] + score[score_off.offset(n, 0, y, x)]));
                }
            }
        }
    }
    return make_pair(count, result);
}

pair<vector<float>, vector<float>> net24_output_rearrange(int size, float* rate, float* data) {
    vector<float> rate_output;
    vector<float> data_output;
    for (int i = 0; i < size; i++) {
        rate_output.push_back(0);
        rate_output.push_back(rate[size + i] / (rate[i] + rate[i + size]));
        data_output.push_back(data[i]);
        data_output.push_back(data[i + size]);
        data_output.push_back(data[i + 2 * size]);
        data_output.push_back(data[i + 3 * size]);
    }
    return make_pair(rate_output, data_output);
}