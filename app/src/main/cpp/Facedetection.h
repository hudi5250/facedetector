//
// Created by yjd1 on 2018/5/2.
//

#ifndef AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_FACEDETECTION_H
#define AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_FACEDETECTION_H
#include <fstream>
#include <thread>
#include<memory>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>


#include"caffe_sane.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include"boundingbox.h"
#include"calcarea.h"
#undef assert
#define assert(_Expression) if(!((_Expression)))printf("error: %s %d : %s\n", __FILE__, __LINE__, (#_Expression))
#define USE_GPU_MAT 0



extern float global_time;
using namespace cv;
using namespace std;
const int g_batchsize = 500;
const int kHeightStart = 640;
const int kWidthStart = 480;
const int kMaxNet12Num = 20;
extern int global_p;
extern int global_r;
namespace FaceInception {
    class CascadeCNN {
    public:

        CascadeCNN(AAssetManager *_mgr) :scale_decay_(0.707),mgr(_mgr) {
            Network_net12 = loadNet("det1_init.pb", "det1_predict.pb");
            Network_net24 = loadNet("det2_init.pb", "det2_predict.pb");
            Network_net48 = loadNet("det3_init.pb", "det3_predict.pb");
        }
        ~CascadeCNN() {
            caffeDestroyNetwork(Network_net12);
            caffeDestroyNetwork(Network_net24);
            caffeDestroyNetwork(Network_net48);

        }

        //Only work for small images.


        vector<pair<Rect2d, float>> getNet12Proposal(Mat& input_image, double min_confidence = 0.6, double start_scale = 1,
                                                     bool do_nms = true, double nms_threshold = 0.3) {
            int short_side = min(input_image.cols, input_image.rows);
            assert(log(12.0 / start_scale / (double)short_side) / log(scale_decay_) < kMaxNet12Num);
            std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();
            vector<double> scales;
            double scale = start_scale;
            if (floor(input_image.rows * scale) < 1200 && floor(input_image.cols * scale) < 1200) {
                scales.push_back(scale);
            }
            do {
                scale *= scale_decay_;
                if (floor(input_image.rows * scale) < 1200 && floor(input_image.cols * scale) < 1200) {
                    scales.push_back(scale);
                }
            } while (floor(input_image.rows * scale * scale_decay_) >= 12 && floor(input_image.cols * scale * scale_decay_) >= 12);

            vector<vector<pair<Rect2d, float>>> sub_rects(scales.size());
#if USE_GPU_MAT
            cuda::GpuMat gpu_input_image;
			gpu_input_image.upload(input_image);
#endif

            for (int s = 0; s < scales.size(); s++) {
                //std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
                //net12_thread_group.create_thread([&]() {
                Mat small_image;

#if USE_GPU_MAT
                cuda::GpuMat gpu_small_image;
				cuda::resize(gpu_input_image, gpu_small_image, Size(0, 0), scales[s], scales[s]);
				gpu_small_image.download(small_image);
#else
                resize(input_image, small_image, Size(0, 0), scales[s], scales[s]);
#endif
                int bounding_box_size = 0;
                const float *bounding_box_data = NULL;

                vector<float> tmp_data = image_preprosess(small_image);
                //for (int i = 0; i < 2000; i++) {
                //	cout << tmp_data[i] << "   " << tmp_data[3468 + i] << "  " << tmp_data[3468 * 2 + i] << endl;
                //}
                //cout << small_image.cols << "  " << small_image.rows << endl;
                int inputnet12[4] = { 1,3,small_image.cols,small_image.rows };


                std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
                auto tmp_output_data = caffeRunNetwork(Network_net12, tmp_data.data(), inputnet12, 4);
                std::chrono::time_point<std::chrono::system_clock> p2 = std::chrono::system_clock::now();
                global_time+=(float)std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count() / 1000;
                //assert(output_size == 2);

                auto net12_result = bounding_box_reg(tmp_output_data[0], tmp_output_data[1], inputnet12);
                bounding_box_size = net12_result.first;
                bounding_box_data = net12_result.second.data();

                //cout << s << "th:   " << bounding_box_size << endl<<endl;
                //for (int i = 0; i < bounding_box_size; i++) {
                //	cout << bounding_box_data[5 * i] << "  " << bounding_box_data[5 * i + 1] << "  " << bounding_box_data[5 * i + 2] << "  " << bounding_box_data[5 * i + 3] << endl;
                //}

                if (!(bounding_box_size == 1 && bounding_box_data[0] == 0)) {
                    vector<pair<Rect2d, float>> before_nms;
                    for (int i = 0; i < bounding_box_size; i++) {
                        Rect2d this_rect = Rect2d(bounding_box_data[i * 5 + 1] / scales[s], bounding_box_data[i * 5] / scales[s],
                                                  bounding_box_data[i * 5 + 3] / scales[s], bounding_box_data[i * 5 + 2] / scales[s]);
                        before_nms.push_back(make_pair(this_rect, bounding_box_data[i * 5 + 4]));
                    }
                    if (do_nms && before_nms.size() > 1) {
                        vector<int> picked = soft_nms_max(before_nms, 0.5, min_confidence);
                        for (auto p : picked) {
                            //cout << before_nms[p].first << " " << before_nms[p].second << endl;
                            sub_rects[s].push_back(before_nms[p]);
                        }
                    }
                    else {
                        sub_rects[s].insert(sub_rects[s].end(), before_nms.begin(), before_nms.end());
                    }
                }

                //});
                //cout << "scale:" << scales[s] << " time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - p1).count() / 1000 << "ms" << endl;
            }
            //net12_thread_group.join_all();
            vector<pair<Rect2d, float>> accumulate_rects;
            for (int s = 0; s < scales.size(); s++) {
                //cout << "scale:" << scales[s] << " rects:" << sub_rects[s].size() << endl;
                accumulate_rects.insert(accumulate_rects.end(), sub_rects[s].begin(), sub_rects[s].end());
            }
            vector<pair<Rect2d, float>> result;
            if (do_nms) {
                //std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
                vector<int> picked = soft_nms_max(accumulate_rects, nms_threshold, min_confidence);
                for (auto& p : picked) {
                    //make_rect_square(rect_for_test[p].first);
                    result.push_back(accumulate_rects[p]);
                }
                //nms_avg(rects, scores, nms_threshold);
                //cout << "nms time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - p1).count() / 1000 << "ms" << endl;
            }
            else {
                result = accumulate_rects;
            }

            //for (auto& rect : result) {
            //  if (rect.second > 0.6) {
            //    rectangle(input_image, rect.first, Scalar(0, 255, 0), 1);
            //  }
            //}
            //
            //imshow("proposal", input_image);
            //waitKey(0);


            return result;
        }

        vector<pair<Rect2d, float>> getNet24Refined(vector<Mat> sub_images, vector<Rect2d> image_boxes, double min_confidence = 0.7,
                                                    bool do_nms = true, double nms_threshold = 0.3,
                                                    int batch_size = 500,
                                                    bool output_points = false, vector<vector<Point2d>> points = vector<vector<Point2d>>()) {
            int num = sub_images.size();
            if (num == 0) return vector<pair<Rect2d, float>>();
            assert(sub_images[0].cols == 24 && sub_images[0].rows == 24);
            vector<pair<Rect2d, float>> rect_and_scores;
            vector<vector<Point2d> > allPoints;

            int total_iter = ceil((double)num / (double)batch_size);
            for (int i = 0; i < total_iter; i++) {
                int start_pos = i * batch_size;
                if (i == total_iter - 1) batch_size = num - (total_iter - 1) * batch_size;
                vector<Mat> net_input = vector<Mat>(sub_images.begin() + start_pos, sub_images.begin() + start_pos + batch_size);

                int net24_output_size = 0;
                const float* net24_output_data = NULL;
                const float* net24_output_rate = NULL;

                int net24_input_size[4] = { batch_size,3,net_input[0].cols,net_input[0].rows };
                auto net24_input_data = image_preprocess(net_input);
                //std::vector<float*> tmp_output_data;
                //for (int jf = 0; jf < 10000; jf++) {
                //Network_net24 = loadNet("F:\\det2_init.pb", "F:\\det2_predict.pb", net24_input_size, 4);

                std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
                auto tmp_output_data = caffeRunNetwork(Network_net24, net24_input_data.data(), net24_input_size, 4);
                std::chrono::time_point<std::chrono::system_clock> p2 = std::chrono::system_clock::now();
                global_time+=(float)std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count() / 1000;
                //assert(output_size == 2);
                //caffeDestroyNetwork(Network_net24);
                //}

                net24_output_size = batch_size;
                //auto net24_rearage_output = net24_output_rearrange(batch_size, tmp_output_data[1], tmp_output_data[0]);
                net24_output_data = tmp_output_data[0];

                net24_output_rate = tmp_output_data[1];



                for (int j = 0; j < net24_output_size; j++) {
                    //cout << net24output["Prob"].data[j * 2 + 1] << endl;
                    if ((net24_output_rate[j * 2 + 1] / (net24_output_rate[j * 2] + net24_output_rate[j * 2 + 1])) > min_confidence) {
                        Rect2d this_rect = Rect2d(image_boxes[start_pos + j].x + image_boxes[start_pos + j].width * net24_output_data[j * 4 + 0],
                                                  image_boxes[start_pos + j].y + image_boxes[start_pos + j].height * net24_output_data[j * 4 + 1],
                                                  image_boxes[start_pos + j].width + image_boxes[start_pos + j].width * (net24_output_data[j * 4 + 2] - net24_output_data[j * 4 + 0]),
                                                  image_boxes[start_pos + j].height + image_boxes[start_pos + j].height * (net24_output_data[j * 4 + 3] - net24_output_data[j * 4 + 1]));
                        rect_and_scores.push_back(make_pair(this_rect, (net24_output_rate[j * 2 + 1] / (net24_output_rate[j * 2] + net24_output_rate[j * 2 + 1]))));
                        //rects.push_back(this_rect);
                        //scores.push_back(net24output["Prob"].data[j * 2 + 1]);
                        //if (output_points) {
                        //  vector<Point2d> point_list;
                        //  for (int p = 0; p < 5; p++) {
                        //    point_list.push_back(Point2d((net24output["conv5-2"].data[j * 10 + p * 2] + 12) / 24 * image_boxes[start_pos + j].width + image_boxes[start_pos + j].x,
                        //      (net24output["conv5-2"].data[j * 10 + p * 2 + 1] + 12) / 24 * image_boxes[start_pos + j].height + image_boxes[start_pos + j].y));
                        //  }
                        //  allPoints.push_back(point_list);
                        //}

                    }
                }


            }
            //if (output_points) assert(allPoints.size() == rect_and_scores.size());
            vector<pair<Rect2d, float>> result;

            if (do_nms) {
                //std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
                vector<int> picked = soft_nms_max(rect_and_scores, nms_threshold, min_confidence);
                for (auto& p : picked) {
                    result.push_back(rect_and_scores[p]);
                    //if (output_points) points.push_back(allPoints[p]);
                }
            }
            else {
                result = rect_and_scores;
            }

            return result;
        }

        vector<pair<Rect2d, float>> getNet48Final(vector<Mat> sub_images, vector<Rect2d> image_boxes, double min_confidence = 0.7,
                                                  bool do_nms = true, double nms_threshold = 0.3,
                                                  int batch_size = 500,
                                                  bool output_points = false, vector<vector<Point2d>> points = vector<vector<Point2d>>()) {
            int num = sub_images.size();
            if (num == 0) return vector<pair<Rect2d, float>>();
            assert(sub_images[0].rows == 48 && sub_images[0].cols == 48);
            vector<pair<Rect2d, float>> rect_and_scores;
            vector<vector<Point2d> > allPoints;

            int total_iter = ceil((double)num / (double)batch_size);
            for (int i = 0; i < total_iter; i++) {
                int start_pos = i * batch_size;
                if (i == total_iter - 1) batch_size = num - (total_iter - 1) * batch_size;
                vector<Mat> net_input = vector<Mat>(sub_images.begin() + start_pos, sub_images.begin() + start_pos + batch_size);
                int net48_output_size = batch_size;
                const float* net48_output_rate = NULL;
                const float* net48_output_bbbox = NULL;
                const float* net48_output_landmark = NULL;

                int net48_input_size[4] = { batch_size,3,net_input[0].cols,net_input[0].rows };
                auto net48_input_data = image_preprocess(net_input);


                std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
                auto tmp_output_data = caffeRunNetwork(Network_net48, net48_input_data.data(), net48_input_size, 4);
                std::chrono::time_point<std::chrono::system_clock> p2 = std::chrono::system_clock::now();
                global_time+=(float)std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count() / 1000;
                //assert(output_size == 2);

                net48_output_rate = tmp_output_data[0];
                net48_output_bbbox = tmp_output_data[1];
                net48_output_landmark = tmp_output_data[2];

                //auto net24_rearage_output = net24_output_rearrange(batch_size, tmp_output_data[1], tmp_output_data[0]);





                for (int j = 0; j < net48_output_size; j++) {
                    //cout << net48output["Prob"].data[j * 2 + 1] << endl;
                    //cout << net48_output_rate[j * 2 + 1] / (net48_output_rate[j * 2 + 1] + net48_output_rate[j * 2]) << endl;
                    if ((net48_output_rate[j * 2 + 1] / (net48_output_rate[j * 2 + 1] + net48_output_rate[j * 2])) > min_confidence) {
                        Rect2d this_rect = Rect2d(image_boxes[start_pos + j].x + image_boxes[start_pos + j].width * net48_output_bbbox[j * 4 + 0],
                                                  image_boxes[start_pos + j].y + image_boxes[start_pos + j].height * net48_output_bbbox[j * 4 + 1],
                                                  image_boxes[start_pos + j].width + image_boxes[start_pos + j].width * (net48_output_bbbox[j * 4 + 2] - net48_output_bbbox[j * 4 + 0]),
                                                  image_boxes[start_pos + j].height + image_boxes[start_pos + j].height * (net48_output_bbbox[j * 4 + 3] - net48_output_bbbox[j * 4 + 1]));
                        rect_and_scores.push_back(make_pair(this_rect, (net48_output_rate[j * 2 + 1] / (net48_output_rate[j * 2 + 1] + net48_output_rate[j * 2]))));
                        //rects.push_back(this_rect);
                        //scores.push_back(net48output["conv6-3"].data[j * 2 + 1]);
                        if (output_points) {
                            vector<Point2d> point_list;
                            for (int p = 0; p < 5; p++) {
                                point_list.push_back(Point2d(net48_output_landmark[j * 10 + p] * image_boxes[start_pos + j].width + image_boxes[start_pos + j].x,
                                                             net48_output_landmark[j * 10 + p + 5] * image_boxes[start_pos + j].height + image_boxes[start_pos + j].y));
                            }
                            allPoints.push_back(point_list);
                        }

                    }
                }

            }
            if (output_points) assert(allPoints.size() == rect_and_scores.size());
            vector<pair<Rect2d, float>> result;
            if (do_nms) {
                //std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
                vector<int> picked = soft_nms_max(rect_and_scores, nms_threshold, min_confidence, IoU_MIN);
                for (auto& p : picked) {
                    result.push_back(rect_and_scores[p]);
                    if (output_points) points.push_back(allPoints[p]);
                }
                //cout << "nms time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - p1).count() / 1000 << "ms" << endl;
                //nms_avg(rects, scores, nms_threshold);
                //for (auto& p : rects) {
                //  result.push_back(make_pair<Rect2d, float>(Rect2d(p.x, p.y, p.width, p.height), 1.0f));
                //}
            }
            else {
                result = rect_and_scores;
            }


            return result;
        }


        vector<pair<Rect2d, float>> GetDetection(Mat& input_image, double start_scale = 1, vector<double> confidence_threshold = { 0.6, 0.7,0.7 },
                                                 bool do_nms = true, double nms_threshold = 0.7,
                                                 bool output_points = false, vector<vector<Point2d>> points = vector<vector<Point2d>>()) {
            Mat clone_image = input_image.clone();//for drawing
            cout << start_scale << endl;
            //std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
            auto proposal = getNet12Proposal(clone_image, confidence_threshold[0], start_scale, do_nms, nms_threshold);
            //std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();
            //cout << "proposal time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p1 - p0).count() / 1000 << "ms" << endl;
            cout << "proposal: " << proposal.size() << endl;
            global_p=proposal.size();
            //hd
            if(proposal.size()>5){
                //proposal=vector<pair<Rect2d,float>>(proposal.begin(),proposal.begin()+5);
            }
            if (proposal.size() == 0) return vector<pair<Rect2d, float>>();
            vector<Mat> sub_images;
            sub_images.reserve(proposal.size());
            vector<Rect2d> image_boxes;
            image_boxes.reserve(proposal.size());
            for (auto& p : proposal) {
                make_rect_square(p.first);
                //fixRect(p.first, input_image.size(), true);
                if (p.first.width < 9 || p.first.height < 9) continue;
                Mat sub_image = cropImage(input_image, p.first, Size(24, 24), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
                sub_images.push_back(sub_image);
                image_boxes.push_back(p.first);
                //Mat image_show = input_image.clone();
                //for (auto& rect : image_boxes) {
                //    rectangle(image_show, rect, Scalar(0, 255, 0), 1);
                //}
                //imshow("ha", sub_image);
                //imshow("refined", image_show);
                //waitKey(0);
            }
            std::chrono::time_point<std::chrono::system_clock> p2 = std::chrono::system_clock::now();
            //cout << "gen_list time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p2 - p1).count() / 1000 << "ms" << endl;
            auto refined = getNet24Refined(sub_images, image_boxes, confidence_threshold[1], do_nms, nms_threshold, 500,output_points,points);
            std::chrono::time_point<std::chrono::system_clock> p3 = std::chrono::system_clock::now();
            //cout << "refine time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p3 - p2).count() / 1000 << "ms" << endl;
            cout << "refined: " << refined.size() << endl;
            global_r=refined.size();
            //hd
            if(refined.size()>1){
                //refined=vector<pair<Rect2d,float>>(refined.begin(),refined.begin()+1);
            }
            if (refined.size() == 0) return vector<pair<Rect2d, float>>();
            //Mat image_show = input_image.clone();
            //for (auto& rect : refined) {
            //    rectangle(image_show, rect.first, Scalar(0, 255, 0), 1);
            //}

            //imshow("refined", image_show);
            //waitKey(0);

            vector<Mat> sub_images48;
            sub_images48.reserve(refined.size());
            vector<Rect2d> image_boxes48;
            image_boxes48.reserve(refined.size());
            //int start = 0;
            for (auto& p : refined) {
                make_rect_square(p.first);
                //fixRect(p.first, input_image.size(), true);
                if (p.first.width < 9 || p.first.height < 9) continue;
                Mat sub_image = cropImage(input_image, p.first, Size(48, 48), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

                sub_images48.push_back(sub_image);
                image_boxes48.push_back(p.first);
            }
            //if(refined.size()!=0){
            //    return vector<pair<Rect2d, float>>({ make_pair(image_boxes[0],1) });
           // }
            auto final = getNet48Final(sub_images48, image_boxes48, confidence_threshold[2], do_nms, nms_threshold, 500, output_points, points);
           // std::chrono::time_point<std::chrono::system_clock> p4 = std::chrono::system_clock::now();
            //cout << "final time:" << (float)std::chrono::duration_cast<std::chrono::microseconds>(p4 - p3).count() / 1000 << "ms" << endl;
            cout << "final: " << final.size() << endl;
            std::chrono::time_point<std::chrono::system_clock> p5 = std::chrono::system_clock::now();

            //cout<<"total time:"<< (float)std::chrono::duration_cast<std::chrono::microseconds>(p5 - p0).count() / 1000 << "ms" << endl;
            return final;
        }

        vector<pair<Rect2d, float>> GetTrace1( vector<Mat> sub_images, vector<Rect2d> image_boxes, double start_scale = 1, vector<double> confidence_threshold = { 0.6, 0.7,0.7 },
                                               bool do_nms = false, double nms_threshold = 0.7,
                                               bool output_points = false) {
            for (auto &i : sub_images) {
                Mat tmp = i;
                resize(tmp, tmp, Size(48, 48), 0, 0, INTER_LINEAR);
                i = tmp;
            }
            vector<vector<Point2d>> points = vector<vector<Point2d>>();
            auto final = getNet48Final(sub_images, image_boxes, 0.1, do_nms, nms_threshold, 500, output_points, points);
            return final;
        }

        void *loadNet(const string& net, const string& model) {
            AAsset* asset1 = AAssetManager_open(mgr, net.c_str(), AASSET_MODE_BUFFER);
            assert(asset1 != nullptr);
            const void *data_net = AAsset_getBuffer(asset1);
            assert(data_net != nullptr);
            off_t len_net = AAsset_getLength(asset1);
            assert(len_net != 0);
            AAsset* asset2 = AAssetManager_open(mgr, model.c_str(), AASSET_MODE_BUFFER);
            assert(asset2 != nullptr);
            const void *data_model = AAsset_getBuffer(asset2);
            assert(data_net != nullptr);
            off_t len_model = AAsset_getLength(asset2);
            assert(len_model != 0);
            void* result=caffeLoadNetwork(data_net,len_net,data_model,len_model);
            AAsset_close(asset2);
            AAsset_close(asset1);


            return result;
        }

        vector<float> boundingboxreg(vector<float> proba, vector<float> box, float positive_thresh) {
            const int real_receptive_field = 12;
            const int stride = 2;

        }

        vector<float> image_preprosess(const Mat& input_image) {
            Mat image;
            input_image.convertTo(image, CV_32FC3, 0.0078125, -0.99609375);

            cv::cvtColor(image, image, CV_BGR2RGB);
            transpose(image, image);

            std::vector<cv::Mat> channels(3);
            cv::split(image, channels);
            std::vector<float> tmp_data;
            for (auto &c : channels) {
                tmp_data.insert(tmp_data.end(), (float *)c.datastart, (float *)c.dataend);
            }
            return tmp_data;
        }

        vector<float> image_preprocess(const vector<Mat>& input_image) {
            std::vector<float> tmp_data;
            for (auto m : input_image) {
                Mat image;
                m.convertTo(image, CV_32FC3, 0.0078125, -0.99609375);
                cv::cvtColor(image, image, CV_BGR2RGB);
                transpose(image, image);

                std::vector<cv::Mat> channels(3);
                cv::split(image, channels);
                for (auto &c : channels) {
                    tmp_data.insert(tmp_data.end(), (float *)c.datastart, (float *)c.dataend);
                }
            }
            return tmp_data;
        }


        AAssetManager *mgr;
        void *Network_net12, *Network_net12_stitch, *Network_net24, *Network_net48, *Network_netLoc;
        unsigned char* data_net12, *data_net12_stitch, *data_net24, *data_net48, *data_netLoc;
        unsigned char* data_net12_model, *data_net12_stitch_model, *data_net24_model, *data_net48_model, *data_netLoc_model;
        int net12, net12_stitch, net24, net48, netLoc;
        float scale_decay_;
        int input_width, input_height;
    };
}
#endif //AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_FACEDETECTION_H
