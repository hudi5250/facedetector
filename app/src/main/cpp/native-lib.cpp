#include <jni.h>
#include <string>
#include <algorithm>
#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/core/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"
#include"Facedetection.h"
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "classes.h"
#define IMG_H 227
#define IMG_W 227
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "F8DEMO", __VA_ARGS__);

static FaceInception::CascadeCNN* MTCNN;

int global_p=0;
int global_r=0;
float global_time=0;
static char raw_data[MAX_DATA_SIZE];
static float input_data[MAX_DATA_SIZE];

static caffe2::Workspace ws;
static int flag=0;
// A function to load the NetDefs from protobufs.
void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}

extern "C"
void
Java_facebook_f8demo_ClassifyCamera_initCaffe2(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    alog("Attempting to load protobuf netdefs...");
    MTCNN=new FaceInception::CascadeCNN(mgr);

    alog("done.");

}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;
int detection_num=0;
vector<Rect2d> tmp_box;
extern "C"
JNIEXPORT jintArray JNICALL
Java_facebook_f8demo_ClassifyCamera_classificationFromCaffe2(
        JNIEnv *env,
        jobject /* this */,
        jint h, jint w, jbyteArray Y, jbyteArray U, jbyteArray V,
        jint rowStride, jint pixelStride,
        jboolean infer_HWC) {

    if (!MTCNN) {
        jintArray jarr=env->NewIntArray(8);
        jint* resultarray=env->GetIntArrayElements(jarr,NULL);
        resultarray[0]=-1;
        env->ReleaseIntArrayElements(jarr,resultarray,0);
        return jarr;
    }
    jsize Y_len = env->GetArrayLength(Y);
    jbyte * Y_data = env->GetByteArrayElements(Y, 0);
    assert(Y_len <= MAX_DATA_SIZE);
    jsize U_len = env->GetArrayLength(U);
    jbyte * U_data = env->GetByteArrayElements(U, 0);
    assert(U_len <= MAX_DATA_SIZE);
    jsize V_len = env->GetArrayLength(V);
    jbyte * V_data = env->GetByteArrayElements(V, 0);
    assert(V_len <= MAX_DATA_SIZE);
    Mat YUV(h*1.5,w,CV_8UC1);
    memcpy(YUV.data,(unsigned char*)Y_data,Y_len);
    memcpy(YUV.data+Y_len,(unsigned char*)U_data,Y_len/4);
    memcpy(YUV.data+Y_len+Y_len/4,(unsigned char*)V_data,Y_len/4);
    Mat rbg(h, w, CV_8UC3);
    cvtColor(YUV,rbg,CV_YUV2BGR_I420,3);
    transpose(rbg,rbg);
    flip(rbg,rbg,-1);
    //bool res=imwrite("/mnt/sdcard/test1.jpg",rbg);
    float scalew=32.0f/w;
    float scaleh=32.0f/h;
    float scale;
    if(scalew>scaleh){
        scale=scalew;
    }
    else{
        scale=scaleh;
    }
    std::vector<std::pair<Rect2d, float>> detectresult;
    vector<vector<Point2d>> points;
    std::chrono::time_point<std::chrono::system_clock> p0 = std::chrono::system_clock::now();
    if(detection_num==0) {
        detectresult = MTCNN->GetDetection(rbg, scale, {0.6, 0.7, 0.7}, true, 0.7, true,
                                                points);
    }
    else{
        if(tmp_box.size()!=0){
            vector<Mat> tmp_image;
            tmp_image.push_back(FaceInception::cropImage(rbg, tmp_box[0], Size(48, 48), INTER_LINEAR, BORDER_CONSTANT, Scalar(0)));
            //imshow("tmptmp", tmp_image[0]);
            //waitKey(0);
            detectresult = MTCNN->GetTrace1(tmp_image, tmp_box, 0.1, { 0.6,0.7,0.7 }, true, 0.7, true);
        }
    }
    std::chrono::time_point<std::chrono::system_clock> p1 = std::chrono::system_clock::now();

    detection_num++;
    if(detection_num==10){
        detection_num=0;
    }
    if(detectresult.size()==0){
        jintArray jarr=env->NewIntArray(8);
        jint* resultarray=env->GetIntArrayElements(jarr,NULL);
        resultarray[0]=-1;
        env->ReleaseIntArrayElements(jarr, resultarray, 0);
        return jarr;
    }
    tmp_box.clear();
    auto tmp_rect = detectresult[0].first;
    tmp_rect.x = tmp_rect.x - (tmp_rect.height - tmp_rect.width) / 2;
    tmp_rect.x -= 0.1*tmp_rect.height;

    tmp_rect.y -= 0.1*tmp_rect.height;


    tmp_rect.height *= 1.2;
    tmp_rect.width = tmp_rect.height;
    tmp_box.push_back(tmp_rect);
    jintArray jarr=env->NewIntArray(4*detectresult.size()+5);
    jint* resultarray=env->GetIntArrayElements(jarr,NULL);
    resultarray[0]=detectresult.size();
    for(int i=0;i<detectresult.size();i++) {
        resultarray[4*i+1] = detectresult[i].first.x;
        resultarray[4*i+2] = detectresult[i].first.y;
        resultarray[4*i+3] = detectresult[i].first.width + resultarray[4*i+1];
        resultarray[4*i+4] = detectresult[i].first.height + resultarray[4*i+2];

        resultarray[4*i+2] = resultarray[4*i+4] - detectresult[i].first.height * 0.8;
    }
    resultarray[4*detectresult.size()+1] = (float) ((p1 - p0).count()) / 1000;
    resultarray[4*detectresult.size()+2]=global_p;
    resultarray[4*detectresult.size()+3]=global_r;
    resultarray[4*detectresult.size()+4]=global_time;
    global_time=0;
    env->ReleaseIntArrayElements(jarr, resultarray, 0);
    return jarr;
}
