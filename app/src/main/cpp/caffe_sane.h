//
// Created by yjd1 on 2018/5/2.
//

#ifndef AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_CAFFE_SANE_H
#define AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_CAFFE_SANE_H

#pragma once
#ifndef __CAFFE_SANE_H
#define __CAFFE_SANE_H

#if defined(_WIN32)
#define YH_OS_WIN
#elif defined(__APPLE__) && defined(__MACH__)
#define YH_OS_MAC
#else
#define YH_OS_LNX
#endif

#if defined(YH_OS_WIN)
#include <windows.h>
	#undef min
	#undef max
#elif defined(YH_OS_MAC)
// http://developer.apple.com/qa/qa2004/qa1398.html
	#include <mach/mach_time.h>
	#include <unistd.h>
#elif defined(YH_OS_LNX)
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#endif

#if defined(YH_OS_WIN)
inline void myusleep(__int64 usec)
  {
    HANDLE timer;
    LARGE_INTEGER ft;

    ft.QuadPart = -(10 * usec); // Convert to 100 nanosecond interval, negative value indicates relative time

    timer = CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &ft, 0, NULL, NULL, 0);
    WaitForSingleObject(timer, INFINITE);
    CloseHandle(timer);
  }
#else
inline void myusleep(useconds_t  waitTime)
{
    usleep(waitTime);
}


#endif


#ifdef __cplusplus
extern "C"{
#endif

/**
\brief Load a caffe network
\param data_init points to the content of xxxx_init_net.pb
\param size_init is the size of xxxx_init_net.pb
\param data_predict points to the content of xxxx_predict_net.pb
\param size_predict is the size of xxxx_predict_net.pb
\param input_dims points to the input data dimensions: [1,n_channels,height,width]
\param n_input_dims should be 4 for now
\return A network handle
*/
void* caffeLoadNetwork(const void* data_init,int size_init,const void* data_predict,int size_predict);
/**
\brief Run a caffe network
\param handle is the handle returned by caffeLoadNetwork
\param input_data points to the input image stored in Caffe 2 order - n_channels*height*width
\param poutput_data receives a pointer that points to the output layer, stored in Caffe 2 order
\return The output layer size in number of floats
*/
const std::vector<float*>& caffeRunNetwork(void* handle, float* input_data, int* input_dims, int n_input_dims);

void* caffeCreateThread();
int caffeRunThread(void* handle);
int caffeDestroyThread(void* handle);

int caffeSetModelThread(void* handle, void* model);
int caffeSetInputThread(void* handle, float* input, int size);
int caffeGetOutputThread(void* handle, float** poutput_data);
int caffeRunNetworkThread(void* handle, float* input_data, float** poutput_data, void* thread);

/**
\brief Destroy a caffe network
\param handle is the handle returned by caffeLoadNetwork
*/
void caffeDestroyNetwork(void* handle);
#ifdef __cplusplus
}
#endif
#endif

#endif //AICAMERA_58F32CD596303AFFB95BF2E1876196AA268A717B_CAFFE_SANE_H
