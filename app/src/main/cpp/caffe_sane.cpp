//
// Created by yjd1 on 2018/5/2.
//

#include <stdlib.h>
#include <vector>
#include<iostream>
#include<fstream>
#include "caffe2/core/init.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/timer.h"
#include "caffe_sane.h"
#ifdef _WIN32
#include <windows.h>
	#undef USE_STL_THREAD
	#undef USE_PTHREAD
#endif

#if defined(__APPLE__)
#define USE_STL_THREAD
	#include <thread>
	#include <mutex>
#endif

#if defined(__ANDROID__) || defined(ANDROID)
#define USE_PTHREAD
#include "pthread.h"
#endif
//

static int g_caffe_inited = 0;
static int caffeLazyInit() {
    if (!g_caffe_inited) {
        char* argv_data[] = { "./caffe" };
        int argc = 1;
        char** argv = argv_data;
        g_caffe_inited = 1;
        caffe2::GlobalInit(&argc, &argv);
        return 1;
    }
    return 0;
}

class CCaffeContext {
public:
    caffe2::Predictor* predictor;
    caffe2::TensorCPU* input;
    caffe2::Predictor::TensorVector* inputVec;
    std::vector<float*> current_output;
    std::vector<int> output_size;
    CCaffeContext() {
        predictor = nullptr;
        input = nullptr;
        inputVec = nullptr;

    }
    ~CCaffeContext() {
        if (current_output.size()!=0) {
            for (auto p : current_output) {
                delete[] p;
            }

            //current_output.clear();
            //output_size.clear();
            //std::vector<float*>(current_output).swap(current_output);
            //std::vector<int>(output_size).swap(output_size);
        }
        if (inputVec != nullptr) { delete inputVec; inputVec = nullptr; }
        if (input != nullptr) { delete input; input = nullptr; }
        if (predictor!=nullptr) { delete predictor; predictor = nullptr; }
    }
};

struct cnn_thread_data {
    int  thread_id;
    void *handle;
    std::vector<float> input_data;
    //float *input_data;
    float *poutput_data;
};
#ifdef USE_PTHREAD
class CCaffeThread {
public:
    pthread_t m_thread;
    pthread_mutex_t m_mutex;
    struct cnn_thread_data cnn_data;
    int output_size;
    int m_shouldThreadStop;
    CCaffeThread() {
        cnn_data.handle = NULL;
        //cnn_data.input_data = NULL;
        cnn_data.input_data.resize(0);
        cnn_data.poutput_data = NULL;
        cnn_data.thread_id = 1;
        m_shouldThreadStop = 0;
        output_size = 0;
    }
    ~CCaffeThread() {

        cnn_data.handle = NULL;
        //cnn_data.input_data = NULL;
        cnn_data.poutput_data = NULL;
        threadRelease();
    }


    int threadSetInput(float* input, int size)
    {
        if (!input) return 0;

        if (cnn_data.input_data.size() == size)
        {
            pthread_mutex_lock(&m_mutex);
            memcpy(cnn_data.input_data.data(), input, sizeof(float)*size);
            pthread_mutex_unlock(&m_mutex);
            return 1;
        }
        else
        {
            pthread_mutex_lock(&m_mutex);
            cnn_data.input_data.resize(size);
            memcpy(cnn_data.input_data.data(), input, sizeof(float)*size);
            pthread_mutex_unlock(&m_mutex);
            return 1;
        }
        return 0;
    }

    int threadSetModel(void* handle)
    {
        if (handle)
        {
            pthread_mutex_lock(&m_mutex);
            cnn_data.handle = handle;
            pthread_mutex_unlock(&m_mutex);
            return 1;
        }
        return 0;
    }

    int threadGetOutput(float** poutput)
    {
        if (cnn_data.poutput_data)
        {
            pthread_mutex_unlock(&m_mutex);
            *poutput = cnn_data.poutput_data;
            pthread_mutex_unlock(&m_mutex);
        }
        return output_size;
    }

    static void* threadFunc(void* arg)
    {

        CCaffeThread* thread = (CCaffeThread*)arg;
        cnn_thread_data* cnn_data = &(thread->cnn_data);

        // break condition
        while (true)
        {
            //std::cout << "run cnn" << "\n";
            myusleep(1);
            if (cnn_data && cnn_data->handle &&cnn_data->input_data.size()>0 && thread)
            {
                //std::cout << "bless2" << "\n";
                thread->output_size = caffeRunNetworkThread(cnn_data->handle, cnn_data->input_data.data(), &cnn_data->poutput_data, thread);
                /*std::cout << "c++" << " " << thread->output_size << " " << "finish run cnn thread 1.\n";
                for (int i = 0; i < thread->output_size; i++)
                    std::cout << cnn_data->poutput_data[i] << " ";
                std::cout << "\n"; */
            }

            if (thread->m_shouldThreadStop)
            {
                pthread_exit(NULL);
                break;
            }
        } // end while

        return nullptr;
    }

    int threadInit()
    {

        m_mutex = PTHREAD_MUTEX_INITIALIZER;
        memset(&m_thread, 0, sizeof(m_thread));

        if (int r = pthread_create(&m_thread, nullptr, threadFunc, this))
        {
            std::cout << "Thread creation failed." << std::endl;
            return 0;
        }
        return 1;
    }

    int threadRelease()
    {
        pthread_mutex_lock(&m_mutex);
        m_shouldThreadStop = 1;
        pthread_mutex_unlock(&m_mutex);

        //if (cnn_data.input_data) delete cnn_data.input_data;
        //cnn_data.input_data = NULL;

        pthread_join(m_thread, nullptr);
        cnn_data.input_data.resize(0);
        return 1;
    }

};
#else
#ifdef USE_STL_THREAD

class CCaffeThread {
public:

	std::thread m_thread;

	//pthread_t m_thread;
	std::mutex m_mutex;
	struct cnn_thread_data cnn_data;
	int output_size;
	int m_shouldThreadStop;
	CCaffeThread() {
		cnn_data.handle = NULL;
		//cnn_data.input_data = NULL;
		cnn_data.input_data.shrink_to_fit();
		cnn_data.input_data.resize(0);
		cnn_data.poutput_data = NULL;
		cnn_data.thread_id = 1;
		m_shouldThreadStop = 0;
		output_size = 0;
	}
	~CCaffeThread() {

		cnn_data.handle = NULL;
		cnn_data.poutput_data = NULL;
		threadRelease();
	}


	int threadSetInput(float* input, int size)
	{
		if (!input) return 0;

		if (cnn_data.input_data.size() == size)
		{
			m_mutex.lock();
			memcpy(cnn_data.input_data.data(), input, sizeof(float)*size);
			m_mutex.unlock();
			return 1;
		}
		else
		{
			m_mutex.lock();
			cnn_data.input_data.resize(size);
			memcpy(cnn_data.input_data.data(), input, sizeof(float)*size);
			m_mutex.unlock();
			return 1;
		}
		return 0;
	}

	int threadSetModel(void* handle)
	{
		if (handle)
		{
			m_mutex.lock();
			cnn_data.handle = handle;
			m_mutex.unlock();
			return 1;
		}
		return 0;
	}

	int threadGetOutput(float** poutput)
	{
		if (cnn_data.poutput_data)
		{
			m_mutex.lock();
			*poutput = cnn_data.poutput_data;
			m_mutex.unlock();
		}
		return output_size;
	}

	static void* threadFunc(void* arg)
	{

		CCaffeThread* thread = (CCaffeThread*)arg;
		cnn_thread_data* cnn_data = &(thread->cnn_data);

		// break condition
		while (true)
		{
			myusleep(1);
			if (cnn_data && cnn_data->handle &&cnn_data->input_data.size()>0 && thread)
			{
				thread->output_size = caffeRunNetworkThread(cnn_data->handle, cnn_data->input_data.data(), &cnn_data->poutput_data, thread);
			}

			if (thread->m_shouldThreadStop)
			{
				break;
			}
		} // end while

		return nullptr;
	}

	int threadInit()
	{
		m_thread = std::thread(threadFunc, this);
		return 1;
	}

	int threadRelease()
	{
		m_mutex.lock();
		m_shouldThreadStop = 1;
		m_mutex.unlock();
		m_thread.join();
		return 1;
	}

};


#else

class CCaffeThread {
	/*
public:

	HANDLE m_thread;
	HANDLE m_mutex;
	struct cnn_thread_data cnn_data;
	int output_size;
	int m_shouldThreadStop;
	CCaffeThread() {
		cnn_data.handle = NULL;
		cnn_data.input_data.shrink_to_fit();
		cnn_data.input_data.resize(0);
		cnn_data.poutput_data = NULL;
		cnn_data.thread_id = 1;
		m_shouldThreadStop = 0;
		output_size = 0;
	}
	~CCaffeThread() {

		cnn_data.handle = NULL;
		cnn_data.poutput_data = NULL;
		threadRelease();
	}


	int threadSetInput(float* input, int size)
	{
		if (!input) return 0;

		if (cnn_data.input_data.size() == size)
		{
			WaitForSingleObject(m_mutex, INFINITE);
			memcpy(cnn_data.input_data.data(), input, sizeof(float)*size);
			ReleaseMutex(m_mutex);
			return 1;
		}
		else
		{
			WaitForSingleObject(m_mutex, INFINITE);
			cnn_data.input_data.resize(size);
			memcpy(cnn_data.input_data.data(), input, sizeof(float)*size);
			ReleaseMutex(m_mutex);
			return 1;
		}
		return 0;
	}

	int threadSetModel(void* handle)
	{
		if (handle)
		{
			WaitForSingleObject(m_mutex, INFINITE);
			cnn_data.handle = handle;
			ReleaseMutex(m_mutex);
			return 1;
		}
		return 0;
	}

	int threadGetOutput(float** poutput)
	{
		if (cnn_data.poutput_data)
		{
			WaitForSingleObject(m_mutex, INFINITE);
			*poutput = cnn_data.poutput_data;
			ReleaseMutex(m_mutex);
		}
		return output_size;
	}

	static void* threadFunc(void* arg)
	{

		CCaffeThread* thread = (CCaffeThread*)arg;
		cnn_thread_data* cnn_data = &(thread->cnn_data);

		// break condition
		while (true)
		{
			myusleep(1);
			if (cnn_data && cnn_data->handle &&cnn_data->input_data.size()>0 && thread)
			{
				thread->output_size = caffeRunNetworkThread(cnn_data->handle, cnn_data->input_data.data(), &cnn_data->poutput_data, thread);

			}

			if (thread->m_shouldThreadStop)
			{
				break;
			}
		} // end while

		return nullptr;
	}

	int threadInit()
	{

		m_thread = CreateThread(NULL, NULL, (LPTHREAD_START_ROUTINE)threadFunc, this, NULL, NULL);
		m_mutex = CreateMutex(NULL, FALSE, NULL);
		return 1;
	}

	int threadRelease()
	{
		WaitForSingleObject(m_mutex, INFINITE);
		m_shouldThreadStop = 1;
		ReleaseMutex(m_mutex);

		WaitForSingleObject(m_thread, INFINITE);
		CloseHandle(m_mutex);
		return 1;
	}
	*/
};
#endif
#endif
void* caffeLoadNetwork(const void* data_init, int size_init,const void* data_predict, int size_predict) {
    caffeLazyInit();
    /////////////////////////
    caffe2::NetDef init_net, predict_net;
    //auto regis=caffe2::gDeviceTypeRegistry();
    init_net.ParseFromArray(data_init, size_init);
    predict_net.ParseFromArray(data_predict, size_predict);
    CCaffeContext* pcaffe_ctx = new CCaffeContext();
    /////////////////////////
    /*
    for(int i=0;i<predict_net.op_size();i++){
        if ("Conv" == predict_net.op(i).type()){
            predict_net.mutable_op(i)->set_engine("NNPACK");
            //std::cout << "Conv engine used:" << predict_net.op(i).has_engine() << std::endl;
            //std::cout << "Conv engine set:" << predict_net.op(i).engine() << std::endl;

        }
        if("ConvTranspose"==predict_net.op(i).type()){
            predict_net.mutable_op(i)->set_engine("BLOCK");
            //std::cout << "Conv engine used:" << predict_net.op(i).has_engine() << std::endl;
            //std::cout << "Conv engine set:" << predict_net.op(i).engine() << std::endl;

        }
    }
*/
    pcaffe_ctx->predictor = new caffe2::Predictor(init_net, predict_net);
    auto see=pcaffe_ctx->predictor->def().DebugString();
    return (void*)pcaffe_ctx;
}

const std::vector<float*>& caffeRunNetwork(void* handle, float* input_data,int* input_dims,int n_input_dims) {
    caffe2::Timer t_time;
    //clock_t start_time = clock();
    CCaffeContext* pcaffe_ctx = (CCaffeContext*)handle;
    std::vector<caffe2::TIndex> dims;
    for (int i = 0; i < n_input_dims; i++) {
        dims.push_back(static_cast<caffe2::TIndex>(input_dims[i]));
    }
    if (pcaffe_ctx->input == nullptr&&pcaffe_ctx->inputVec == nullptr) {
        pcaffe_ctx->input = new caffe2::TensorCPU(dims);
        pcaffe_ctx->inputVec = new caffe2::Predictor::TensorVector({ pcaffe_ctx->input });

    }
    else {

        pcaffe_ctx->input->Resize(dims);

    }

    pcaffe_ctx->input->ShareExternalPointer(input_data, 0);

    /////////////
    caffe2::Predictor::TensorVector outputVec;
    pcaffe_ctx->predictor->run(*pcaffe_ctx->inputVec, &outputVec);
    //std::cout << pcaffe_ctx->predictor->ws()->Nets()[0] << std::endl;


    int size = outputVec.size();
    for (int i = 0; i < size; i++) {
        caffe2::TensorCPU* output = outputVec[i];
        if (pcaffe_ctx->current_output.size() <= i) {
            pcaffe_ctx->current_output.push_back(NULL);
            pcaffe_ctx->output_size.push_back(static_cast<int>(output->size()));
            pcaffe_ctx->current_output[i] = new float[pcaffe_ctx->output_size[i]];
        }
        if (pcaffe_ctx->output_size[i] < static_cast<int>(output->size())) {
            pcaffe_ctx->output_size[i] = static_cast<int>(output->size());
            delete[] pcaffe_ctx->current_output[i];
            pcaffe_ctx->current_output[i] = new float[pcaffe_ctx->output_size[i]];
        }
        memcpy(pcaffe_ctx->current_output[i], output->data<float>(), static_cast<int>(output->size()) * sizeof(float));

    }
    return pcaffe_ctx->current_output;
    /*
    poutput_data = (float***)calloc(size, sizeof(float**));
    caffe2::TensorCPU* output = outputVec[0];
    pcaffe_ctx->
    /////////////
    if (!pcaffe_ctx->current_output) {
        pcaffe_ctx->output_size = (int)output->size();
        pcaffe_ctx->current_output = (float*)calloc(pcaffe_ctx->output_size + 1, sizeof(float));
    }
    memcpy(pcaffe_ctx->current_output, output->data<float>(), pcaffe_ctx->output_size * sizeof(float));
    *poutput_data = pcaffe_ctx->current_output;
    //clock_t end_time = clock();
    //std::cout << static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC * 1000 << "ms\n";
    *(*poutput_data + pcaffe_ctx->output_size) = float(t_time.MilliSeconds());
    //std::cout << "conv:" << t_time.MilliSeconds() << "ms\n";
    return (pcaffe_ctx->output_size + 1);
    */

}



void* caffeCreateThread() {
    CCaffeThread* pcaffe_thread = new CCaffeThread();
    return (void*)pcaffe_thread;
}

int caffeSetModelThread(void* handle, void* model) {
    //CCaffeThread* pcaffe_thread = (CCaffeThread*)handle;
    //return pcaffe_thread->threadSetModel(model);
    return 0;
}

int caffeSetInputThread(void* handle, float* input, int size) {
    //CCaffeThread* pcaffe_thread = (CCaffeThread*)handle;
    //return pcaffe_thread->threadSetInput(input, size);
    return 0;
}

int caffeGetOutputThread(void* handle, float** poutput_data) {
    //CCaffeThread* pcaffe_thread = (CCaffeThread*)handle;
    //return pcaffe_thread->threadGetOutput(poutput_data);
    return 0;
}

int caffeRunThread(void* handle) {
    //CCaffeThread* pcaffe_thread = (CCaffeThread*)handle;
    //return pcaffe_thread->threadInit();
    return 0;
}

int caffeDestroyThread(void* handle) {
    //CCaffeThread* pcaffe_thread = (CCaffeThread*)handle;
    //return pcaffe_thread->threadRelease();
    //delete pcaffe_thread;
    return 0;
}

int caffeRunNetworkThread(void* handle, float* input_data, float** poutput_data, void* thread) {
    //
    //Not implemented
    //
    /*
    caffe2::Timer t_time;
    //clock_t start_time = clock();
    CCaffeContext* pcaffe_ctx = (CCaffeContext*)handle;
    pcaffe_ctx->input->ShareExternalPointer(input_data, 0);
    /////////////
    caffe2::Predictor::TensorVector outputVec;
    pcaffe_ctx->predictor->run(*pcaffe_ctx->inputVec, &outputVec);
    caffe2::TensorCPU* output = outputVec[0];
    /////////////
    if (!pcaffe_ctx->current_output) {
        pcaffe_ctx->output_size = (int)output->size();
        pcaffe_ctx->current_output = (float*)calloc(pcaffe_ctx->output_size + 1, sizeof(float));
    }
    if (!thread)
    {
        return 0;
    }


#ifdef USE_PTHREAD
    pthread_mutex_lock(&((CCaffeThread*)thread)->m_mutex);
#else
    #ifdef USE_STL_THREAD
        ((CCaffeThread*)thread)->m_mutex.lock();
    #else
        WaitForSingleObject(((CCaffeThread*)thread)->m_mutex, INFINITE);
    #endif
#endif

    memcpy(pcaffe_ctx->current_output, output->data<float>(), pcaffe_ctx->output_size * sizeof(float));
    *poutput_data = pcaffe_ctx->current_output;
    *(*poutput_data + pcaffe_ctx->output_size) = float(t_time.MilliSeconds());

#ifdef USE_PTHREAD
    pthread_mutex_unlock(&((CCaffeThread*)thread)->m_mutex);
#else
    #ifdef USE_STL_THREAD
        ((CCaffeThread*)thread)->m_mutex.unlock();
    #else
        ReleaseMutex(((CCaffeThread*)thread)->m_mutex);
    #endif
#endif

    return (pcaffe_ctx->output_size + 1);
    */
    return 0;
}


void caffeDestroyNetwork(void* handle) {
    delete (CCaffeContext*)handle;
}

