// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
// ncnn
#include "layer.h"
#include "net.h"
#include "benchmark.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

ncnn::Net p2pNet;

struct CrowdPoint
{
	cv::Point pt;
	float prob;
};

static void shift(int w, int h,int stride, std::vector<float> anchor_points,std::vector<float>&shifted_anchor_points)
{
	std::vector<float> x_, y_;
	for (int i = 0; i < w; i++)
	{
		float x = (i + 0.5) * stride;
		x_.push_back(x);
	}
	for (int i = 0; i < h; i++)
	{
		float y = (i + 0.5) * stride;
		y_.push_back(y);
	}

	std::vector<float> shift_x(w * h, 0), shift_y(w * h, 0);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			shift_x[i * w + j] = x_[j];
		}

	}
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			shift_y[i * w + j] = y_[i];
		}
	}

	std::vector<float> shifts(w * h * 2, 0);
	for (int i = 0; i < w * h; i++)
	{
		shifts[i * 2] = shift_x[i];
		shifts[i * 2 + 1] = shift_y[i];
	}

	shifted_anchor_points.resize(2 * w * h * anchor_points.size() / 2, 0);
	for (int i = 0; i < w * h; i++)
	{
		for (int j = 0; j < anchor_points.size() / 2; j++)
		{
			float x = anchor_points[j * 2]+shifts[i*2];
			float y = anchor_points[j * 2 + 1] + shifts[i * 2 + 1];
			shifted_anchor_points[i * anchor_points.size() / 2*2 + j * 2] = x;
			shifted_anchor_points[i * anchor_points.size() / 2*2 + j * 2+1] = y;
		}
	}
}
static void generate_anchor_points(int stride, int row,int line, std::vector<float> &anchor_points)
{
	float row_step = (float)stride / row;
	float line_step = (float)stride / line;

	std::vector<float> x_, y_;
	for (int i = 1; i < line + 1; i++)
	{
		float x = (i - 0.5) * line_step - stride / 2;
		x_.push_back(x);
	}
	for (int i = 1; i < row + 1; i++)
	{
		float y = (i - 0.5) * row_step - stride / 2;
		y_.push_back(y);
	}
	std::vector<float> shift_x(row* line,0), shift_y(row *line,0);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < line; j++)
		{
			shift_x[i * line + j] = x_[j];
		}

	}
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < line; j++)
		{
			shift_y[i * line + j] = y_[i];
		}
	}
	anchor_points.resize(row * line * 2, 0);
	for (int i = 0; i < row * line; i++)
	{
		float x = shift_x[i];
		float y = shift_y[i];
		anchor_points[i*2] = x;
		anchor_points[i * 2 + 1] = y;
	}

}
static void generate_anchor_points(int img_w, int img_h,std::vector<int> pyramid_levels,int row,int line, std::vector<float> &all_anchor_points)
{
	
	std::vector<std::pair<int, int>> image_shapes;
	std::vector<int> strides;
	for (int i = 0; i < pyramid_levels.size(); i++)
	{
		int new_h = std::floor((img_h + std::pow(2, pyramid_levels[i])-1) / std::pow(2, pyramid_levels[i]));
		int new_w = std::floor((img_w + std::pow(2, pyramid_levels[i]) - 1) / std::pow(2, pyramid_levels[i]));
		image_shapes.push_back(std::make_pair(new_w, new_h));
		strides.push_back(std::pow(2, pyramid_levels[i]));
	}
	
	all_anchor_points.clear();
	for (int i = 0; i < pyramid_levels.size(); i++)
	{
		std::vector<float> anchor_points;
		generate_anchor_points(std::pow(2, pyramid_levels[i]), row, line, anchor_points);
		std::vector<float> shifted_anchor_points;
		shift(image_shapes[i].first, image_shapes[i].second, strides[i], anchor_points, shifted_anchor_points);
		all_anchor_points.insert(all_anchor_points.end(), shifted_anchor_points.begin(),shifted_anchor_points.end());
	}

}

static int detect_crowd(const cv::Mat& bgr, std::vector<CrowdPoint>& crowd_points)
{
	int width = bgr.cols;
	int height = bgr.rows;

    int target_size = 640;
    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat input = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(input, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

	std::vector<int> pyramid_levels(1, 3);
	std::vector<float> all_anchor_points;
	generate_anchor_points(in_pad.w, in_pad.h, pyramid_levels, 2, 2, all_anchor_points);

	ncnn::Mat anchor_points = ncnn::Mat(2, all_anchor_points.size() / 2, all_anchor_points.data());


	ncnn::Extractor ex = p2pNet.create_extractor();
	const float mean_vals1[3] = { 123.675f, 116.28f,  103.53f };
	const float norm_vals1[3] = { 0.01712475f, 0.0175f, 0.01742919f };

    in_pad.substract_mean_normalize(mean_vals1, norm_vals1);

	ex.input("input", in_pad);
	ex.input("anchor", anchor_points);

	ncnn::Mat score, points;
	ex.extract("pred_scores", score);
	ex.extract("pred_points", points);

	for (int i = 0; i < points.h; i++)
	{
		float* score_data = score.row(i);
		float* points_data = points.row(i);
		if(score_data[1] > 0.5)
        {
            CrowdPoint cp;
            int x = (points_data[0] - (wpad/2))/scale;
            int y = (points_data[1] - (hpad/2))/scale;
            cp.pt = cv::Point(x, y);
            cp.prob = score_data[1];
            crowd_points.push_back(cp);
        }

	}

	return 0;
}

extern "C" {

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID x0Id;
static jfieldID y0Id;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "P2PNetNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "P2PNetNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_p2pnetncnn_P2PNetNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    p2pNet.opt = opt;
    // init param
    {
        int ret = p2pNet.load_param(mgr, "p2pnet.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "P2PNetNcnn", "load_p2pnet_param failed");
            return JNI_FALSE;
        }
        
    }

    // init bin
    {
        int ret = p2pNet.load_model(mgr, "p2pnet.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "P2PNetNcnn", "load_p2pnet_model failed");
            return JNI_FALSE;
        }
        
    }
    
    
    // init jni glue
    jclass localObjCls = env->FindClass("com/tencent/p2pnetncnn/P2PNetNcnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tencent/p2pnetncnn/P2PNetNcnn;)V");

    x0Id = env->GetFieldID(objCls, "x0", "F");
    y0Id = env->GetFieldID(objCls, "y0", "F");

    return JNI_TRUE;
}

// public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jobjectArray JNICALL Java_com_tencent_p2pnetncnn_P2PNetNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return NULL;
        //return env->NewStringUTF("no vulkan capable gpu");
    }

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_RGB);

    cv::Mat rgb = cv::Mat::zeros(in.h,in.w,CV_8UC3);
    in.to_pixels(rgb.data, ncnn::Mat::PIXEL_RGB);

    std::vector<CrowdPoint> objects; 
    detect_crowd(rgb, objects);

    // objects to Obj[]
    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);

    for (size_t i = 0; i < objects.size(); i++)
    {
        jobject jObj = env->NewObject(objCls, constructortorId, thiz);
        float x0 = objects[i].pt.x;
        float y0 = objects[i].pt.y;

        env->SetFloatField(jObj, x0Id, x0);
        env->SetFloatField(jObj, y0Id, y0);

        env->SetObjectArrayElement(jObjArray, i, jObj);

    }

    return jObjArray;
}

}
