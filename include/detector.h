#pragma once
#include <iostream>
#include <string>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "tracking_by_matching.hpp"
#include "detectedobject.h"

using namespace cv;
using namespace cv::dnn;
using namespace std;
using namespace cv::tbm;

class Detector
{
public:
    virtual vector<DetectedObject> Detect(Mat image) = 0 {}
};
class DnnDetector : public Detector 
{
	Net net;
	string model;
	string config;
	string labels;
	int inputWidth;
	int inputHeight;
	Scalar mean;
	double scale;
	bool swapRB;
public:
	DnnDetector(string _model, 
				string _config, 
				string _labels,
				int _inputWidth, 
				int _inputHeight, 
				Scalar _mean,
				double _scale,
				bool _swapRB);
	vector <DetectedObject> Detect(Mat image);
};


//cv::Ptr<ITrackerByMatching> createTrackerByMatchingWithFastDescriptor();
//
//class DnnObjectDetector
//{
//public:
//	DnnObjectDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
//		int desired_class_id = -1,
//		float confidence_threshold = 0.2,
//		//the following parameters are default for popular MobileNet_SSD caffe model
//		const String& net_input_name = "data",
//		const String& net_output_name = "detection_out",
//		double net_scalefactor = 0.007843,
//		const Size& net_size = Size(300, 300),
//		const Scalar& net_mean = Scalar(127.5, 127.5, 127.5),
//		bool net_swapRB = false)
//		:desired_class_id(desired_class_id),
//		confidence_threshold(confidence_threshold),
//		net_input_name(net_input_name),
//		net_output_name(net_output_name),
//		net_scalefactor(net_scalefactor),
//		net_size(net_size),
//		net_mean(net_mean),
//		net_swapRB(net_swapRB)
//	{
//		net = dnn::readNet(net_caffe_model_path, net_caffe_weights_path);
//		if (net.empty())
//			CV_Error(Error::StsError, "Cannot read Caffe net");
//	}
//	TrackedObjects detect(const cv::Mat& frame, int frame_idx)
//	{
//		Mat resized_frame;
//		resize(frame, resized_frame, net_size);
//		Mat inputBlob = cv::dnn::blobFromImage(resized_frame, net_scalefactor, net_size, net_mean, net_swapRB);
//
//		net.setInput(inputBlob, net_input_name);
//		Mat detection = net.forward(net_output_name);
//		Mat detection_as_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
//
//		TrackedObjects res;
//		for (int i = 0; i < detection_as_mat.rows; i++)
//		{
//			float cur_confidence = detection_as_mat.at<float>(i, 2);
//			int cur_class_id = static_cast<int>(detection_as_mat.at<float>(i, 1));
//			int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * frame.cols);
//			int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * frame.rows);
//			int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * frame.cols);
//			int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * frame.rows);
//
//			Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));
//
//			if (cur_confidence < confidence_threshold)
//				continue;
//			if ((desired_class_id >= 0) && (cur_class_id != desired_class_id))
//				continue;
//
//			//clipping by frame size
//			cur_rect = cur_rect & Rect(Point(), frame.size());
//			if (cur_rect.empty())
//				continue;
//
//			TrackedObject cur_obj(cur_rect, cur_confidence, frame_idx, -1);
//			res.push_back(cur_obj);
//		}
//		return res;
//	}
//private:
//	cv::dnn::Net net;
//	int desired_class_id;
//	float confidence_threshold;
//	String net_input_name;
//	String net_output_name;
//	double net_scalefactor;
//	Size net_size;
//	Scalar net_mean;
//	bool net_swapRB;
//};
//
//cv::Ptr<ITrackerByMatching>
//createTrackerByMatchingWithFastDescriptor() {
//	cv::tbm::TrackerParams params;
//
//	cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatching(params);
//
//	std::shared_ptr<IImageDescriptor> descriptor_fast =
//		std::make_shared<ResizedImageDescriptor>(
//			cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
//	std::shared_ptr<IDescriptorDistance> distance_fast =
//		std::make_shared<MatchTemplateDistance>();
//
//	tracker->setDescriptorFast(descriptor_fast);
//	tracker->setDistanceFast(distance_fast);
//
//	return tracker;
//}
