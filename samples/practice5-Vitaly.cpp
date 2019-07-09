#include <string>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "tracking_by_matching.hpp"
#include "classificator.h"
#include "detector.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace cv::tbm;

const char* cmdAbout =
    "This is an empty application that can be treated as a template for your "
    "own doing-something-cool applications.";

const char* cmdOptions =
{
	"{video_name       | | video name								}"
	"{start_frame      |0| Start frame								}"
	"{frame_step       |1| Frame step								}"
	"{detector_model   | | Path to detector's Caffe *.prototxt		}"
	"{detector_config | | Path to detector's Caffe *.model			}"
	"{detector_label   | |Path to detector's Caffe *.label			}"
	"{desired_class_id |-1| The desired class that should be tracked}"
	"{classi_model	   | | Path to classificator's *.prototxt		}"
	"{classi_config	   | | Path to classificator's *.model			}"
	"{classi_label	   | | Path to classificator's *.label			}"
};

void help() 
{
	cout << "there is something here " << endl;
}
string nameClassId(string label,int classId) 
{
	
	ifstream file(label);
	string str;
	int n = 0;
	while (!file.eof())
	{
		getline(file, str);
		if (n == classId) {
			break;
		}
		n++;
	}
	file.close();
	return str;	
}

const string label_mobilenet[21] = { "background",
"aeroplane",
"bicycle",
"bird",
"boat",
"bottle",
"bus",
"car",
"cat",
"chair",
"cow",
"diningtable",
"dog",
"horse",
"motorbike",
"person",
"pottedplant",
"sheep",
"sofa",
"train",
"tvmonitor"
};

class DnnObjectDetector
{
public:
	vector<int> classId;
	DnnObjectDetector(const String& net_caffe_model_path, const String& net_caffe_weights_path,
		int desired_class_id = -1,
		float confidence_threshold = 0.2,
		//the following parameters are default for popular MobileNet_SSD caffe model
		const String& net_input_name = "data",
		const String& net_output_name = "detection_out",
		double net_scalefactor = 0.007843,
		const Size& net_size = Size(300, 300),
		const Scalar& net_mean = Scalar(127.5, 127.5, 127.5),
		bool net_swapRB = false)
		:desired_class_id(desired_class_id),
		confidence_threshold(confidence_threshold),
		net_input_name(net_input_name),
		net_output_name(net_output_name),
		net_scalefactor(net_scalefactor),
		net_size(net_size),
		net_mean(net_mean),
		net_swapRB(net_swapRB)
	{
		net = dnn::readNet(net_caffe_model_path, net_caffe_weights_path);
		if (net.empty())
			CV_Error(Error::StsError, "Cannot read Caffe net");
	}
	TrackedObjects detect(const cv::Mat& frame, int frame_idx)
	{
		Mat resized_frame;
		resize(frame, resized_frame, net_size);
		Mat inputBlob = cv::dnn::blobFromImage(resized_frame, net_scalefactor, net_size, net_mean, net_swapRB);

		net.setInput(inputBlob, net_input_name);
		Mat detection = net.forward(net_output_name);
		Mat detection_as_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		TrackedObjects res;
		for (int i = 0; i < detection_as_mat.rows; i++)
		{
			float cur_confidence = detection_as_mat.at<float>(i, 2);
			int cur_class_id = static_cast<int>(detection_as_mat.at<float>(i, 1));
			int x_left = static_cast<int>(detection_as_mat.at<float>(i, 3) * frame.cols);
			int y_bottom = static_cast<int>(detection_as_mat.at<float>(i, 4) * frame.rows);
			int x_right = static_cast<int>(detection_as_mat.at<float>(i, 5) * frame.cols);
			int y_top = static_cast<int>(detection_as_mat.at<float>(i, 6) * frame.rows);

			Rect cur_rect(x_left, y_bottom, (x_right - x_left), (y_top - y_bottom));

			if (cur_confidence < confidence_threshold)
				continue;
			if ((desired_class_id >= 0) && (cur_class_id != desired_class_id))
				continue;

			//clipping by frame size
			cur_rect = cur_rect & Rect(Point(), frame.size());
			if (cur_rect.empty())
				continue;

			TrackedObject cur_obj(cur_rect, cur_confidence, frame_idx, -1);
			res.push_back(cur_obj);
		}
		return res;
	}
private:
	cv::dnn::Net net;
	int desired_class_id;
	float confidence_threshold;
	String net_input_name;
	String net_output_name;
	double net_scalefactor;
	Size net_size;
	Scalar net_mean;
	bool net_swapRB;
};

cv::Ptr<ITrackerByMatching>
createTrackerByMatchingWithFastDescriptor() {
	cv::tbm::TrackerParams params;

	cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatching(params);

	std::shared_ptr<IImageDescriptor> descriptor_fast =
		std::make_shared<ResizedImageDescriptor>(
			cv::Size(16, 32), cv::InterpolationFlags::INTER_LINEAR);
	std::shared_ptr<IDescriptorDistance> distance_fast =
		std::make_shared<MatchTemplateDistance>();

	tracker->setDescriptorFast(descriptor_fast);
	tracker->setDistanceFast(distance_fast);

	return tracker;
}





int main(int argc, const char** argv) {
  // Parse command line arguments.
  CommandLineParser parser(argc, argv, cmdOptions);
  parser.about(cmdAbout);
 /* String video_name = parser.get<String>("video_name");
  int start_frame = parser.get<int>("start_frame");
  int frame_step = parser.get<int>("frame_step");
  String detector_model = parser.get<String>("detector_model");
  String detector_config = parser.get<String>("detector_config");
  String detector_label = parser.get<String>("detector_label");
  int desired_class_id = parser.get<int>("desired_class_id");
  String classificator_model = parser.get<String>("classi_model");
  String classificator_config = parser.get<String>("classi_config");
  String classificator_label = parser.get<String>("classi_label");*/


  //For class
  /*String video_name = "C:\\Users\\temp2019\\GitProject\\CV-SUMMER-CAMP\\data\\topdogs.mp4";
  int start_frame = 0;
  int frame_step = 1;
  String detector_model = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel";
  String detector_config = "C:/Users/temp2019\GitProject/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.prototxt";
  String detector_label = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.labels";
  int desired_class_id = 12;
  String classificator_model ="C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.caffemodel";
  String classificator_config ="C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.prototxt" ;
  String classificator_label = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.labels";*/

  //For home
  
  String video_name = "D:/IntelComputerVision/CV-SUMMER-CAMP/data/topdogs.mp4";
  int start_frame = 0;
  int frame_step = 1;
  String detector_model = "D:/IntelComputerVision/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel";
  String detector_config = "D:/IntelComputerVision/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.prototxt";
  String detector_label = "D:/IntelComputerVision\\CV-SUMMER-CAMP/data/mobilenet-ssd/caffemobilenet-ssd.labels";
  int desired_class_id = 12;
  String classificator_model = "D:\\IntelComputerVision\\CV-SUMMER-CAMP\\data\\squeezenet\\1.1\\caffe\\squeezenet1.1.caffemodel";
  String classificator_config = "D:\\IntelComputerVision\\CV-SUMMER-CAMP\\data\\squeezenet\\1.1\\caffe\\squeezenet1.1.prototxt";
  String classificator_label = "D:\\IntelComputerVision\\CV-SUMMER-CAMP\\data\\squeezenet\\1.1\\caffe\\squeezenet1.1.labels";



  if (video_name.empty() || detector_model.empty() || detector_config.empty() || detector_label.empty()
	  || classificator_model.empty() || classificator_config.empty() || classificator_label.empty())
  {
	  help();
	  return -1;
  }

  //ERROR
  //// If help option is given, print help message and exit.
  //if (parser.get<bool>("help")) {
  //  parser.printMessage();
  //  return 0;
  //}


  //create mas labels for squeezenet
  string label_squeezenet[1000];
  int count = 0;
  ifstream ifs(classificator_label);
  while (count < 1000)
  {
		getline(ifs,label_squeezenet[count]);
		count++;
  }
  
	//Open Video
  Mat frame;
  VideoCapture cap;
    if (cap.open(video_name))
	  cout << "Work!" << endl;
  cap.set(CAP_PROP_POS_FRAMES, start_frame);


  if (!cap.isOpened())
  {
	  help();
	  cout << "***Could not initialize capturing...***\n";
	  cout << "Current parameter's value: \n";
	  parser.printMessage();
	  return -1;
  }

  DnnObjectDetector detector(detector_config, detector_model, desired_class_id);
  DnnClassificator dnnClassificator(classificator_model, classificator_config,
					 classificator_label, 227, 227, Scalar {104, 117, 123}, 0);
  
  int frame_counter = -1;
  int64 time_total = 0;
  bool paused = false;

  namedWindow("Top dogs", 1);
  cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();

  while (1)
  {
		vector<int> v(130);

		if (paused)
		{
			char c = (char)waitKey(30);
			if (c == 'p')
				paused = !paused;
			if (c == 'q')
				break;
			continue;
		}

		cap >> frame;
		
		if (frame.empty()) {
			break;
		}
		frame_counter++;
		if (frame_counter < start_frame)
			continue;
		if (frame_counter % frame_step != 0)
			continue;

		int64 frame_time = getTickCount();
		TrackedObjects detections = detector.detect(frame, frame_counter);
		// timestamp in milliseconds
		uint64_t cur_timestamp = static_cast<uint64_t>(1000.0 / 30 * frame_counter);
		tracker->process(frame, detections, cur_timestamp);

		frame_time = getTickCount() - frame_time;
		time_total += frame_time;

		// Drawing colored "worms" (tracks);
		int i = 0;
		Mat croped_image;

		// Drawing all detected objects on a frame by BLUE COLOR
		// Drawing tracked detections only by RED color and print ID and detection
		// confidence level.
		for (const auto &detection : tracker->trackedDetections()) {
			cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 1);
			std::string text = (label_mobilenet[detector.classId[i]]) +
				" conf: " + std::to_string(detection.confidence);
			cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_COMPLEX,
				1.0, cv::Scalar(0, 0, 255), 1);
			Rect crop = detection.rect;
			croped_image = frame(crop);
			Mat res = dnnClassificator.Classify(croped_image);
			double confidence;
			Point classIdPoint;
			minMaxLoc(res, 0, &confidence, 0, &classIdPoint);

			if (classIdPoint.x > 150 && classIdPoint.x < 277) {
				cout << "Conf:" << confidence << endl << "Class:" << label_mobilenet[classIdPoint.x] << endl;
				if (find(v.begin(), v.end(), classIdPoint.x) == v.end()) {
					v.push_back(classIdPoint.x);
					string text = "../Dogs/" + label_mobilenet[classIdPoint.x] + ".png";
					imwrite(text, croped_image);
				}
			}

		}


		imshow("Top dogs", frame);
		char c = (char)waitKey(2);
		if (c == 'q')
			break;
		if (c == 'p')
			paused = !paused;
	 
  }

   
  // Do something cool.
  double s = frame_counter / (time_total / getTickFrequency());
  printf("FPS: %f\n", s);
  return 0;
}