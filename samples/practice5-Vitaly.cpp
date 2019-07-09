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
void safeClass(string classId) 
{
	ofstream ofstr;

}

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

 /* if (video_name.empty() || detector_model.empty() || detector_config.empty() || detector_label.empty()
	  || classificator_model.empty() || classificator_config.empty() || classificator_label.empty())
  {
	  help();
	  return -1;
  }*/

  //String video_name = "C:\\Users\\temp2019\\GitProject\\CV-SUMMER-CAMP\\data\\topdogs.mp4";
  //int start_frame = 0;
  //int frame_step = 1;
  //String detector_model = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.caffemodel";
  //String detector_config = "C:/Users/temp2019\GitProject/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.prototxt";
  //String detector_label = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.labels";
  //int desired_class_id = 12;
  //String classificator_model ="C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.caffemodel";
  //String classificator_config ="C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.prototxt" ;
  //String classificator_label = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.labels";



  String video_name = "C:\\Users\\temp2019\\GitProject\\CV-SUMMER-CAMP\\data\\topdogs.mp4";
  int start_frame = 0;
  int frame_step = 1;
  String detector_model = "C:\\Users\\temp2019\\GitProject\\CV-SUMMER-CAMP\\data\\mobilenet-ssd\\caffe\\mobilenet-ssd.caffemodel";
  String detector_config = "C:\\Users\\temp2019\\GitProject\\CV-SUMMER-CAMP\\data\\mobilenet-ssd\\caffe\\mobilenet-ssd.prototxt";
  String detector_label = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/mobilenet-ssd/caffe/mobilenet-ssd.labels";
  int desired_class_id = 12;
  String classificator_model = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.caffemodel";
  String classificator_config = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.prototxt";
  String classificator_label = "C:/Users/temp2019/GitProject/CV-SUMMER-CAMP/data/squeezenet/1.1/caffe/squeezenet1.1.labels";



  //// If help option is given, print help message and exit.
  //if (parser.get<bool>("help")) {
  //  parser.printMessage();
  //  return 0;
  //}


	//Open Video
  Mat frame;
  VideoCapture cap;
  cap.open(video_name);
  cap.set(CAP_PROP_POS_FRAMES, start_frame);
  DnnObjectDetector detector(detector_config, detector_model, desired_class_id);
  int frame_counter = -1;
  int64 time_total = 0;
  bool paused = false;

  namedWindow("Top dogs", 1);
  cv::Ptr<ITrackerByMatching> tracker = createTrackerByMatchingWithFastDescriptor();
  while (1)
  {
		/*cap >> frame;
		DnnClassificator dnnClassificator(classificator_model, classificator_config,
			classificator_label, 227, 227, Scalar {104, 117, 123}, 0);
		Point classIdPoint;
		double confidence;
		Mat dst = dnnClassificator.Classify(frame);
		minMaxLoc(dst.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
		string nameclass=nameClassId(classificator_model, classIdPoint.x); ///Здесь будет порода
		imshow("Top dogs", frame);
		char c = (char)waitKey(1);
		if (c == 'q')
			break;*/
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

	  // Drawing colored "worms" (tracks).
	  frame = tracker->drawActiveTracks(frame);


	  // Drawing all detected objects on a frame by BLUE COLOR
	  for (const auto &detection : detections) {
		  cv::rectangle(frame, detection.rect, cv::Scalar(255, 0, 0), 3);
	  }

	  // Drawing tracked detections only by RED color and print ID and detection
	  // confidence level.
	  for (const auto &detection : tracker->trackedDetections()) {
		  cv::rectangle(frame, detection.rect, cv::Scalar(0, 0, 255), 3);
		  std::string text = std::to_string(detection.object_id) +
			  " conf: " + std::to_string(detection.confidence);
		  cv::putText(frame, text, detection.rect.tl(), cv::FONT_HERSHEY_COMPLEX,
			  1.0, cv::Scalar(0, 0, 255), 3);
	  }

	  imshow("Tracking by Matching", frame);

	  char c = (char)waitKey(2);
	  if (c == 'q')
		  break;
	  if (c == 'p')
		  paused = !paused;
  }

   
  // Do something cool.
  cout << "This is empty template sample." << endl;

  return 0;
}