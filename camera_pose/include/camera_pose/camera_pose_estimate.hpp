#pragma once
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <geometry_msgs/Pose.h>
#include <hogcascade.hpp>
#include <tf/tf.h>
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include <tf2_ros/transform_broadcaster.h>
#include "geometry_msgs/PointStamped.h"
#include <geometry_msgs/PoseArray.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <geometry_msgs/TransformStamped.h>
#include <yaml-cpp/yaml.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ros/package.h>

using namespace std;
using namespace cv;

template <class M, class T>
ros::SubscribeOptions getSubscribeOptions(
    const std::string &topic, uint32_t queue_size,
    void (T::*fp)(const boost::shared_ptr<M const> &),
    T* obj,
    ros::CallbackQueueInterface* queue,
    const ros::TransportHints &transport_hints = ros::TransportHints()) {
    ros::SubscribeOptions ops;
    ops.template init<M>(topic, queue_size, boost::bind(fp, obj, _1));
    ops.callback_queue = queue;
    ops.transport_hints = transport_hints;
    return ops;
}

class CameraPose{
public:
    CameraPose(tf2_ros::Buffer& buffer);
    ~CameraPose();

    void solve_camera_pose();

    void solve_robot_pose();

    void read_video();

    bool calculcorners(const cv::Mat& srcImage,vector<cv::Point2f>& corners,int cornums);

    void set_poses_of_signs();
    void codeRotateByX(double y, double z, double thetax, double& outy, double& outz);
    void codeRotateByY(double y, double z, double thetay, double& outy, double& outz);
    void codeRotateByZ(double y, double z, double thetaz, double& outy, double& outz);
    void geometry2Point3D(vector<geometry_msgs::PointStamped>& geo,vector<cv::Point3f>& poi);
    void pub_signs_pose();


    void print_poses_of_signs();
    void print_poses_of_robot();
    void pub_poses_of_robot();

    void Cam_RGB_Callback(const sensor_msgs::ImageConstPtr& msg);
    void image_callback_thread();

    void arrow_detection(const Mat& faceROI,vector<Point2f>& corners,int fx,int fy,bool& rev_flag);
    void square_detection(const Mat& faceROI,vector<Point2f>& corners,int fx,int fy,bool rev_flag);
private:
    vector<geometry_msgs::PointStamped> realtransPoints; //reference points on the sign
    vector<geometry_msgs::TransformStamped>  poses_of_signs; //The poses of all signs
    vector<geometry_msgs::PoseStamped>  poses_of_camera; //All estimated camera poses
    vector<tf2::Transform>  poses_of_robot; //All estimated robot poses
    vector<vector<cv::Point3f>> Points3D; //所有牌子上的3D点
    vector<vector<cv::Point2f>> Points2D; //所有框中的2D角点坐标

    cv::HOGCascadeClassifier hogclassifier,squareclassifier,arrowclassifier;

    cv::String window_name;
    cv_bridge::CvImagePtr cv_ptr;
    VideoCapture testVideo;
    Mat videoImg;
    tf2_ros::Buffer& buffer;

    YAML::Node signs,points;


    double camD[9] = {
		533.802379, 0, 319.134879,
		0, 533.800444, 239.505738,
		0, 0, 1 };
	cv::Mat camera_matrix;

	//畸变参数
    double distCoeffD[5] = { -0.002413, 0.001889, -0.000103, -0.000037, 0.0000000 };
	cv::Mat distortion_coefficients;

    ros::NodeHandle private_nh_;
    ros::Publisher particlecloud_pub_;
    ros::Subscriber rgb_sub;
    boost::thread* m_image_thread;

    tf2_ros::StaticTransformBroadcaster signs_pub; 
    tf2_ros::TransformBroadcaster robot_pose_pub;
    boost::thread* tf_pub_thread_;
    boost::thread* pose_cacul_thread_;
    bool set_poses_of_signs_flag;

    ros::CallbackQueue m_image_queue;


    string pack_path;
};