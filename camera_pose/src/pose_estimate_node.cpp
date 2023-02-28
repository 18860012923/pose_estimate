#include <ros/ros.h>
#include "hogcascade.hpp"
#include <camera_pose_estimate.hpp>
using namespace std;

int main(int argc,char *argv[])  
{
    ros::init(argc,argv,"static_sub");
    ros::NodeHandle nh_;

    tf2_ros::Buffer buffer_;
    tf2_ros::TransformListener listener(buffer_);

    CameraPose camera_pose(buffer_);

    ros::Rate rate(1);
    while(ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
