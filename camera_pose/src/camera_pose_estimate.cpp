#include "camera_pose_estimate.hpp"
using namespace std;
using namespace cv;

struct cmp{
	bool operator()(const Point2f &a, const Point2f &b) {
		if (a.x > b.x) {
			return false;
		}
		else {
			return true;
		}
	}
};

struct cmpy{
	bool operator()(const Point2f &a, const Point2f &b) {
		if (a.y > b.y) {
			return false;
		}
		else {
			return true;
		}
	}
};


CameraPose::CameraPose(tf2_ros::Buffer& buffer_): 
			buffer(buffer_)
{
	pack_path = ros::package::getPath("camera_pose");
	particlecloud_pub_ = private_nh_.advertise<geometry_msgs::PoseArray>("camera_cloud", 1000);
	//cout << "----+++------" << endl;
	ros::SubscribeOptions ops = getSubscribeOptions("/image_raw",1,&CameraPose::Cam_RGB_Callback,this, &m_image_queue);//image_topic为订阅的图像话题名称
	rgb_sub = private_nh_.subscribe(ops);
	
	camera_matrix = Mat(3, 3, CV_64FC1, camD);
	//畸变参数
	distortion_coefficients = Mat(5, 1, CV_64FC1, distCoeffD);
	cv_ptr = boost::make_shared <cv_bridge::CvImage	> ();
	//读取每个牌子上设定好的角点;  
	points = YAML::LoadFile(pack_path + "/param/points.yaml");
	realtransPoints.resize(points["points"].size());
	for(int j = 0;j < realtransPoints.size();++j){
		realtransPoints[j].point.x = points["points"][j]["pose"]["position"]["x"].as<double>();
		realtransPoints[j].point.y = 0;
		realtransPoints[j].point.z = points["points"][j]["pose"]["position"]["z"].as<double>();
	}
	//读取所有牌子相对于地图所在的位置 
	signs = YAML::LoadFile(pack_path + "/param/param_real.yaml");
	poses_of_signs.resize(signs["signpoints"].size());
	Points3D.resize(signs["signpoints"].size());

	window_name = "Exit Sign Detection";
	testVideo = VideoCapture(0);//调用0号摄像头
	
    videoImg  = imread("/home/zzw/下载/opencv3_CascadeHOG/src/10.jpg");
    cv::pyrDown(videoImg,videoImg,Size(videoImg.cols/2, videoImg.rows/2));
    cv::pyrDown(videoImg,videoImg,Size(videoImg.cols/2, videoImg.rows/2));
    cv::pyrDown(videoImg,videoImg,Size(videoImg.cols/2, videoImg.rows/2));
    String filename = "/home/zzw/下载/opencv3_CascadeHOG/data/cascade_new.xml";
	hogclassifier.load(filename);

	//加载箭头识别与方框识别器
	filename = "/home/zzw/下载/opencv3_CascadeHOG/data-man/cascade-man.xml";
	squareclassifier.load(filename);
	filename = "/home/zzw/下载/opencv3_CascadeHOG/data-arr/cascade.xml";
	arrowclassifier.load(filename);
	//cout << "----------" << endl;
    tf_pub_thread_ = new boost::thread(boost::bind(&CameraPose::pub_signs_pose, this));
	//m_image_thread = new boost::thread(boost::bind(&CameraPose::image_callback_thread,this));//这里，Rotation是该类的类名
	pose_cacul_thread_ = new boost::thread(boost::bind(&CameraPose::solve_robot_pose, this));
	set_poses_of_signs_flag = false;
	
}

// void CameraPose::image_callback_thread()
// {
// 	cout << "----------" << endl;
//     m_image_queue.callAvailable(ros::WallDuration(0.4));
// }


CameraPose::~CameraPose()
{
	tf_pub_thread_->interrupt();
    tf_pub_thread_->join();

    delete tf_pub_thread_;
	pose_cacul_thread_->interrupt();
    pose_cacul_thread_->join();

    delete pose_cacul_thread_;
	if (NULL != m_image_thread) 
	{
		m_image_thread->interrupt();
		m_image_thread->join();
		delete m_image_thread;
	}
}

bool CameraPose::calculcorners(const Mat& srcImage,vector<Point2f>& corners,int cornums)
{
	if (srcImage.empty())
	{
		printf("could not load image..\n");
		return false;
	}
	
	//2、Shi-Tomasi算法：确定图像强角点
	int maxcorners = cornums;
	double qualityLevel = 0.1;  //角点检测可接受的最小特征值
	double minDistance = 20;	//角点之间最小距离
	int blockSize = 3;//计算导数自相关矩阵时指定的领域范围
	double  k = 0.4; //权重系数
 
	goodFeaturesToTrack(srcImage, corners, maxcorners, qualityLevel, minDistance, Mat(), blockSize, false, k);

	return true;
}

void CameraPose::solve_camera_pose()
{
	for(int i = 0;i < Points2D.size();++i)
	{
		for(int j = 0;j < Points3D.size();++j)
		{
			//Points2D[i]表示每个矩形框里面识别的角点位姿，是在当时看到的图片的像素位置
			//Points3D[j]表示每个已知的牌子角点在map坐标系下的位姿
			//初始化输出矩阵
			cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
			cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

			solvePnP(Points3D[j], Points2D[i], camera_matrix, distortion_coefficients, rvec, tvec, false, 0);
			//Gao的方法可以使用任意四个特征点，特征点数量不能少于4也不能多于4,这里强制认定为4+3个角点

			//旋转向量变旋转矩阵
			//提取旋转矩阵
			double rm[9];
			cv::Mat rotM(3, 3, CV_64FC1, rm);
			Rodrigues(rvec, rotM);
			double r11 = rotM.ptr<double>(0)[0];
			double r12 = rotM.ptr<double>(0)[1];
			double r13 = rotM.ptr<double>(0)[2];
			double r21 = rotM.ptr<double>(1)[0];
			double r22 = rotM.ptr<double>(1)[1];
			double r23 = rotM.ptr<double>(1)[2];
			double r31 = rotM.ptr<double>(2)[0];
			double r32 = rotM.ptr<double>(2)[1];
			double r33 = rotM.ptr<double>(2)[2];

			/*************************************此处计算出相机的旋转角**********************************************/
			//计算出相机坐标系的三轴旋转欧拉角，旋转后可以转出世界坐标系。
			//旋转顺序为z、y、x
			//原理见帖子：
			double thetaz = -1 * atan2(r21, r11) / CV_PI * 180;
			double thetay = -1 * atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / CV_PI * 180;
			double thetax = -1 * atan2(r32, r33) / CV_PI * 180;
			
			cout << "相机的三轴旋转角：" << thetax << ", " << thetay << ", " << thetaz << endl;
			// fout.close();
			/*************************************此处计算出相机的旋转角END**********************************************/
			/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置**********************************************/
			/* 当原始坐标系经过旋转z、y、x三次旋转后，会与世界坐标系完全平行，而三次旋转中向量OcOw会跟着旋转 */
			/* 而我们想知道的是两个坐标系完全平行时，OcOw的值 */
			/* 因此，原始坐标系每次旋转完成后，对向量OcOw进行一次反相旋转，最终可以得到两个坐标系完全平行时的OcOw */
			/* 该向量乘以-1就是世界坐标系下相机的坐标 */
			/***********************************************************************************/

			//提出平移矩阵，表示从相机坐标系原点，跟着向量(x,y,z)走，就到了世界坐标系原点
			double tx = tvec.ptr<double>(0)[0];
			double ty = tvec.ptr<double>(0)[1];
			double tz = tvec.ptr<double>(0)[2];

			//x y z 为唯一向量在相机原始坐标系下的向量值
			//也就是向量OcOw在相机坐标系下的值
			double x = tx, y = ty, z = tz;

			//进行三次反向旋转
			codeRotateByZ(x, y, thetaz, x, y);
			codeRotateByY(x, z, thetay, x, z);
			codeRotateByX(y, z, thetax, y, z);


			//获得相机在世界坐标系下的位置坐标
			//即向量OcOw在世界坐标系下的值
			double Cx = x*-1;
			double Cy = y*-1;
			double Cz = z*-1;

			// ofstream fout2("D:\\pnp_t.txt");
			// fout2 << Cx << endl << Cy << endl << Cz << endl;
			cout << "相机的世界坐标：" << Cx << ", " << Cy << ", " << Cz << endl;
			// fout2.close();
			/*************************************此处计算出相机坐标系原点Oc在世界坐标系中的位置END**********************************************/

			//将计算的位姿存到poses_of_camera中
			geometry_msgs::PoseStamped tempose;
			tempose.header.frame_id = "map";
			tempose.header.stamp = ros::Time::now();
			tempose.pose.position.x = Cx;
			tempose.pose.position.y = Cy;
			tempose.pose.position.z = Cz;
			tempose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(thetax, thetay, thetaz);
			poses_of_camera.emplace_back(tempose);
			//重投影测试位姿解是否正确
			// vector<cv::Point2f> projectedPoints;
			// Points3D[i].push_back(cv::Point3f(0, 100, 105));
			// cv::projectPoints(Points3D, rvec, tvec, *camera_matrix, *distortion_coefficients, projectedPoints);
		}
	}
}


void CameraPose::Cam_RGB_Callback(const sensor_msgs::ImageConstPtr& msg)
{
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
		//imshow(window_name, cv_ptr->image);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
}


void CameraPose::solve_robot_pose()
{
	ros::Rate rate(10);
	while(ros::ok())
	{
		read_video();
		if(!set_poses_of_signs_flag)
		{
			cout << "读取逃生出口位置..."<<endl;
			set_poses_of_signs();
			//cout << "打印读取逃生出口位置..."<<endl;
			print_poses_of_signs();
			set_poses_of_signs_flag = true;
		}
		try  
		{  
			if(Points2D.size())
			{
				auto tem = buffer.lookupTransform("base_link","base_camera_link",ros::Time::now());
				tf2::Transform tem_tf2;
				tf2::convert(tem.transform,tem_tf2);
				//cout << "求解相机坐标..."<<endl;
				solve_camera_pose();
				poses_of_robot.resize(poses_of_camera.size());
				for(int i = 0;i < poses_of_camera.size();++i)
				{
					tf2::Transform pose_camera;
  					tf2::convert(poses_of_camera[i].pose, pose_camera);
					poses_of_robot[i] = tem_tf2.inverse() * pose_camera;
				}
				cout << "打印求解相机坐标..."<<endl;
				print_poses_of_robot();
				cout << "发布相机坐标..."<<endl;
				pub_poses_of_robot();
			}
			else continue;
		}  
		catch(const std::exception& e)  
		{  
			std::cerr << e.what() << '\n';  
		}
		rate.sleep();
	} 
}

void CameraPose::read_video()
{
	
	//while (true) {//循环不断获取新的图片并显示出来
		testVideo.read(videoImg);//读取摄像头的照片并存放在Mat类型的videoImg中
		m_image_queue.callAvailable(ros::WallDuration(0.4));
		videoImg = cv_ptr->image;
		if(!videoImg.data) return;
        std::vector<Rect> xpiders;
        Mat frame_gray;
        cvtColor(videoImg, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);
        hogclassifier.detectMultiScale(frame_gray, xpiders,1.25, 9, 0, Size(15, 15));


		Points2D.clear();
        for(auto&rect :xpiders)
		{
			//rectangle(videoImg,rect,(255,255,255),2);
			//在每张图上检测角点
			Mat faceROI = frame_gray(rect);
			vector<Point2f> corners_arrow;
			vector<Point2f> corners_square;
			bool rev_flag;
			arrow_detection(faceROI,corners_arrow,rect.x,rect.y,rev_flag);
			square_detection(faceROI,corners_square,rect.x,rect.y,rev_flag);
			vector<Point2f> corners;
			corners.insert(corners.end(),corners_square.begin(),corners_square.end());
			corners.insert(corners.end(),corners_arrow.begin(),corners_arrow.end());
			if(corners.size() != 7) 
			{
				ROS_WARN("sign detection fail......");
				continue;
			}
			Points2D.push_back(corners);
		}
		Mat edge,grayImage;
		cvtColor(videoImg, grayImage, COLOR_BGR2GRAY);
		threshold(grayImage, grayImage, 50, 255, THRESH_BINARY);
		blur(grayImage, edge, Size(3, 3));
		Canny(edge, edge, 128, 200, 3);
		imshow("Canny", edge);
		imshow(window_name, videoImg);//将图片显示出来
		waitKey(20);//相邻每帧图片间隔时间（单位：ms)
	//}
}

// void CameraPose::read_video()
// {
	
// 	//while (true) {//循环不断获取新的图片并显示出来
// 		//testVideo.read(videoImg);//读取摄像头的照片并存放在Mat类型的videoImg中
// 		m_image_queue.callAvailable(ros::WallDuration(0.4));
// 		videoImg = cv_ptr->image;
// 		if(!videoImg.data) return;
//         std::vector<Rect> xpiders;
//         Mat frame_gray;
//         cvtColor(videoImg, frame_gray, COLOR_BGR2GRAY);
//         equalizeHist(frame_gray, frame_gray);
//         hogclassifier.detectMultiScale(frame_gray, xpiders,1.25, 9, 0, Size(15, 15));


// 		Points2D.clear();
//         for(auto&rect :xpiders)
// 		{
// 			Mat faceROI = frame_gray(rect);
// 			//首先对每一个大框直接找角点
// 			// vector<Point2f> corners_hog;
// 			// calculcorners(faceROI,corners_hog,10);
// 			// if(corners_hog.size() == 0) continue;//没角点直接pass
// 			rectangle(videoImg,rect,(255,255,255),2);
			

// 			std::vector<Rect> xpiders_arr;
// 			arrowclassifier.detectMultiScale(faceROI, xpiders_arr,1.2, 3, 0, Size(15, 15));
// 			if(xpiders_arr.size() != 1)  continue;//没找到直接pass
// 			std::vector<Rect> xpiders_man;
// 			squareclassifier.detectMultiScale(faceROI, xpiders_man,1.15, 5, 0, Size(15, 15));
// 			if(xpiders_man.size() != 1)  continue;//没找到直接pass
			
// 			//用两个Rect的位置判断方向
// 			//设定箭头靠近0坐标位置，若下式成立的话就要翻转
// 			bool rev_flag = xpiders_arr[0].tl().x > xpiders_man[0].tl().x? true:false;
			
// 			std::cout << "rev_flag:" << rev_flag << std::endl;

// 			// vector<Point2f> corners_arrow;
// 			// vector<Point2f> corners_square;
// 			// square_detection(faceROI,corners_square,rect.x,rect.y,rev_flag);
// 			// arrow_detection(faceROI,corners_arrow,rect.x,rect.y,rev_flag);
// 			// //在每张图上检测人型角点
// 			Mat square = faceROI(xpiders_man[0]);
// 			vector<Point2f> corners_square;
// 			calculcorners(square,corners_square,4);
// 			std::cout << "corners_square:" << corners_square.size() << std::endl;
// 			xpiders_man[0].x += rect.x;
// 			xpiders_man[0].y += rect.y;
// 			rectangle(videoImg,xpiders_man[0],(255,255,255),6);
// 			// //用corners_hog也就是整体的角点对corners_square箭头角点进行补充
// 			// for(int i = 0;i < corners_hog.size();++i)
// 			// {
// 			// 	if(xpiders_man[0].contains(corners_hog[i]))
// 			// 	{
// 			// 		corners_square.push_back(corners_hog[i]);
// 			// 	}
// 			// }

// 			if(corners_square.size() < 4) 
// 			{
// 				ROS_WARN("square_corners detection fail......");
// 				continue;
// 			}
// 			ROS_INFO_STREAM("\033[1;32msquare_corners detection succeed!!!!!!!!!\033[0m");


// 			// //////////////////计算人型的特征角点
// 			// //先从横方向从小到大进行排序
// 			sort(corners_square.begin(),corners_square.end(),cmp());
// 			float c_x = (corners_square[0].x + corners_square.back().x) / 2;
// 			float d_x = (corners_square.back().x - corners_square[0].x) / 4;

// 			for(auto it = corners_square.begin();it != corners_square.end();++it)
// 			{
// 				if(it->x > c_x - d_x && it->x < c_x + d_x)
// 				{
// 					corners_square.erase(it);
// 				}
// 			}
// 			sort(corners_square.begin(),corners_square.end(),cmpy());
// 			if(corners_square[0].x > corners_square[1].x) swap(corners_square[0],corners_square[1]);
// 			if(corners_square.back().x < corners_square[corners_square.size()-2].x) swap(corners_square.back(), corners_square[corners_square.size()]);
// 			corners_square.erase(corners_square.begin() + 2,corners_square.end() - 2);



// 			//先从纵方向从小到大进行排序
// 			sort(corners_square.begin(),corners_square.end(),cmpy());
// 			float c_y = (corners_square[0].y + corners_square.back().y) / 2;
// 			float d_y = (corners_square.back().y - corners_square[0].y) / 4;

// 			for(auto it = corners_square.begin();it != corners_square.end();++it)
// 			{
// 				if(it->y > c_y - d_y && it->y < c_y + d_y)
// 				{
// 					corners_square.erase(it);
// 				}
// 			}
// 			std::cout << "new_corners_square:" << corners_square.size() << std::endl;
// 			std::cout << "c_x:" << c_x << " " << "c_y:" << c_y << std::endl;
// 			//以上两步是为了把中间掏空
// 			if(corners_square.size() < 4) 
// 			{
// 				ROS_WARN("point_square_corners is wrong......");
// 				continue;
// 			}

// 			vector<Point2f> as = vector<Point2f>(4);
// 			for(int i = 0;i < as.size();++i)
// 			{
// 				as[i] = Point2f(c_x,c_y);
// 			}

// 			for(int i = 0;i < corners_square.size();++i)
// 			{
// 				if(corners_square[i].x > as[0].x && corners_square[i].y < as[0].y) as[0]=corners_square[i];
// 				if(corners_square[i].x < as[1].x && corners_square[i].y < as[1].y) as[1]=corners_square[i];
// 				if(corners_square[i].x < as[2].x && corners_square[i].y > as[2].y) as[2]=corners_square[i];
// 				if(corners_square[i].x > as[3].x && corners_square[i].y > as[3].y) as[3]=corners_square[i];
// 			}
// 			if(rev_flag)
// 			{
// 				swap(as[0],as[1]);
// 				swap(as[2],as[3]);
// 			}

// 			// 绘制角点
// 			for (size_t i = 0; i < corners_square.size(); i++)
// 			{
// 				corners_square[i].x += xpiders_man[0].x;
// 				corners_square[i].y += xpiders_man[0].y;
// 				Point center(corners_square[i].x, corners_square[i].y);
// 				circle(videoImg, center, 2, Scalar(0, 0, 0), 2, LINE_AA, 0);
// 			}



// 			//在每张图上检测箭头角点
// 			Mat arrow = faceROI(xpiders_arr[0]);
// 			vector<Point2f> corners_arrow;
// 			calculcorners(arrow,corners_arrow,7);

// 			xpiders_arr[0].x += rect.x;
// 			xpiders_arr[0].y += rect.y;
// 			rectangle(videoImg,xpiders_arr[0],(255,255,255),6);
// 			//用corners_hog也就是整体的角点对corners_arrow箭头角点进行补充
// 			// for(int i = 0;i < corners_hog.size();++i)
// 			// {
// 			// 	if(xpiders_arr[0].contains(corners_hog[i]))
// 			// 	{
// 			// 		corners_arrow.push_back(corners_hog[i]);
// 			// 	}
// 			// }
// 			if(corners_arrow.size() < 3) 
// 			{
// 				ROS_WARN("arrow_corners detection fail......");
// 				continue;
// 			}
// 			ROS_INFO_STREAM("\033[1;32marrow_corners detection succeed!!!!!!!!!\033[0m");


// 			//////////////////计算箭头的特征角点，用aj1,aj2,aj3保存
// 			//先从横方向从小到大进行排序
// 			sort(corners_arrow.begin(),corners_arrow.end(),cmp());
// 			Point2f aj1,aj2,aj3;
// 			if(rev_flag) 
// 			{
// 				aj3 = corners_arrow.back();
// 				int k = 0;
// 				if(corners_arrow[0].y < aj3.y) 
// 				{
// 					aj1 = corners_arrow[0];
// 					while(++k<corners_arrow.size())
// 					{
// 						if(corners_arrow[k].y > aj3.y) {aj2 = corners_arrow[k];break;}
// 					}
// 					if(k == corners_arrow.size())
// 					{
// 						ROS_WARN("points_arrow_corners is wrong......");
// 						continue;
// 					}
// 				}
// 				else
// 				{
// 					aj2 = corners_arrow[0];
// 					while(++k<corners_arrow.size())
// 					{
// 						if(corners_arrow[k].y < aj3.y) {aj1 = corners_arrow[k];break;}
// 					}
// 					if(k == corners_arrow.size())
// 					{
// 						ROS_WARN("points_arrow_corners is wrong......");
// 						continue;
// 					}
// 				}
// 			}
// 			else 
// 			{
// 				aj3 = corners_arrow[0];
// 				int k = corners_arrow.size();
// 				if(corners_arrow.back().y < aj3.y) 
// 				{
// 					aj1 = corners_arrow.back();
// 					while(--k>=0)
// 					{
// 						if(corners_arrow.back().y > aj3.y) {aj2 = corners_arrow.back();break;}
// 					}
// 					if(k < 0)
// 					{
// 						ROS_WARN("points_arrow_corners is wrong......");
// 						continue;
// 					}
// 				}
// 				else
// 				{
// 					aj2 = corners_arrow.back();
// 					while(--k>=0)
// 					{
// 						if(corners_arrow.back().y < aj3.y) {aj1 = corners_arrow.back();break;}
// 					}
// 					if(k < 0)
// 					{
// 						ROS_WARN("points_arrow_corners is wrong......");
// 						continue;
// 					}
// 				}
// 			}
			

// 			corners_arrow.resize(3);
// 			corners_arrow[0] = aj1;corners_arrow[1]=aj2;corners_arrow[2]=aj3;
// 			//绘制角点
// 			for (size_t i = 0; i < corners_arrow.size(); i++)
// 			{
// 				corners_arrow[i].x += xpiders_arr[0].x;
// 				corners_arrow[i].y += xpiders_arr[0].y;
// 				Point center(corners_arrow[i].x, corners_arrow[i].y);
// 				circle(videoImg, center, 2, Scalar(0, 0, 0), 2, LINE_AA, 0);
// 			}


			


// 			vector<Point2f> corners;
// 			corners.insert(corners.end(),corners_square.begin(),corners_square.end());
// 			corners.push_back(aj1);
// 			corners.push_back(aj2);
// 			corners.push_back(aj3);
// 			//calculcorners(faceROI,corners,7);
// 			if(corners.size() != 7) 
// 			{
// 				ROS_WARN("sign detection fail......");
// 				continue;
// 			}
// 			Points2D.push_back(corners);
// 		}
// 		// Mat edge,grayImage;
// 		// cvtColor(videoImg, grayImage, COLOR_BGR2GRAY);
// 		// threshold(grayImage, grayImage, 50, 255, THRESH_BINARY);
// 		// blur(grayImage, edge, Size(3, 3));
// 		// Canny(edge, edge, 128, 200, 3);
// 		// imshow("Canny", edge);
// 		imshow(window_name, videoImg);//将图片显示出来
// 		waitKey(20);//相邻每帧图片间隔时间（单位：ms)
// 	//}
// }

void CameraPose::arrow_detection(const Mat& faceROI,vector<Point2f>& corners,int fx,int fy,bool& rev_flag)
{
	std::vector<Rect> xpiders;
	arrowclassifier.detectMultiScale(faceROI, xpiders,1.2, 9, 0, Size(15, 15));

	if(xpiders.size() != 1) 
	{
		ROS_WARN("arrow detection fail......");
		return;
	}
	//在每张图上检测角点
	Mat arrow = faceROI(xpiders[0]);
	calculcorners(arrow,corners,7);

	xpiders[0].x += fx;
	xpiders[0].y += fy;
	rectangle(videoImg,xpiders[0],(255,255,255),6);
	if(corners.size() < 3) 
	{
		ROS_WARN("arrow_corners detection fail......");
		return;
	}
	ROS_INFO_STREAM("\033[1;32marrow_corners detection succeed!!!!!!!!!\033[0m");
	sort(corners.begin(),corners.end(),cmp());
	if(abs(corners[0].y - corners[1].y) < 10){
		corners.erase(corners.begin() + 2,corners.end() - 1);
		if(corners[0].y < corners[1].y) swap(corners[0],corners[1]);
		rev_flag = false;
	}
	else{
		corners.erase(corners.begin() + 1,corners.end() - 2);
		if(corners[1].y < corners[2].y) swap(corners[1],corners[2]);
		reverse(corners.begin(),corners.end());
		rev_flag = true;
	}
	for (size_t i = 0; i < corners.size(); i++)
	{
		corners[i].x += xpiders[0].x;
		corners[i].y += xpiders[0].y;
		Point center(corners[i].x, corners[i].y);
		circle(videoImg, center, 2, Scalar(0, 0, 0), 2, LINE_AA, 0);
	}
}

void CameraPose::square_detection(const Mat& faceROI,vector<Point2f>& corners,int fx,int fy,bool rev_flag)
{
	std::vector<Rect> xpiders;
	squareclassifier.detectMultiScale(faceROI, xpiders,1.2, 9, 0, Size(15, 15));

	if(xpiders.size() != 1) 
	{
		ROS_WARN("square detection fail......");
		return;
	}
	//在每张图上检测角点
	Mat square = faceROI(xpiders[0]);
	calculcorners(square,corners,15);
	xpiders[0].x += fx;
	xpiders[0].y += fy;
	rectangle(videoImg,xpiders[0],(255,255,255),4);
	if(corners.size() < 4) 
	{
		ROS_WARN("square_corners detection fail......");
		return;
	}
	ROS_INFO_STREAM("\033[1;32msquare_corners detection succeed!\033[0m");
	sort(corners.begin(),corners.end(),cmpy());
	if(corners[0].x > corners[1].x) swap(corners[0],corners[1]);
	if(corners[2].x < corners[3].x) swap(corners[2],corners[3]);
	corners.erase(corners.begin() + 2,corners.end() - 2);
	//vector<int> flagxy(4,0);//按顺序找x最小、最大，y最小、最大
	// for (size_t i = 1; i < corners.size(); i++)
	// {
	// 	if(corners[i].x < corners[flagxy[0]].x) flagxy[0] = i;
	// 	if(corners[i].x > corners[flagxy[1]].x) flagxy[1] = i;
	// 	if(corners[i].y < corners[flagxy[2]].y) flagxy[2] = i;
	// 	if(corners[i].y > corners[flagxy[3]].y) flagxy[3] = i;
	// }
	// if(corners[flagxy[0]].y - corners[flagxy[2]].y > corners[flagxy[3]].y - corners[flagxy[0]].y)
	// {
	// 	corners.push_back(corners[flagxy[0]]);
	// 	corners.push_back(corners[flagxy[2]]);
	// 	corners.push_back(corners[flagxy[1]]);
	// 	corners.push_back(corners[flagxy[3]]);
	// 	corners.erase(corners.begin(),corners.end() - 4);
	// }
	// else
	// {
	// 	corners.push_back(corners[flagxy[2]]);
	// 	corners.push_back(corners[flagxy[1]]);
	// 	corners.push_back(corners[flagxy[3]]);
	// 	corners.push_back(corners[flagxy[0]]);
	// 	corners.erase(corners.begin(),corners.end() - 4);
	// }
	if(rev_flag)
	{
		swap(corners[0],corners[1]);
		swap(corners[2],corners[3]);
	}
	for (size_t i = 0; i < corners.size(); i++)
	{
		corners[i].x += xpiders[0].x;
		corners[i].y += xpiders[0].y;
		Point center(corners[i].x, corners[i].y);
		circle(videoImg, center, 2, Scalar(0, 0, 0), 2, LINE_AA, 0);
	}
}

void CameraPose::pub_signs_pose()
{
	//poses_of_signs初始化
	geometry_msgs::PoseArray pub_msg;
	pub_msg.header.frame_id = "map";
	pub_msg.poses.resize(poses_of_signs.size());
	for(int i = 0;i < poses_of_signs.size();++i){
		poses_of_signs[i].header.frame_id = "map";
		poses_of_signs[i].child_frame_id = "signs_" + to_string(i);
		poses_of_signs[i].header.stamp = ros::Time::now();
		poses_of_signs[i].transform.translation.x = signs["signpoints"][i]["pose"]["position"]["x"].as<double>(); 
		poses_of_signs[i].transform.translation.y = signs["signpoints"][i]["pose"]["position"]["y"].as<double>();
		poses_of_signs[i].transform.translation.z = signs["signpoints"][i]["pose"]["position"]["z"].as<double>();

		poses_of_signs[i].transform.rotation.x = signs["signpoints"][i]["pose"]["orientation"]["x"].as<double>();
		poses_of_signs[i].transform.rotation.y = signs["signpoints"][i]["pose"]["orientation"]["y"].as<double>();
		poses_of_signs[i].transform.rotation.z = signs["signpoints"][i]["pose"]["orientation"]["z"].as<double>();
		poses_of_signs[i].transform.rotation.w = signs["signpoints"][i]["pose"]["orientation"]["w"].as<double>();
		
		pub_msg.poses[i].orientation.x = poses_of_signs[i].transform.rotation.x;
		pub_msg.poses[i].orientation.y = poses_of_signs[i].transform.rotation.y;
		pub_msg.poses[i].orientation.z = poses_of_signs[i].transform.rotation.z;
		pub_msg.poses[i].orientation.w = poses_of_signs[i].transform.rotation.w;
		pub_msg.poses[i].position.x = poses_of_signs[i].transform.translation.x;
		pub_msg.poses[i].position.y = poses_of_signs[i].transform.translation.y;
		pub_msg.poses[i].position.z = poses_of_signs[i].transform.translation.z;
	}
	//signs_pub.sendTransform(poses_of_signs);
	ros::Rate rate(1);
	while(ros::ok())
	{
		pub_msg.header.stamp = ros::Time::now();
		signs_pub.sendTransform(poses_of_signs); 
        particlecloud_pub_.publish(pub_msg);
		rate.sleep();
	}
}


//获取所有牌子上的3D点,存到Points3D中
void CameraPose::set_poses_of_signs()
{
	//poses_of_signs初始化
	for(int i = 0;i < poses_of_signs.size();++i){
		try  
        {  
			auto tem = buffer.lookupTransform("map","signs_" + to_string(i),ros::Time::now());
			//下面是一个牌子上的所有点
			vector<geometry_msgs::PointStamped> temPoints;
			for(int j = 0;j < realtransPoints.size();++j){
				realtransPoints[j].header.frame_id = poses_of_signs[i].child_frame_id;
				temPoints.push_back(buffer.transform(realtransPoints[j],"map"));  
			}
			//获取所有牌子上的3D点
			cout << "获取所有牌子上的3D点,存到Points3D中" << endl;
			geometry2Point3D(temPoints,Points3D[i]);
		}
        catch(const std::exception& e)  
        {  
            std::cerr << e.what() << '\n';  
        } 
	}
}

void CameraPose::geometry2Point3D(vector<geometry_msgs::PointStamped>& geo,vector<cv::Point3f>& poi)
{
	for(int j = 0;j < geo.size();++j){
		poi.emplace_back(geo[j].point.x,geo[j].point.y,geo[j].point.z);
	}
}

void CameraPose::print_poses_of_signs()
{
	cout << "打印所有角点......" << endl; 
	for(int i = 0;i < Points3D.size();++i){		
		for(int j = 0;j < Points3D[0].size();++j){
			cout << Points3D[i][j].x << " " << Points3D[i][j].y << " " << Points3D[i][j].z << endl;
		}
		cout << endl;
	}
}

void CameraPose::print_poses_of_robot()
{
	cout << "打印所有机器人估计点......" << endl; 
	for(int i = 0;i < poses_of_robot.size();++i){
		cout << poses_of_robot[i].getOrigin().getX()<< " " << poses_of_robot[i].getOrigin().getY() << " " << poses_of_robot[i].getOrigin().getZ() << endl;
		cout << poses_of_robot[i].getRotation().getX() << " " << poses_of_robot[i].getRotation().getY() << " " << poses_of_robot[i].getRotation().getZ() << " " << poses_of_robot[i].getRotation().getW() << endl;
		cout << endl;
	}
}

void CameraPose::pub_poses_of_robot()
{
	cout << "发布所有机器人估计点......" << endl;
	geometry_msgs::PoseArray cloud_msg;
	cloud_msg.header.stamp = ros::Time::now();
	cloud_msg.header.frame_id = "map";
	cloud_msg.poses.resize(poses_of_robot.size());
	for(int i=0;i<poses_of_robot.size();i++)
	{
		cloud_msg.poses[i].position.x = poses_of_robot[i].getOrigin().getX();
		cloud_msg.poses[i].position.y = poses_of_robot[i].getOrigin().getX();
		cloud_msg.poses[i].position.z = 0;
		tf2::convert(poses_of_robot[i].getRotation(), cloud_msg.poses[i].orientation);
	}
	particlecloud_pub_.publish(cloud_msg);
}

//将空间点绕Z轴旋转
//输入参数 x y为空间点原始x y坐标
//thetaz为空间点绕Z轴旋转多少度，角度制范围在-180到180
//outx outy为旋转后的结果坐标
void CameraPose::codeRotateByZ(double x, double y, double thetaz, double& outx, double& outy)
{
	double x1 = x;//将变量拷贝一次，保证&x == &outx这种情况下也能计算正确
	double y1 = y;
	double rz = thetaz * CV_PI / 180;
	outx = cos(rz) * x1 - sin(rz) * y1;
	outy = sin(rz) * x1 + cos(rz) * y1;
}

//将空间点绕Y轴旋转
//输入参数 x z为空间点原始x z坐标
//thetay为空间点绕Y轴旋转多少度，角度制范围在-180到180
//outx outz为旋转后的结果坐标
void CameraPose::codeRotateByY(double x, double z, double thetay, double& outx, double& outz)
{
	double x1 = x;
	double z1 = z;
	double ry = thetay * CV_PI / 180;
	outx = cos(ry) * x1 + sin(ry) * z1;
	outz = cos(ry) * z1 - sin(ry) * x1;
}

//将空间点绕X轴旋转
//输入参数 y z为空间点原始y z坐标
//thetax为空间点绕X轴旋转多少度，角度制，范围在-180到180
//outy outz为旋转后的结果坐标
void CameraPose::codeRotateByX(double y, double z, double thetax, double& outy, double& outz)
{
	double y1 = y;//将变量拷贝一次，保证&y == &y这种情况下也能计算正确
	double z1 = z;
	double rx = thetax * CV_PI / 180;
	outy = cos(rx) * y1 - sin(rx) * z1;
	outz = cos(rx) * z1 + sin(rx) * y1;
}


//点绕任意向量旋转，右手系
//输入参数old_x，old_y，old_z为旋转前空间点的坐标
//vx，vy，vz为旋转轴向量
//theta为旋转角度角度制，范围在-180到180
//返回值为旋转后坐标点
cv::Point3f RotateByVector(double old_x, double old_y, double old_z, double vx, double vy, double vz, double theta)
{
	double r = theta * CV_PI / 180;
	double c = cos(r);
	double s = sin(r);
	double new_x = (vx*vx*(1 - c) + c) * old_x + (vx*vy*(1 - c) - vz*s) * old_y + (vx*vz*(1 - c) + vy*s) * old_z;
	double new_y = (vy*vx*(1 - c) + vz*s) * old_x + (vy*vy*(1 - c) + c) * old_y + (vy*vz*(1 - c) - vx*s) * old_z;
	double new_z = (vx*vz*(1 - c) - vy*s) * old_x + (vy*vz*(1 - c) + vx*s) * old_y + (vz*vz*(1 - c) + c) * old_z;
	return cv::Point3f(new_x, new_y, new_z);
}
