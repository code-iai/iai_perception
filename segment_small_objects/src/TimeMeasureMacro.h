/*
 * TimeMeasureMacro.h
 *
 *  Created on: Feb 14, 2013
 *      Author: jalapeno
 */

#ifndef TIMEMEASUREMACRO_H_
#define TIMEMEASUREMACRO_H_

#ifdef DEBUG

#define DEBUG_TIME(name, code)\
{\
	ros::Time time = ros::Time::now();\
	ROS_ERROR(name "begin");\
	code\
	ROS_ERROR(name ": %f ms", 0.000001 * (ros::Time::now() - time).nsec);\
}

#endif

#ifndef DEBUG

#define DEBUG_TIME(name, code) code

#endif

#endif /* TIMEMEASUREMACRO_H_ */
