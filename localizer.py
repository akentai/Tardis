#!/usr/bin/env python
import rospy
import numpy.matlib
import numpy as np
import matplotlib.pyplot as plt
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

Yaw0 = 1.43
M_PI = 3.14159265358979323846

class sonar:
    F_val = 0.0
    FL_val = 0.0
    FR_val = 0.0
    L_val = 0.0
    R_val = 0.0

class imu:
    Roll = 0.0 # orientation
    Pitch = 0.0
    Yaw = 0.0
    AngVelX = 0.0 # angular velocity
    AngVelY = 0.0
    AngVelZ = 0.0
    LinAccX = 0.0 # linear acceleration
    LinAccY = 0.0
    LinAccZ = 0.0

class state:
    x = 0.0
    y = 0.0
    theta = Yaw0
    P = np.zeros( (3, 3) )

class control:
    v = 0.0
    w = 0.0

class timer:
    prev = 0.0
    phase = 0
    index = 2
    T = 0.099

class odomm:
    x = 0
    y = 0
    time = 1100
    errx = np.zeros(1)
    erry = np.zeros(1)

estimRoll = 0.0 # always zero value in 2D navigation
estimPitch = 0.0 # always zero value in 2D navigation
estimYaw = 0.0

ekf_estimation_msg = Odometry()
ekf_estimation_msg.header.frame_id = "odom"
ekf_estimation_msg.child_frame_id = "chassis"

def EKF():

    T = timer.T
    P = state.P
    (x, y, theta) = [ state.x, state.y, state.theta ]
    (v, w) = [control.v, control.w]
    ss1 = 0.01277
    ss2 = 0.0538
    ss = 0.002
    
    arg = theta
    # Predict state estimate
    (xk, yk, thetak) = [ x+v*T*np.cos(arg), y+v*T*np.sin(arg), theta-w*T ]

    # Predict covariance estimate
    F = np.array( [ [1, 0 , -v*T*np.sin(arg)], [0, 1, v*T*np.cos(arg)], [0, 0, 1] ] )
    F_Trans = F.transpose()

    V = np.array( [ [T*np.cos(arg), v*T*T/2*np.sin(arg)], [T*np.sin(arg), -v*T*T/2*np.cos(arg)], [0, -T]  ] )
    V_Trans = V.transpose()
    ControlNoise = np.array( [ [ss1, 0], [0, ss2] ] )

    matr = V.dot(ControlNoise)
    Mt = matr.dot(V_Trans)

    matr = F.dot(P)
    matr = matr.dot(F_Trans)

    Pk = matr + Mt

    # Measurements (angle & front sonar)
    thet = (imu.Yaw + Yaw0)%(-np.pi)
    print('angle =' ,thet)
    err1 = (thet - thetak%(-np.pi))

    
    if v==0 and timer.prev > 0:
        timer.phase = (timer.phase+1)%4
    timer.prev = v

    phase = timer.phase

    lf = sonar.F_val
    if v == 0:
        err2 = 0.0
    elif lf == 2:
        err2 = 0.05
    elif phase == 1:
        lf_model = (2.0-xk)/np.cos(thetak)-0.15
        err2 = lf-lf_model
    elif phase == 2:
        lf_model = -(2.0+yk)/np.sin(thetak)-0.15
        err2 = lf-lf_model
    elif phase == 3:
        lf_model = -(2.0+xk)/np.cos(thetak)-0.15
        err2 = lf-lf_model
    elif phase == 0:
        lf_model = (2.0-yk)/np.sin(thetak)-0.15
        err2 = lf-lf_model
    
    print(err1, err2)

    # Innovation Covariance
    Qt = np.array( [ [0.002, 0], [0, 0.01] ] ) 
    if phase == 1:
        H = np.array( [ [0, 0, 1], [-1/np.cos(thetak), 0, (2-xk)*np.sin(thetak)*1/np.cos(thetak)*1/np.cos(thetak)] ] )
    elif phase == 2:
        H = np.array( [ [0, 0, 1], [0, -1/np.sin(thetak) , (2+yk)*np.cos(thetak)*1/np.sin(thetak)*1/np.sin(thetak)] ] )
    elif phase == 3:
        H = np.array( [ [0, 0, 1], [-1/np.cos(thetak), 0, -(2+xk)*np.sin(thetak)*1/np.cos(thetak)*1/np.cos(thetak)] ] )
    elif phase == 0:
        H = np.array( [ [0, 0, 1], [0, -1/np.sin(thetak) , -(2-yk)*np.cos(thetak)*1/np.sin(thetak)*1/np.sin(thetak)] ] )
    H_Trans = H.transpose()
    
    matr = H.dot(Pk)
    matr = matr.dot(H_Trans)
    S = matr + Qt
    
    # Kalman Gain
    S_Inv = np.linalg.inv(S)
    
    matr = Pk.dot(H_Trans)
    K = matr.dot(S_Inv)

    # Update state estimate
    xf = xk + K[0][0]*err1 + K[0][1]*err2
    yf = yk + K[1][0]*err1 + K[1][1]*err2
    thetaf = thetak + K[2][0]*err1 + K[2][1]*err2
    
    print( xk, yk, thetak )
    print( xf, yf, thetaf%(np.pi) )

    (state.x, state.y, state.theta) = [ xf, yf, thetaf ]   
    # Updated Covariance Estimate
    I = np.array( [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ] )
    matr = K.dot(H)
    matr = I - matr
    state.P = matr.dot(Pk)

    control.v = np.sqrt( (x-xf)*(x-xf) + (y-yf)*(y-yf) )/T
    
def send_velocity():
    ekf_pub = rospy.Publisher('/ekf_estimation', Odometry, queue_size=1)

    ekf_estimation_msg.header.seq += 1
    ekf_estimation_msg.header.stamp = rospy.Time.now()

    if timer.index > 0:
        timer.index = timer.index - 1
        timer.prevt = rospy.get_time()
    else:
        timer.T = rospy.get_time() - timer.prevt
        timer.prevt = rospy.get_time()

    """
    PUT YOUR MAIN CODE HERE
    """
    EKF()
    
    if odomm.time > 0:
        odomm.errx = np.append( odomm.errx, state.x - odomm.x )
        odomm.erry = np.append( odomm.erry, state.y - odomm.y )
        odomm.time = odomm.time - 1
        print(odomm.time)
    else:
        print( 'Mean and Std in x-direction', np.mean(odomm.errx), np.std(odomm.errx) )
        print( 'Mean and Std in y-direction', np.mean(odomm.erry), np.std(odomm.erry) )
        plt.figure(1)
        plt.plot(odomm.errx)
        plt.plot(odomm.erry)
        plt.show()
        wait = input("press key")
    
    estimYaw = 0.0 # orientation to be estimated (-pi,pi]
    # position to be estimated
    ekf_estimation_msg.pose.pose.position.x = state.x
    ekf_estimation_msg.pose.pose.position.y = state.y
    ekf_estimation_msg.pose.pose.position.z = 0.0
    # RPY to quaternion
    quaternion = quaternion_from_euler(estimRoll, estimPitch, estimYaw)
    ekf_estimation_msg.pose.pose.orientation.x = quaternion[0]
    ekf_estimation_msg.pose.pose.orientation.y = quaternion[1]
    ekf_estimation_msg.pose.pose.orientation.z = quaternion[2]
    ekf_estimation_msg.pose.pose.orientation.w = quaternion[3]
    # velocities to be estimated
    ekf_estimation_msg.twist.twist.linear.x = 0.0 # x-linear velocity to be estimated
    ekf_estimation_msg.twist.twist.linear.y = 0.0 # y-linear velocity to be estimated
    ekf_estimation_msg.twist.twist.linear.z = 0.0 # always zero value in 2D navigation
    ekf_estimation_msg.twist.twist.angular.x = 0.0 # always zero value in 2D navigation
    ekf_estimation_msg.twist.twist.angular.y = 0.0 # always zero value in 2D navigation
    ekf_estimation_msg.twist.twist.angular.z = 0.0 # angular velocity to be estimated

    """
    OPTIONAL
    in case your extended kalman filter (EKF) is able to estimate covariances,
    fill in the following variables:
    ekf_estimation_msg.pose.covariance Matrix:6x6
    ekf_estimation_msg.twist.covariance Matrix:6x6
    http://docs.ros.org/melodic/api/nav_msgs/html/msg/Odometry.html
    """

    ekf_pub.publish(ekf_estimation_msg)

def sonarFrontCallback(msg):
    sonar.F_val = msg.range;
    rospy.loginfo("Front Scan %s", sonar.F_val)
    send_velocity()

def sonarFrontLeftCallback(msg):
    sonar.FL_val = msg.range

def sonarFrontRightCallback(msg):
    sonar.FR_val = msg.range

def sonarLeftCallback(msg):
    sonar.L_val = msg.range

def sonarRightCallback(msg):
    sonar.R_val = msg.range

def imuCallback(msg):

    # orientation:: quaternion to RPY (rool, pitch, yaw)
    orientation_q = msg.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (imu.Roll, imu.Pitch, imu.Yaw) = euler_from_quaternion (orientation_list)

    # angular velocity
    imu.AngVelX = msg.angular_velocity.x
    imu.AngVelY = msg.angular_velocity.y
    imu.AngVelZ = msg.angular_velocity.z

    # linear acceleration
    imu.LinAccX = msg.linear_acceleration.x
    imu.LinAccY = msg.linear_acceleration.y
    imu.LinAccZ = msg.linear_acceleration.z

def controlCallback(msg):
    control.v = msg.linear.x
    control.w = msg.angular.z

def odomCallback(msg):
    odomm.x = msg.pose.pose.position.x
    odomm.y = msg.pose.pose.position.y

def follower_py():
    # Starts a new node
    rospy.init_node('localizer_node', anonymous=True)
    rospy.Subscriber("sonarFront_scan", Range, sonarFrontCallback)
    rospy.Subscriber("sonarFrontLeft_scan", Range, sonarFrontLeftCallback)
    rospy.Subscriber("sonarFrontRight_scan", Range, sonarFrontRightCallback)
    rospy.Subscriber("sonarLeft_scan", Range, sonarLeftCallback)
    rospy.Subscriber("sonarRight_scan", Range, sonarRightCallback)
    rospy.Subscriber("imu_data", Imu, imuCallback)
    rospy.Subscriber("cmd_vel", Twist, controlCallback)
    rospy.Subscriber("odom", Odometry, odomCallback)
    
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == '__main__':
    try:
        #Testing our function
        follower_py()
    except rospy.ROSInterruptException: pass
