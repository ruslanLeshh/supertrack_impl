import pybullet as p
import time
import csv 
import json
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import numpy as np


#                                   BULLET SETUP



def get_state():
    #   STATE
    # - world space positions  pos [0, 0, 0] of each rigid body
    # - world space velocities vel [0, 0, 0] of each rigid body
    # - world space rotations rot [0, 0, 0, 0] of each rigid body
    # - world space rotation velocities rotVel [0, 0, 0] of each rigid body

    # For storing data of all rigid bodies
    all_pos = []       # World space positions
    all_vel = []       # World space linear velocities
    all_rot = []       # World space orientations (quaternions)
    all_rot_vel = []   # World space angular velocities
    
    # # getBasePositionAndOrientation returns the position list of 3 floats and orientation as list of 4 floats in [x,y,z,w] order
    # base_pos, base_ori = self.p0.getBasePositionAndOrientation(self.humanoid, self.client)
    # # getBaseVelocity returns a list of two vector3 values (3 floats in a list) representing the linear velocity [x,y,z] and angular velocity [wx,wy,wz]
    # # in Cartesian worldspace coordinates.
    # base_lin_vel, base_ang_vel = self.p0.getBaseVelocity(self.humanoid, self.client)
    
    # # Append base information
    # all_pos.append(base_pos)
    # all_vel.append(base_lin_vel)
    # all_rot.append(base_ori)
    # all_rot_vel.append(base_ang_vel)
    
    # Number of links (rigid bodies) associated with the object
    num_joints = p.getNumJoints(humanoid)
    
    # Loop through each link to get its world space state
    for i in range(num_joints):
        # Get position, orientation (quaternion)
        link_state = p.getLinkState(humanoid, i, computeLinkVelocity=True)
        link_pos = link_state[0]        # vec3, list of 3 floats | world position of the URDF link mass origin
        link_ori = link_state[1]        # vec4, list of 4 floats | world orientation of the URDF link mass origin
        link_lin_vel = link_state[6]    # vec3, list of 3 floats | Cartesian world velocity
        link_ang_vel = link_state[7]    # vec3, list of 3 floats | Cartesian world rot velocity
        
        # Append link information to lists
        all_pos.append(link_pos)
        all_vel.append(link_lin_vel)
        all_rot.append(link_ori)
        all_rot_vel.append(link_ang_vel)
    
    # Format the final observation structure
    state = [
        all_pos,                 # List of [x, y, z] for each rigid body
        all_vel,             # List of [vx, vy, vz] for each rigid body
        all_rot,                 # List of [x, y, z, w] quaternions for each rigid body
        all_rot_vel          # List of [wx, wy, wz] for each rigid body
    ]
    
    return state



useGUI = True

if useGUI:
  p.connect(p.GUI)
else:
 p.connect(p.DIRECT)

plane_id = p.loadURDF('Simple-Driving/simple_driving/resources/simpleplane.urdf', basePosition=[0, 0, 0],baseOrientation= p.getQuaternionFromEuler([1.57,0,0]), useFixedBase=False)

# from pdControllerExplicit import PDControllerExplicitMultiDof
# from pdControllerStable import PDControllerStableMultiDof

# explicitPD = PDControllerExplicitMultiDof(p) # PD for kinematic 
# stablePD = PDControllerStableMultiDof(p) # PD for simulated

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.resetDebugVisualizerCamera(cameraDistance=3,
                             cameraYaw=180,
                             cameraPitch=-20,
                             cameraTargetPosition=[0, 0, 0])

p.setPhysicsEngineParameter(numSolverIterations=20) # number of solver iterations

# timeStep = 1/60 #240 fps

# p.setPhysicsEngineParameter(fixedTimeStep=timeStep) # simulation update
# p.setRealTimeSimulation(1)

t = 0.0
dt = 1./60. 

currentTime = time.time()
accumulator = 0.0

p.setPhysicsEngineParameter(numSolverIterations=50, fixedTimeStep=dt) 

#                                   LOAD MOCAP DATA




# with open('humanoid3d_long_walk2_mirror.txt', 'r') as f:
#   motion_dict = json.load(f)
# numFrames = len(motion_dict['Frames'])
# print("#frames = ", numFrames)

from bvh import Bvh
with open('walk1_subject5.bvh') as f:
    mocap = Bvh(f.read())
# mocap_data = mocap.frames
mocap_data = [[float(value) for value in frame] for frame in mocap.frames]
# mocap_data= np.radians(mocap_data)

numFrames = mocap.nframes
# GUI param
# frameId = p.addUserDebugParameter("frame", 0, numFrames - 1, 0)
# erpId = p.addUserDebugParameter("erp", 0, 1, 0.2)
# kpMotorId = p.addUserDebugParameter("kpMotor", 0, 1, .2)
# forceMotorId = p.addUserDebugParameter("forceMotor", 0, 2000, 1000)

#                                   BASICS SET UP

# jointTypes = ["JOINT_REVOLUTE", "JOINT_PRISMATIC", "JOINT_SPHERICAL", "JOINT_PLANAR", "JOINT_FIXED"]

startLocations = [[0, 0, 0]]

flags=p.URDF_MAINTAIN_LINK_ORDER+p.URDF_USE_SELF_COLLISION 
humanoid = p.loadURDF("11s3_I.urdf",
                      startLocations[0],
                      globalScaling=0.01,
                      useFixedBase=False,
                      flags=flags)

# startPose = [
#     2, 0.847532, 0, 0.9986781045, 0.01410400148, -0.0006980000731, -0.04942300517, 0.9988133229,
#     0.009485003066, -0.04756001538, -0.004475001447, 1, 0, 0, 0, 0.9649395871, 0.02436898957,
#     -0.05755497537, 0.2549218909, -0.249116, 0.9993661511, 0.009952001505, 0.03265400494,
#     0.01009800153, 0.9854981188, -0.06440700776, 0.09324301124, -0.1262970152, 0.170571,
#     0.9927545808, -0.02090099117, 0.08882396249, -0.07817796699, -0.391532, 0.9828788495,
#     0.1013909845, -0.05515999155, 0.143618978, 0.9659421276, 0.1884590249, -0.1422460188,
#     0.105854014, 0.581348
# ]

# print("11111111111",len(startPose))
# print("22222222222",len(motion_dict['Frames'][0]))

# startVel = [
#     1.235314324, -0.008525509087, 0.1515293946, -1.161516553, 0.1866449799, -0.1050802848, 0,
#     0.935706195, 0.08277326387, 0.3002461862, 0, 0, 0, 0, 0, 1.114409628, 0.3618553952,
#     -0.4505575061, 0, -1.725374735, -0.5052852598, -0.8555179722, -0.2221173515, 0, -0.1837617357,
#     0.00171895706, 0.03912837591, 0, 0.147945294, 1.837653345, 0.1534535548, 1.491385941, 0,
#     -4.632454387, -0.9111172777, -1.300648184, -1.345694622, 0, -1.084238535, 0.1313680236,
#     -0.7236998534, 0, -0.5278312973
# ]

# p.resetBasePositionAndOrientation(humanoid, [0, 0, 0], p.getQuaternionFromEuler([1.57, 0, 0]))
                                        
# y2zOrn = p.getQuaternionFromEuler([1.57, 0, 0]) # rotation about the X-axis by 90 True
# basePos, baseOrn = p.multiplyTransforms(y2zPos, y2zOrn, basePos1, baseOrn1)
# p.resetBasePositionAndOrientation(humanoid, basePos, baseOrn)
                      

#                                   SET startPose / startVel


# # index0 = 7  # Starting index for the pose/velocity data
# joint_names = mocap.get_joints_names()
# for index, j in enumerate(joint_names[1:], start=1):
#     # ji = p.getJointInfo(humanoid, j)
#     # jointType = ji[2]  # Get the joint type
#     # Handle Spherical (multi-DOF) joints
#     # if jointType == p.JOINT_SPHERICAL:
#     # Extract quaternion for rotation (4 elements: x, y, z, w)
#     startRot = [mocap.joint_offset(j)[2],mocap.joint_offset(j)[1],mocap.joint_offset(j)[0]]
#     startRot = p.getQuaternionFromEuler(startRot)
#     print("SR   ",startRot, "\nJ", j)
#     # targetVel = [startVel[index0], startVel[index0 + 1], startVel[index0 + 2]]
#     # index0 += 4  # Update the index for the next joint
#     # print("Spherical velocity:", targetVel)
#     # Set joint state (position and velocity)

#     p.resetJointStateMultiDof(humanoid, index, targetValue=startRot)



#                                   SET ROTATION TO NONE



# Resetting the robot to a neutral position and freezing movement. Resets joint behaviour
for j in range(p.getNumJoints(humanoid)):
    ji = p.getJointInfo(humanoid, j)
    jointType = ji[2]  # Get the joint type

    # Default position for spherical joints (no rotation, identity quaternion)
    if jointType == p.JOINT_SPHERICAL:
        targetPosition = [0, 0, 0, 1]  # Default identity quaternion (no rotation)
        # Control the motor for the spherical joint
        p.setJointMotorControlMultiDof(humanoid, j, p.POSITION_CONTROL, targetPosition, targetVelocity=[0, 0, 0], positionGain=0, velocityGain=1, force=[0, 0, 0])

#ROOT = 0
spine = 1
spine1 = 2
spine2 = 3
neck = 4
head = 5

right_hip = 6
right_knee = 7
right_ankle = 8
right_toe = 9
right_shoulder = 10
right_arm = 11
right_elbow = 12 #hand
right_wrist = 13
                        
left_hip = 14
left_knee = 15
left_ankle = 16
left_toe = 17
left_shoulder = 18
left_arm = 19
left_elbow = 20
left_wrist = 21
joint_indices = {
    "left_hip": 14,
    "left_knee": 15,
    "left_ankle": 16,
    "left_toe": 17,

    "right_hip": 6,
    "right_knee": 7,
    "right_ankle": 8,
    "right_toe": 9,

    "spine": 1,
    "spine1": 2,
    "spine2": 3,
    "neck": 4,
    "head": 5,

    "left_shoulder": 18,
    "left_arm": 19,
    "left_elbow": 20,
    "left_wrist": 21,

    "right_shoulder": 10,
    "right_arm": 11,
    "right_elbow": 12,
    "right_wrist": 13
}


indices = [
    (6, 9),   # left hip
    (9, 12),  # left knee
    (12, 15), # left ankle
    (15, 18), # left toe
    (18, 21), # right hip
    (21, 24), # right knee
    (24, 27), # right ankle
    (27, 30), # right toe
    (30, 33), # spine
    (33, 36), # spine1
    (36, 39), # spine2
    (39, 42), # neck
    (42, 45), # head
    (45, 48), # left shoulder
    (48, 51), # left arm
    (51, 54), # left elbow
    (54, 57), # left wrist
    (57, 60), # right shoulder
    (60, 63), # right arm
    (63, 66), # right elbow
    (66, 69)  # right wrist
]

# kpOrg = [
#     0, 0, 0, 0, 0, 0, 0, 1000, 1000, 1000, 1000, 100, 100, 100, 100, 500, 500, 500, 500, 500, 400,
#     400, 400, 400, 400, 400, 400, 400, 300, 500, 500, 500, 500, 500, 400, 400, 400, 400, 400, 400,
#     400, 400, 300
# ]  # Proportional gains for joint control

# kdOrg = [
#     0, 0, 0, 0, 0, 0, 0, 100, 100, 100, 100, 10, 10, 10, 10, 50, 50, 50, 50, 50, 40, 40, 40, 40,
#     40, 40, 40, 40, 30, 50, 50, 50, 50, 50, 40, 40, 40, 40, 40, 40, 40, 40, 30
# ]  # Derivative gains for joint control
i=0
p.setTimeOut(1000)
# p.setGravity(0, -10, 0)

while (p.isConnected()):
    # print('----start----')
    # erp = 0.2 # (Error Reduction Parameter): Controls how quickly the constraint solver corrects errors in the joint positions.
    # kpMotor = 0.2 # (Proportional Gain): Controls the stiffness of the joint motors during PD control.
    # maxForce = 1000 # Limits the force applied by the motor to move the joints.
    frameReal = i #Represents the current frame in the animation sequence, possibly with fractional values to enable interpolation.

    # kp = kpMotor

    # computes which two frames to interpolate between based on frameReal and frameFraction
    frame = int(frameReal)
    frameNext = frame + 1
    if (frameNext >= numFrames):
        frameNext = frame  # Avoid overflow of frame indices
    
    frameFraction = frameReal - frame  # Interpolated frame 

    frameData = mocap_data[frame]
    frameDataNext = mocap_data[frameNext]



#                            interpolation between two frames


# Change Base Position and Joint Orientations


    # Base Position linear interpolation between two frames
    basePos1Start = [frameData[0]*0.01, frameData[1]*0.01, frameData[2]*0.01]
    basePos1End = [frameDataNext[0]*0.01, frameDataNext[1]*0.01, frameDataNext[2]*0.01]
    basePos1 = [
        basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
        basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
        basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
    ]

    # Orientation spherical linear interpolation
    baseOrn1Start = [frameData[5], frameData[4], frameData[3]]  # y x z
    baseOrn1Start = (R.from_euler('xyz',baseOrn1Start, degrees=True)).as_quat()
    baseOrn1Next = [frameDataNext[5], frameDataNext[4], frameDataNext[3]]
    baseOrn1Next = (R.from_euler('xyz',baseOrn1Next, degrees=True)).as_quat()
    baseOrn1 = p.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)

    # pre-rotates the base orientation by applying a transform
    # y2zPos = [0, 0, 0.0] 
    # y2zOrn = p.getQuaternionFromEuler([0,0,0]) # rotation about the X-axis by 90 True
    # basePos, baseOrn = p.multiplyTransforms(y2zPos, y2zOrn, basePos1, baseOrn1)
    p.resetBasePositionAndOrientation(humanoid, basePos1, baseOrn1)

    # For each joint the code performs quaternion SLERP to smoothly interpolate
    # between joint orientations from the current frame to the next frame.
    
    # leftHipRotStart = [frameData[6], frameData[7], frameData[8]]
    # leftHipRotStart = (R.from_euler('ZYX', leftHipRotStart, degrees=True)).as_quat()
    # leftHipRotEnd = [frameDataNext[6], frameDataNext[7], frameDataNext[8]]
    # leftHipRotEnd = (R.from_euler('ZYX', leftHipRotEnd, degrees=True)).as_quat()
    # leftHipRot = p.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd, frameFraction)

    # leftKneeRotStart = [frameData[9], frameData[10], frameData[11]]
    # leftKneeRotStart = (R.from_euler('ZYX', leftKneeRotStart, degrees=True)).as_quat()
    # leftKneeRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11]]
    # leftKneeRotEnd = (R.from_euler('ZYX', leftKneeRotEnd, degrees=True)).as_quat()
    # leftKneeRot = p.getQuaternionSlerp(leftKneeRotStart, leftKneeRotEnd, frameFraction)

    # leftAnkleRotStart = [frameData[12], frameData[13], frameData[14]]
    # leftAnkleRotStart = (R.from_euler('ZYX', leftAnkleRotStart, degrees=True)).as_quat()
    # leftAnkleRotEnd = [frameDataNext[12], frameDataNext[13], frameDataNext[14]]
    # leftAnkleRotEnd = (R.from_euler('ZYX', leftAnkleRotEnd, degrees=True)).as_quat()
    # leftAnkleRot = p.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd, frameFraction)

    # leftToeRotStart = [frameData[15], frameData[16], frameData[17]]
    # leftToeRotStart = (R.from_euler('ZYX', leftToeRotStart, degrees=True)).as_quat()
    # leftToeRotEnd = [frameDataNext[15], frameDataNext[16], frameDataNext[17]]
    # leftToeRotEnd = (R.from_euler('ZYX', leftToeRotEnd, degrees=True)).as_quat()
    # leftToeRot = p.getQuaternionSlerp(leftToeRotStart, leftToeRotEnd, frameFraction)

    # rightHipRotStart = [frameData[18], frameData[19], frameData[20]]
    # rightHipRotStart = (R.from_euler('ZYX', rightHipRotStart, degrees=True)).as_quat()
    # rightHipRotEnd = [frameDataNext[18], frameDataNext[19], frameDataNext[20]]
    # rightHipRotEnd = (R.from_euler('ZYX', rightHipRotEnd, degrees=True)).as_quat()
    # rightHipRot = p.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd, frameFraction)

    # rightKneeRotStart = [frameData[21], frameData[22], frameData[23]]
    # rightKneeRotStart = (R.from_euler('ZYX', rightKneeRotStart, degrees=True)).as_quat()
    # rightKneeRotEnd = [frameDataNext[21], frameDataNext[22], frameDataNext[23]]
    # rightKneeRotEnd = (R.from_euler('ZYX', rightKneeRotEnd, degrees=True)).as_quat()
    # rightKneeRot = p.getQuaternionSlerp(rightKneeRotStart, rightKneeRotEnd, frameFraction)

    # rightAnkleRotStart = [frameData[24], frameData[25], frameData[26]]
    # rightAnkleRotStart = (R.from_euler('ZYX', rightAnkleRotStart, degrees=True)).as_quat()
    # rightAnkleRotEnd = [frameDataNext[24], frameDataNext[25], frameDataNext[26]]
    # rightAnkleRotEnd = (R.from_euler('ZYX', rightAnkleRotEnd, degrees=True)).as_quat()
    # rightAnkleRot = p.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd, frameFraction)

    # rightToeRotStart = [frameData[27], frameData[28], frameData[29]]
    # rightToeRotStart = (R.from_euler('ZYX', rightToeRotStart, degrees=True)).as_quat()
    # rightToeRotEnd = [frameDataNext[27], frameDataNext[28], frameDataNext[29]]
    # rightToeRotEnd = (R.from_euler('ZYX', rightToeRotEnd, degrees=True)).as_quat()
    # rightToeRot = p.getQuaternionSlerp(rightToeRotStart, rightToeRotEnd, frameFraction)

    # spineRotStart = [frameData[30], frameData[31], frameData[32]]
    # spineRotStart = (R.from_euler('ZYX', spineRotStart, degrees=True)).as_quat()
    # spineRotEnd = [frameDataNext[30], frameDataNext[31], frameDataNext[32]]
    # spineRotEnd = (R.from_euler('ZYX', spineRotEnd, degrees=True)).as_quat()
    # spineRot = p.getQuaternionSlerp(spineRotStart, spineRotEnd, frameFraction)

    # spine1RotStart = [frameData[33], frameData[34], frameData[35]]
    # spine1RotStart = (R.from_euler('ZYX', spine1RotStart, degrees=True)).as_quat()
    # spine1RotEnd = [frameDataNext[33], frameDataNext[34], frameDataNext[35]]
    # spine1RotEnd = (R.from_euler('ZYX', spine1RotEnd, degrees=True)).as_quat()
    # spine1Rot = p.getQuaternionSlerp(spine1RotStart, spine1RotEnd, frameFraction)

    # spine2RotStart = [frameData[36], frameData[37], frameData[38]]
    # spine2RotStart = (R.from_euler('ZYX', spine2RotStart, degrees=True)).as_quat()
    # spine2RotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38]]
    # spine2RotEnd = (R.from_euler('ZYX', spine2RotEnd, degrees=True)).as_quat()
    # spine2Rot = p.getQuaternionSlerp(spine2RotStart, spine2RotEnd, frameFraction)

    # neckRotStart = [frameData[39], frameData[40], frameData[41]]
    # neckRotStart = (R.from_euler('ZYX', neckRotStart, degrees=True)).as_quat()
    # neckRotEnd = [frameDataNext[39], frameDataNext[40], frameDataNext[41]]
    # neckRotEnd = (R.from_euler('ZYX', neckRotEnd, degrees=True)).as_quat()
    # neckRot = p.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)

    # headRotStart = [frameData[42], frameData[43], frameData[44]]
    # headRotStart = (R.from_euler('ZYX', headRotStart, degrees=True)).as_quat()
    # headRotEnd = [frameDataNext[42], frameDataNext[43], frameDataNext[44]]
    # headRotEnd = (R.from_euler('ZYX', headRotEnd, degrees=True)).as_quat()
    # headRot = p.getQuaternionSlerp(headRotStart, headRotEnd, frameFraction)

    # leftShoulderRotStart = [frameData[45], frameData[46], frameData[47]]
    # leftShoulderRotStart = (R.from_euler('ZYX', leftShoulderRotStart, degrees=True)).as_quat()
    # leftShoulderRotEnd = [frameDataNext[45], frameDataNext[46], frameDataNext[47]]
    # leftShoulderRotEnd = (R.from_euler('ZYX', leftShoulderRotEnd, degrees=True)).as_quat()
    # leftShoulderRot = p.getQuaternionSlerp(leftShoulderRotStart, leftShoulderRotEnd, frameFraction)

    # leftArmRotStart = [frameData[48], frameData[49], frameData[50]]
    # leftArmRotStart = (R.from_euler('ZYX', leftArmRotStart, degrees=True)).as_quat()
    # leftArmRotEnd = [frameDataNext[48], frameDataNext[49], frameDataNext[50]]
    # leftArmRotEnd = (R.from_euler('ZYX', leftArmRotEnd, degrees=True)).as_quat()
    # leftArmRot = p.getQuaternionSlerp(leftArmRotStart, leftArmRotEnd, frameFraction)

    # leftElbowRotStart = [frameData[51], frameData[52], frameData[53]]
    # leftElbowRotStart = (R.from_euler('ZYX', leftElbowRotStart, degrees=True)).as_quat()
    # leftElbowRotEnd = [frameDataNext[51], frameDataNext[52], frameDataNext[53]]
    # leftElbowRotEnd = (R.from_euler('ZYX', leftElbowRotEnd, degrees=True)).as_quat()
    # leftElbowRot = p.getQuaternionSlerp(leftElbowRotStart, leftElbowRotEnd, frameFraction)

    # leftWristRotStart = [frameData[54], frameData[55], frameData[56]]
    # leftWristRotStart = (R.from_euler('ZYX', leftWristRotStart, degrees=True)).as_quat()
    # leftWristRotEnd = [frameDataNext[54], frameDataNext[55], frameDataNext[56]]
    # leftWristRotEnd = (R.from_euler('ZYX', leftWristRotEnd, degrees=True)).as_quat()
    # leftWristRot = p.getQuaternionSlerp(leftWristRotStart, leftWristRotEnd, frameFraction)

    # rightShoulderRotStart = [frameData[57], frameData[58], frameData[59]]
    # rightShoulderRotStart = (R.from_euler('ZYX', rightShoulderRotStart, degrees=True)).as_quat()
    # rightShoulderRotEnd = [frameDataNext[57], frameDataNext[58], frameDataNext[59]]
    # rightShoulderRotEnd = (R.from_euler('ZYX', rightShoulderRotEnd, degrees=True)).as_quat()
    # rightShoulderRot = p.getQuaternionSlerp(rightShoulderRotStart, rightShoulderRotEnd, frameFraction)

    # rightArmRotStart = [frameData[60], frameData[61], frameData[62]]
    # rightArmRotStart = (R.from_euler('ZYX', rightArmRotStart, degrees=True)).as_quat()
    # rightArmRotEnd = [frameDataNext[60], frameDataNext[61], frameDataNext[62]]
    # rightArmRotEnd = (R.from_euler('ZYX', rightArmRotEnd, degrees=True)).as_quat()
    # rightArmRot = p.getQuaternionSlerp(rightArmRotStart, rightArmRotEnd, frameFraction)

    # rightElbowRotStart = [frameData[63], frameData[64], frameData[65]]
    # rightElbowRotStart = (R.from_euler('ZYX', rightElbowRotStart, degrees=True)).as_quat()
    # rightElbowRotEnd = [frameDataNext[63], frameDataNext[64], frameDataNext[65]]
    # rightElbowRotEnd = (R.from_euler('ZYX', rightElbowRotEnd, degrees=True)).as_quat()
    # rightElbowRot = p.getQuaternionSlerp(rightElbowRotStart, rightElbowRotEnd, frameFraction)

    # rightWristRotStart = [frameData[66], frameData[67], frameData[68]]
    # rightWristRotStart = (R.from_euler('ZYX', rightWristRotStart, degrees=True)).as_quat()
    # rightWristRotEnd = [frameDataNext[66], frameDataNext[67], frameDataNext[68]]
    # rightWristRotEnd = (R.from_euler('ZYX', rightWristRotEnd, degrees=True)).as_quat()
    # rightWristRot = p.getQuaternionSlerp(rightWristRotStart, rightWristRotEnd, frameFraction)

        


    # Convert Euler angles to quaternions for both start and end frames
    angles_start = np.array([frameData[start:end] for start, end in indices])
    angles_end = np.array([frameDataNext[start:end] for start, end in indices])

    quats_start = R.from_euler('ZYX', angles_start, degrees=True).as_quat()
    quats_end = R.from_euler('ZYX', angles_end, degrees=True).as_quat()

    # Perform SLERP interpolation
    quats_interpolated = [p.getQuaternionSlerp(q_start, q_end, frameFraction) for q_start, q_end in zip(quats_start, quats_end)]

    # print("\n\nQQQQQQQQQQQQ", quats_end)



    #                              COMPUTE PD torques required to move each joint

    # for i, (joint, index) in enumerate(joint_indices.items()):
    #     p.resetJointStateMultiDof(humanoid, index, quats_interpolated[i])

    # time.sleep(1000)

    #                             Configures joint motor control

    # # print("ORN         ", baseOrn , "\n POS", basePos)
    p.resetJointStateMultiDof(humanoid, left_hip, quats_interpolated[0])
    p.resetJointStateMultiDof(humanoid, left_knee, quats_interpolated[1])
    p.resetJointStateMultiDof(humanoid, left_ankle, quats_interpolated[2])
    p.resetJointStateMultiDof(humanoid, left_toe, quats_interpolated[3])

    p.resetJointStateMultiDof(humanoid, right_hip, quats_interpolated[4]) #RightUpLeg
    p.resetJointStateMultiDof(humanoid, right_knee, quats_interpolated[5]) #RightLeg
    p.resetJointStateMultiDof(humanoid, right_ankle, quats_interpolated[6]) #RightFoot
    p.resetJointStateMultiDof(humanoid, right_toe, quats_interpolated[7]) #RightToe

    p.resetJointStateMultiDof(humanoid, spine, quats_interpolated[8])
    p.resetJointStateMultiDof(humanoid, spine1, quats_interpolated[9])
    p.resetJointStateMultiDof(humanoid, spine2, quats_interpolated[10])
    p.resetJointStateMultiDof(humanoid, neck, quats_interpolated[11])
    p.resetJointStateMultiDof(humanoid, head, quats_interpolated[12])
    
    p.resetJointStateMultiDof(humanoid, left_shoulder, quats_interpolated[13])
    p.resetJointStateMultiDof(humanoid, left_arm, quats_interpolated[14])
    p.resetJointStateMultiDof(humanoid, left_elbow, quats_interpolated[15])
    p.resetJointStateMultiDof(humanoid, left_wrist, quats_interpolated[16])

    p.resetJointStateMultiDof(humanoid, right_shoulder, quats_interpolated[17])
    p.resetJointStateMultiDof(humanoid, right_arm, quats_interpolated[18])
    p.resetJointStateMultiDof(humanoid, right_elbow, quats_interpolated[19])
    p.resetJointStateMultiDof(humanoid, right_wrist, quats_interpolated[20])
    
    
      
    newTime = time.time()
    frameTime = newTime - currentTime
    currentTime = newTime
    accumulator += frameTime
    # print(accumulator)
    while ( accumulator >= dt ):
        p.stepSimulation() # ( state, t, dt )
        accumulator -= dt
        t += dt
        
        # print(t)
        i+=dt/(1/30)
        if i >= numFrames:
            p.setTimeOut(1000)
            i=0
        # print(t)
    a = get_state()
    # print('\nki_pos',a[0])
    # print('\nki_vel',a[1])
    # print('\nki_rot',a[2])
    # print('\nki_ang',a[3])
    # linVelSim, angVelSim = p.getBaseVelocity(humanoid)
    # print('#',linVelSim)
    # print('@',angVelSim)
    # no accum
    
    # p.stepSimulation()
    # # time.sleep(frameTime)
    # i+=dt/(1/30)
    # if i >= numFrames:
    #     p.setTimeOut(1000)
    #     i=0


# p.disconnect()