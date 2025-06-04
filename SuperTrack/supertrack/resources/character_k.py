# import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import math

       
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

# joint_indices = {
#     "left_hip": 14,
#     "left_knee": 15,
#     "left_ankle": 16,
#     "left_toe": 17,

#     "right_hip": 6,
#     "right_knee": 7,
#     "right_ankle": 8,
#     "right_toe": 9,

#     "spine": 1,
#     "spine1": 2,
#     "spine2": 3,
#     "neck": 4,
#     "head": 5,

#     "left_shoulder": 18,
#     "left_arm": 19,
#     "left_elbow": 20,
#     "left_wrist": 21,

#     "right_shoulder": 10,
#     "right_arm": 11,
#     "right_elbow": 12,
#     "right_wrist": 13
# }


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


class Character_k:
    def __init__(self, client, mocap_data, rand_frame):
        self.p0 = client
        self.mocap_data = mocap_data
        self.numFrames = len(mocap_data)
        self.k_targets = None
        # self.k_vel_targets = None

        if self.p0.getPhysicsEngineParameters() is None:
            print("Errorrrrrrrr: Not connected to the physics server.")
        else:
            print("Physicssssss server connected in Character_k.")
        
        # frameData = self.mocap_data['Frames'][0]
        # basePos = [frameData[1], frameData[2], frameData[3]]  # 0th frame position (x, y, z)
        # baseOrn = [frameData[5], frameData[6], frameData[7], frameData[4]]  # 0th frame orientation (quaternion)
        # y2zPos = [0, 0, 0.0]
        # y2zOrn = self.p0.getQuaternionFromEuler([1.57, 0, 0])  # Rotate 90 degrees around X-axis
        # basePos, baseOrn = self.p0.multiplyTransforms(y2zPos, y2zOrn, basePos, baseOrn)

        # chestRot = [frameData[9], frameData[10], frameData[11], frameData[8]]
        # neckRot = [frameData[13], frameData[14], frameData[15], frameData[12]]
        # rightHipRot = [frameData[17], frameData[18], frameData[19], frameData[16]]
        # rightKneeRot = [frameData[20]]
        # rightAnkleRot = [frameData[22], frameData[23], frameData[24], frameData[21]]
        # rightShoulderRot = [frameData[26], frameData[27], frameData[28], frameData[25]]
        # rightElbowRot = [frameData[29]]
        # leftHipRot = [frameData[31], frameData[32], frameData[33], frameData[30]]
        # leftKneeRot = [frameData[34]]
        # leftAnkleRot = [frameData[36], frameData[37], frameData[38], frameData[35]]
        # leftShoulderRot = [frameData[40], frameData[41], frameData[42], frameData[39]]
        # leftElbowRot = [frameData[43]]

        # import model
        flags=self.p0.URDF_MAINTAIN_LINK_ORDER
        self.humanoid = self.p0.loadURDF("11s3_I.urdf",
                            [0,0,0],
                            globalScaling=0.01,
                            useFixedBase=False,
                            flags=flags)
        
        #                                   SET ROTATION TO NONE
        # Resetting the robot to a neutral position and freezing movement. Resets joint behaviour
        for j in range(self.p0.getNumJoints(self.humanoid)):
            ji = self.p0.getJointInfo(self.humanoid, j)
            jointType = ji[2]  # Get the joint type
            # Default position for spherical joints (no rotation, identity quaternion)
            if jointType == self.p0.JOINT_SPHERICAL:
                targetPosition = [0, 0, 0, 1]  # Default identity quaternion (no rotation)
                # Control the motor for the spherical joint
                self.p0.setJointMotorControlMultiDof(self.humanoid, j, self.p0.POSITION_CONTROL, targetPosition, targetVelocity=[0, 0, 0], positionGain=0, velocityGain=1, force=[0, 0, 0])
        
        for j in range(self.p0.getNumJoints(self.humanoid)):
            self.p0.changeVisualShape(self.humanoid, j, rgbaColor=[1, 0, 0, 0.3]) # make him red
 
        self.set_frame(rand_frame) # set pose to first frame 
        self.kin_state = self.get_first_state()
        # self.k_targets = [chestRot,neckRot,rightHipRot, rightKneeRot,
        #                 rightAnkleRot, rightShoulderRot, rightElbowRot,
        #                 leftHipRot, leftKneeRot, leftAnkleRot,
        #                 leftShoulderRot, leftElbowRot]
        
        # self.k_targets = [[chestRot,
        #     neckRot,
        #     rightHipRot, 
        #     rightAnkleRot,
        #     rightShoulderRot,
        #     leftHipRot, 
        #     leftAnkleRot,
        #     leftShoulderRot]
        #     [rightKneeRot, 
        #     rightElbowRot, 
        #     leftKneeRot, 
        #     leftElbowRot]]



    # def get_ids(self):
    #     return self.car, self.client
    
    def ComputeLinVel(self, posStart, posEnd, deltaTime):
        vel = [
            (posEnd[0] - posStart[0]) / deltaTime,
            (posEnd[1] - posStart[1]) / deltaTime,
            (posEnd[2] - posStart[2]) / deltaTime
        ]
        return vel

    def ComputeAngVel(self, ornStart, ornEnd, deltaTime):
        dorn = self.p0.getDifferenceQuaternion(ornStart, ornEnd)
        axis, angle = self.p0.getAxisAngleFromQuaternion(dorn)
        angVel = [
            (axis[0] * angle) / deltaTime,
            (axis[1] * angle) / deltaTime,
            (axis[2] * angle) / deltaTime
        ]
        return angVel
    
    # def ComputeLinVel(self, posStart, posEnd, deltaTime):
    #     # Vectorized linear velocity computation
    #     return (np.array(posEnd) - np.array(posStart)) / deltaTime

    # def ComputeAngVel(self, ornStart, ornEnd, deltaTime):
    #     # Quaternion difference to axis-angle
    #     dorn = self.p0.getDifferenceQuaternion(ornStart, ornEnd)
    #     axis, angle = self.p0.getAxisAngleFromQuaternion(dorn)

    #     # Vectorized angular velocity
    #     return np.array(axis) * (angle / deltaTime)
    



    def set_frame(self, frame_i):
        frame = int(frame_i)
        frameNext = frame + 1
        if (frameNext >= self.numFrames):
            frameNext = frame  # Avoid overflow of frame indices

        frameFraction = frame_i - frame  # Interpolated frame 

        frameData = self.mocap_data[frame]
        frameDataNext = self.mocap_data[frameNext]


    #                            interpolation between two frames
        basePos1Start = [frameData[0]*0.01, frameData[1]*0.01, frameData[2]*0.01]
        basePos1End = [frameDataNext[0]*0.01, frameDataNext[1]*0.01, frameDataNext[2]*0.01]
        basePos1 = [
            basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
            basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
            basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
        ]
        # self._baseLinVel = self.ComputeLinVel(basePos1Start, basePos1End, 1./30.)

        baseOrn1Start = [frameData[5], frameData[4], frameData[3]]  # y x z
        baseOrn1Start = (R.from_euler('xyz',baseOrn1Start, degrees=True)).as_quat()
        baseOrn1Next = [frameDataNext[5], frameDataNext[4], frameDataNext[3]]
        baseOrn1Next = (R.from_euler('xyz',baseOrn1Next, degrees=True)).as_quat()
        baseOrn1 = self.p0.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
        # self._baseAngVel = self.ComputeAngVel(baseOrn1Start, baseOrn1Next, 1./30.)

        self.p0.resetBasePositionAndOrientation(self.humanoid, basePos1, baseOrn1)

        # leftHipRotStart = [frameData[6], frameData[7], frameData[8]]
        # leftHipRotStart = (R.from_euler('ZYX', leftHipRotStart, degrees=True)).as_quat()
        # leftHipRotEnd = [frameDataNext[6], frameDataNext[7], frameDataNext[8]]
        # leftHipRotEnd = (R.from_euler('ZYX', leftHipRotEnd, degrees=True)).as_quat()
        # leftHipRot = self.p0.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd, frameFraction)

        # leftKneeRotStart = [frameData[9], frameData[10], frameData[11]]
        # leftKneeRotStart = (R.from_euler('ZYX', leftKneeRotStart, degrees=True)).as_quat()
        # leftKneeRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11]]
        # leftKneeRotEnd = (R.from_euler('ZYX', leftKneeRotEnd, degrees=True)).as_quat()
        # leftKneeRot = self.p0.getQuaternionSlerp(leftKneeRotStart, leftKneeRotEnd, frameFraction)

        # leftAnkleRotStart = [frameData[12], frameData[13], frameData[14]]
        # leftAnkleRotStart = (R.from_euler('ZYX', leftAnkleRotStart, degrees=True)).as_quat()
        # leftAnkleRotEnd = [frameDataNext[12], frameDataNext[13], frameDataNext[14]]
        # leftAnkleRotEnd = (R.from_euler('ZYX', leftAnkleRotEnd, degrees=True)).as_quat()
        # leftAnkleRot = self.p0.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd, frameFraction)

        # leftToeRotStart = [frameData[15], frameData[16], frameData[17]]
        # leftToeRotStart = (R.from_euler('ZYX', leftToeRotStart, degrees=True)).as_quat()
        # leftToeRotEnd = [frameDataNext[15], frameDataNext[16], frameDataNext[17]]
        # leftToeRotEnd = (R.from_euler('ZYX', leftToeRotEnd, degrees=True)).as_quat()
        # leftToeRot = self.p0.getQuaternionSlerp(leftToeRotStart, leftToeRotEnd, frameFraction)

        # rightHipRotStart = [frameData[18], frameData[19], frameData[20]]
        # rightHipRotStart = (R.from_euler('ZYX', rightHipRotStart, degrees=True)).as_quat()
        # rightHipRotEnd = [frameDataNext[18], frameDataNext[19], frameDataNext[20]]
        # rightHipRotEnd = (R.from_euler('ZYX', rightHipRotEnd, degrees=True)).as_quat()
        # rightHipRot = self.p0.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd, frameFraction)

        # rightKneeRotStart = [frameData[21], frameData[22], frameData[23]]
        # rightKneeRotStart = (R.from_euler('ZYX', rightKneeRotStart, degrees=True)).as_quat()
        # rightKneeRotEnd = [frameDataNext[21], frameDataNext[22], frameDataNext[23]]
        # rightKneeRotEnd = (R.from_euler('ZYX', rightKneeRotEnd, degrees=True)).as_quat()
        # rightKneeRot = self.p0.getQuaternionSlerp(rightKneeRotStart, rightKneeRotEnd, frameFraction)

        # rightAnkleRotStart = [frameData[24], frameData[25], frameData[26]]
        # rightAnkleRotStart = (R.from_euler('ZYX', rightAnkleRotStart, degrees=True)).as_quat()
        # rightAnkleRotEnd = [frameDataNext[24], frameDataNext[25], frameDataNext[26]]
        # rightAnkleRotEnd = (R.from_euler('ZYX', rightAnkleRotEnd, degrees=True)).as_quat()
        # rightAnkleRot = self.p0.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd, frameFraction)

        # rightToeRotStart = [frameData[27], frameData[28], frameData[29]]
        # rightToeRotStart = (R.from_euler('ZYX', rightToeRotStart, degrees=True)).as_quat()
        # rightToeRotEnd = [frameDataNext[27], frameDataNext[28], frameDataNext[29]]
        # rightToeRotEnd = (R.from_euler('ZYX', rightToeRotEnd, degrees=True)).as_quat()
        # rightToeRot = self.p0.getQuaternionSlerp(rightToeRotStart, rightToeRotEnd, frameFraction)

        # spineRotStart = [frameData[30], frameData[31], frameData[32]]
        # spineRotStart = (R.from_euler('ZYX', spineRotStart, degrees=True)).as_quat()
        # spineRotEnd = [frameDataNext[30], frameDataNext[31], frameDataNext[32]]
        # spineRotEnd = (R.from_euler('ZYX', spineRotEnd, degrees=True)).as_quat()
        # spineRot = self.p0.getQuaternionSlerp(spineRotStart, spineRotEnd, frameFraction)

        # spine1RotStart = [frameData[33], frameData[34], frameData[35]]
        # spine1RotStart = (R.from_euler('ZYX', spine1RotStart, degrees=True)).as_quat()
        # spine1RotEnd = [frameDataNext[33], frameDataNext[34], frameDataNext[35]]
        # spine1RotEnd = (R.from_euler('ZYX', spine1RotEnd, degrees=True)).as_quat()
        # spine1Rot = self.p0.getQuaternionSlerp(spine1RotStart, spine1RotEnd, frameFraction)

        # spine2RotStart = [frameData[36], frameData[37], frameData[38]]
        # spine2RotStart = (R.from_euler('ZYX', spine2RotStart, degrees=True)).as_quat()
        # spine2RotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38]]
        # spine2RotEnd = (R.from_euler('ZYX', spine2RotEnd, degrees=True)).as_quat()
        # spine2Rot = self.p0.getQuaternionSlerp(spine2RotStart, spine2RotEnd, frameFraction)

        # neckRotStart = [frameData[39], frameData[40], frameData[41]]
        # neckRotStart = (R.from_euler('ZYX', neckRotStart, degrees=True)).as_quat()
        # neckRotEnd = [frameDataNext[39], frameDataNext[40], frameDataNext[41]]
        # neckRotEnd = (R.from_euler('ZYX', neckRotEnd, degrees=True)).as_quat()
        # neckRot = self.p0.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)

        # headRotStart = [frameData[42], frameData[43], frameData[44]]
        # headRotStart = (R.from_euler('ZYX', headRotStart, degrees=True)).as_quat()
        # headRotEnd = [frameDataNext[42], frameDataNext[43], frameDataNext[44]]
        # headRotEnd = (R.from_euler('ZYX', headRotEnd, degrees=True)).as_quat()
        # headRot = self.p0.getQuaternionSlerp(headRotStart, headRotEnd, frameFraction)

        # leftShoulderRotStart = [frameData[45], frameData[46], frameData[47]]
        # leftShoulderRotStart = (R.from_euler('ZYX', leftShoulderRotStart, degrees=True)).as_quat()
        # leftShoulderRotEnd = [frameDataNext[45], frameDataNext[46], frameDataNext[47]]
        # leftShoulderRotEnd = (R.from_euler('ZYX', leftShoulderRotEnd, degrees=True)).as_quat()
        # leftShoulderRot = self.p0.getQuaternionSlerp(leftShoulderRotStart, leftShoulderRotEnd, frameFraction)

        # leftArmRotStart = [frameData[48], frameData[49], frameData[50]]
        # leftArmRotStart = (R.from_euler('ZYX', leftArmRotStart, degrees=True)).as_quat()
        # leftArmRotEnd = [frameDataNext[48], frameDataNext[49], frameDataNext[50]]
        # leftArmRotEnd = (R.from_euler('ZYX', leftArmRotEnd, degrees=True)).as_quat()
        # leftArmRot = self.p0.getQuaternionSlerp(leftArmRotStart, leftArmRotEnd, frameFraction)

        # leftElbowRotStart = [frameData[51], frameData[52], frameData[53]]
        # leftElbowRotStart = (R.from_euler('ZYX', leftElbowRotStart, degrees=True)).as_quat()
        # leftElbowRotEnd = [frameDataNext[51], frameDataNext[52], frameDataNext[53]]
        # leftElbowRotEnd = (R.from_euler('ZYX', leftElbowRotEnd, degrees=True)).as_quat()
        # leftElbowRot = self.p0.getQuaternionSlerp(leftElbowRotStart, leftElbowRotEnd, frameFraction)

        # leftWristRotStart = [frameData[54], frameData[55], frameData[56]]
        # leftWristRotStart = (R.from_euler('ZYX', leftWristRotStart, degrees=True)).as_quat()
        # leftWristRotEnd = [frameDataNext[54], frameDataNext[55], frameDataNext[56]]
        # leftWristRotEnd = (R.from_euler('ZYX', leftWristRotEnd, degrees=True)).as_quat()
        # leftWristRot = self.p0.getQuaternionSlerp(leftWristRotStart, leftWristRotEnd, frameFraction)

        # rightShoulderRotStart = [frameData[57], frameData[58], frameData[59]]
        # rightShoulderRotStart = (R.from_euler('ZYX', rightShoulderRotStart, degrees=True)).as_quat()
        # rightShoulderRotEnd = [frameDataNext[57], frameDataNext[58], frameDataNext[59]]
        # rightShoulderRotEnd = (R.from_euler('ZYX', rightShoulderRotEnd, degrees=True)).as_quat()
        # rightShoulderRot = self.p0.getQuaternionSlerp(rightShoulderRotStart, rightShoulderRotEnd, frameFraction)

        # rightArmRotStart = [frameData[60], frameData[61], frameData[62]]
        # rightArmRotStart = (R.from_euler('ZYX', rightArmRotStart, degrees=True)).as_quat()
        # rightArmRotEnd = [frameDataNext[60], frameDataNext[61], frameDataNext[62]]
        # rightArmRotEnd = (R.from_euler('ZYX', rightArmRotEnd, degrees=True)).as_quat()
        # rightArmRot = self.p0.getQuaternionSlerp(rightArmRotStart, rightArmRotEnd, frameFraction)

        # rightElbowRotStart = [frameData[63], frameData[64], frameData[65]]
        # rightElbowRotStart = (R.from_euler('ZYX', rightElbowRotStart, degrees=True)).as_quat()
        # rightElbowRotEnd = [frameDataNext[63], frameDataNext[64], frameDataNext[65]]
        # rightElbowRotEnd = (R.from_euler('ZYX', rightElbowRotEnd, degrees=True)).as_quat()
        # rightElbowRot = self.p0.getQuaternionSlerp(rightElbowRotStart, rightElbowRotEnd, frameFraction)

        # rightWristRotStart = [frameData[66], frameData[67], frameData[68]]
        # rightWristRotStart = (R.from_euler('ZYX', rightWristRotStart, degrees=True)).as_quat()
        # rightWristRotEnd = [frameDataNext[66], frameDataNext[67], frameDataNext[68]]
        # rightWristRotEnd = (R.from_euler('ZYX', rightWristRotEnd, degrees=True)).as_quat()
        # rightWristRot = self.p0.getQuaternionSlerp(rightWristRotStart, rightWristRotEnd, frameFraction)

        # chest = 1
        # neck = 2
        # rightHip = 3
        # rightKnee = 4
        # rightAnkle = 5
        # rightShoulder = 6
        # rightElbow = 7
        # leftHip = 9
        # leftKnee = 10
        # leftAnkle = 11
        # leftShoulder = 12
        # leftElbow = 13

        # get angles from BVH 
        angles_start = np.array([frameData[start:end] for start, end in indices])
        angles_end = np.array([frameDataNext[start:end] for start, end in indices])
        # conver to quarternions
        quats_start = R.from_euler('ZYX', angles_start, degrees=True).as_quat()
        quats_end = R.from_euler('ZYX', angles_end, degrees=True).as_quat()

        # Perform SLERP interpolation
        quats_interpolated = [self.p0.getQuaternionSlerp(q_start, q_end, frameFraction) for q_start, q_end in zip(quats_start, quats_end)]

#                             Configures joint motor control

        #                                              bullet_indice, 
        self.p0.resetJointStateMultiDof(self.humanoid, left_hip, quats_interpolated[0])
        self.p0.resetJointStateMultiDof(self.humanoid, left_knee, quats_interpolated[1])
        self.p0.resetJointStateMultiDof(self.humanoid, left_ankle, quats_interpolated[2])
        self.p0.resetJointStateMultiDof(self.humanoid, left_toe, quats_interpolated[3])

        self.p0.resetJointStateMultiDof(self.humanoid, right_hip, quats_interpolated[4]) 
        self.p0.resetJointStateMultiDof(self.humanoid, right_knee, quats_interpolated[5]) 
        self.p0.resetJointStateMultiDof(self.humanoid, right_ankle, quats_interpolated[6]) 
        self.p0.resetJointStateMultiDof(self.humanoid, right_toe, quats_interpolated[7]) 

        self.p0.resetJointStateMultiDof(self.humanoid, spine, quats_interpolated[8])
        self.p0.resetJointStateMultiDof(self.humanoid, spine1, quats_interpolated[9])
        self.p0.resetJointStateMultiDof(self.humanoid, spine2, quats_interpolated[10])
        self.p0.resetJointStateMultiDof(self.humanoid, neck, quats_interpolated[11])
        self.p0.resetJointStateMultiDof(self.humanoid, head, quats_interpolated[12])
        
        self.p0.resetJointStateMultiDof(self.humanoid, left_shoulder, quats_interpolated[13])
        self.p0.resetJointStateMultiDof(self.humanoid, left_arm, quats_interpolated[14])
        self.p0.resetJointStateMultiDof(self.humanoid, left_elbow, quats_interpolated[15])
        self.p0.resetJointStateMultiDof(self.humanoid, left_wrist, quats_interpolated[16])

        self.p0.resetJointStateMultiDof(self.humanoid, right_shoulder, quats_interpolated[17])
        self.p0.resetJointStateMultiDof(self.humanoid, right_arm, quats_interpolated[18])
        self.p0.resetJointStateMultiDof(self.humanoid, right_elbow, quats_interpolated[19])
        self.p0.resetJointStateMultiDof(self.humanoid, right_wrist, quats_interpolated[20])
        
        # placed in BVH structure layout
        self.k_targets = [quats_interpolated[0], quats_interpolated[1], quats_interpolated[2], quats_interpolated[3],
                quats_interpolated[4], quats_interpolated[5], quats_interpolated[6], quats_interpolated[7],
                quats_interpolated[8], quats_interpolated[9], quats_interpolated[10], quats_interpolated[11], quats_interpolated[12],
                quats_interpolated[13], quats_interpolated[14], quats_interpolated[15], quats_interpolated[16],
                quats_interpolated[17], quats_interpolated[18], quats_interpolated[19], quats_interpolated[20]]

        # print('0########',self._baseLinVel)
        # print('1########',self._baseAngVel)
        # self.k_targets = [[chestRot,
        #     neckRot,
        #     rightHipRot, 
        #     rightAnkleRot,
        #     rightShoulderRot,
        #     leftHipRot, 
        #     leftAnkleRot,
        #     leftShoulderRot]
        #     [rightKneeRot, 
        #     rightElbowRot, 
        #     leftKneeRot, 
        #     leftElbowRot]]
        
        # self.k_targets = [chestRot,
        #                   neckRot,
        #                   rightHipRot, 
        #                   rightKneeRot, # R
        #                   rightAnkleRot,
        #                   rightShoulderRot,
        #                   rightElbowRot, # R
        #                   leftHipRot, 
        #                   leftKneeRot, # R
        #                   leftAnkleRot,
        #                   leftShoulderRot, 
        #                   leftElbowRot] # R
        # 8sph 4rev 


    def get_first_state(self):
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
        num_joints = self.p0.getNumJoints(self.humanoid)
        
        # Loop through each link to get its world space state
        for i in range(num_joints):
            # Get position, orientation (quaternion)
            link_state = self.p0.getLinkState(self.humanoid, i, computeLinkVelocity=True)
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

        
    def get_state(self):
        # For storing data of all rigid bodies
        all_pos = []       # World space positions
        all_vel = []       # World space linear velocities
        all_rot = []       # World space orientations (quaternions)
        all_rot_vel = []   # World space angular velocities
        
        num_joints = self.p0.getNumJoints(self.humanoid)
        
        # Loop through each link to get its world space state
        for i in range(num_joints):
            # Get position, orientation (quaternion)
            link_state = self.p0.getLinkState(self.humanoid, i, computeLinkVelocity=True)
            link_pos = link_state[0]        # vec3, list of 3 floats | world position of the URDF link mass origin
            link_ori = link_state[1]        # vec4, list of 4 floats | world orientation of the URDF link mass origin
            # link_lin_vel = link_state[6]    # vec3, list of 3 floats | Cartesian world velocity
            # link_ang_vel = link_state[7]    # vec3, list of 3 floats | Cartesian world rot velocity
            link_vel = self.ComputeLinVel(self.kin_state[0][i], link_pos, 1./30.)
            link_ang_vel = self.ComputeAngVel(self.kin_state[2][i], link_ori, 1./30.) 
            # Append link information to lists
            all_pos.append(link_pos)
            all_vel.append(link_vel)
            all_rot.append(link_ori)
            all_rot_vel.append(link_ang_vel) 
        # print('KIN_VELOCITY',all_vel)
        # print('KIN_ANG_VELOCITY',all_rot_vel)
        # Format the final observation structure
        state = [
            all_pos,                 # List of [x, y, z] for each rigid body
            all_vel,             # List of [vx, vy, vz] for each rigid body
            all_rot,                 # List of [x, y, z, w] quaternions for each rigid body
            all_rot_vel          # List of [wx, wy, wz] for each rigid body
        ]
        self.kin_state = state #!
        return state
