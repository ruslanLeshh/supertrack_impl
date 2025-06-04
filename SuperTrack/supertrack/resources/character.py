# import pybullet as p
import numpy as np
import os
import math
from scipy.spatial.transform import Rotation as R

#BASE = -1 
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

indices = [ #21
    left_hip,
    left_knee,
    left_ankle,
    left_toe,
    right_hip,
    right_knee,
    right_ankle,
    right_toe,
    spine,
    spine1,
    spine2,
    neck,
    head,
    left_shoulder,
    left_arm,
    left_elbow,
    left_wrist,
    right_shoulder,
    right_arm,
    right_elbow,
    right_wrist
]

kps = [1.0]*21
# kds = [0.1]*21
maxForce = 100
maxForces = [[maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
            [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
            [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],

            [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
            [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
            [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
            
            [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
            ]

# targetPositions = [
#     leftHipRot,
#     leftKneeRot,
#     leftAnkleRot,
#     leftToeRot,
#     rightHipRot,
#     rightKneeRot,
#     rightAnkleRot,
#     rightToeRot,
#     spineRot,
#     spine1Rot,
#     spine2Rot,
#     neckRot,
#     headRot,
#     leftShoulderRot,
#     leftArmRot,
#     leftElbowRot,
#     leftWristRot,
#     rightShoulderRot,
#     rightArmRot,
#     rightElbowRot,
#     rightWristRot
# ]


class Character:
    def __init__(self, client, mocap_data, rand_frame):
        self.p0 = client
        self.mocap_data = mocap_data

        if self.p0.getPhysicsEngineParameters() is None:
            print("Errorrrrrrrr: Not connected to the physics server.")
        else:
            print("Physicssssss server connected in Character.")
        
        # self.mocap_data[rand_frame] = self.mocap_data['Frames'][0]
        # basePos = [self.mocap_data[rand_frame][1], self.mocap_data[rand_frame][2], self.mocap_data[rand_frame][3]]  # 0th frame position (x, y, z)
        # baseOrn = [self.mocap_data[rand_frame][5], self.mocap_data[rand_frame][6], self.mocap_data[rand_frame][7], self.mocap_data[rand_frame][4]]  # 0th frame orientation (quaternion)
        # y2zPos = [0, 0, 0.0]
        # y2zOrn = self.p0.getQuaternionFromEuler([1.57, 0, 0])  # Rotate 90 degrees around X-axis
        # basePos, baseOrn = self.p0.multiplyTransforms(y2zPos, y2zOrn, basePos, baseOrn)

        # chestRot = [self.mocap_data[rand_frame][9], self.mocap_data[rand_frame][10], self.mocap_data[rand_frame][11], self.mocap_data[rand_frame][8]]
        # neckRot = [self.mocap_data[rand_frame][13], self.mocap_data[rand_frame][14], self.mocap_data[rand_frame][15], self.mocap_data[rand_frame][12]]
        # rightHipRot = [self.mocap_data[rand_frame][17], self.mocap_data[rand_frame][18], self.mocap_data[rand_frame][19], self.mocap_data[rand_frame][16]]
        # rightKneeRot = [self.mocap_data[rand_frame][20]]
        # rightAnkleRot = [self.mocap_data[rand_frame][22], self.mocap_data[rand_frame][23], self.mocap_data[rand_frame][24], self.mocap_data[rand_frame][21]]
        # rightShoulderRot = [self.mocap_data[rand_frame][26], self.mocap_data[rand_frame][27], self.mocap_data[rand_frame][28], self.mocap_data[rand_frame][25]]
        # rightElbowRot = [self.mocap_data[rand_frame][29]]
        # leftHipRot = [self.mocap_data[rand_frame][31], self.mocap_data[rand_frame][32], self.mocap_data[rand_frame][33], self.mocap_data[rand_frame][30]]
        # leftKneeRot = [self.mocap_data[rand_frame][34]]
        # leftAnkleRot = [self.mocap_data[rand_frame][36], self.mocap_data[rand_frame][37], self.mocap_data[rand_frame][38], self.mocap_data[rand_frame][35]]
        # leftShoulderRot = [self.mocap_data[rand_frame][40], self.mocap_data[rand_frame][41], self.mocap_data[rand_frame][42], self.mocap_data[rand_frame][39]]
        # leftElbowRot = [self.mocap_data[rand_frame][43]]
        
        basePos1Start = [self.mocap_data[rand_frame][0]*0.01, self.mocap_data[rand_frame][1]*0.01, self.mocap_data[rand_frame][2]*0.01]

        # Orientation spherical linear interpolation
        baseOrn1Start = [self.mocap_data[rand_frame][5], self.mocap_data[rand_frame][4], self.mocap_data[rand_frame][3]]  # y x z
        baseOrn1Start = (R.from_euler('xyz',baseOrn1Start, degrees=True)).as_quat()
        
        # self.p0.resetBasePositionAndOrientation(self.humanoid, basePos1Start, baseOrn1Start)

        leftHipRotStart = [self.mocap_data[rand_frame][6], self.mocap_data[rand_frame][7], self.mocap_data[rand_frame][8]]
        leftHipRotStart = (R.from_euler('ZYX', leftHipRotStart, degrees=True)).as_quat()
        
        leftKneeRotStart = [self.mocap_data[rand_frame][9], self.mocap_data[rand_frame][10], self.mocap_data[rand_frame][11]]
        leftKneeRotStart = (R.from_euler('ZYX', leftKneeRotStart, degrees=True)).as_quat()
        
        leftAnkleRotStart = [self.mocap_data[rand_frame][12], self.mocap_data[rand_frame][13], self.mocap_data[rand_frame][14]]
        leftAnkleRotStart = (R.from_euler('ZYX', leftAnkleRotStart, degrees=True)).as_quat()

        leftToeRotStart = [self.mocap_data[rand_frame][15], self.mocap_data[rand_frame][16], self.mocap_data[rand_frame][17]]
        leftToeRotStart = (R.from_euler('ZYX', leftToeRotStart, degrees=True)).as_quat()

        rightHipRotStart = [self.mocap_data[rand_frame][18], self.mocap_data[rand_frame][19], self.mocap_data[rand_frame][20]]
        rightHipRotStart = (R.from_euler('ZYX', rightHipRotStart, degrees=True)).as_quat()

        rightKneeRotStart = [self.mocap_data[rand_frame][21], self.mocap_data[rand_frame][22], self.mocap_data[rand_frame][23]]
        rightKneeRotStart = (R.from_euler('ZYX', rightKneeRotStart, degrees=True)).as_quat()

        rightAnkleRotStart = [self.mocap_data[rand_frame][24], self.mocap_data[rand_frame][25], self.mocap_data[rand_frame][26]]
        rightAnkleRotStart = (R.from_euler('ZYX', rightAnkleRotStart, degrees=True)).as_quat()

        rightToeRotStart = [self.mocap_data[rand_frame][27], self.mocap_data[rand_frame][28], self.mocap_data[rand_frame][29]]
        rightToeRotStart = (R.from_euler('ZYX', rightToeRotStart, degrees=True)).as_quat()

        spineRotStart = [self.mocap_data[rand_frame][30], self.mocap_data[rand_frame][31], self.mocap_data[rand_frame][32]]
        spineRotStart = (R.from_euler('ZYX', spineRotStart, degrees=True)).as_quat()

        spine1RotStart = [self.mocap_data[rand_frame][33], self.mocap_data[rand_frame][34], self.mocap_data[rand_frame][35]]
        spine1RotStart = (R.from_euler('ZYX', spine1RotStart, degrees=True)).as_quat()

        spine2RotStart = [self.mocap_data[rand_frame][36], self.mocap_data[rand_frame][37], self.mocap_data[rand_frame][38]]
        spine2RotStart = (R.from_euler('ZYX', spine2RotStart, degrees=True)).as_quat()

        neckRotStart = [self.mocap_data[rand_frame][39], self.mocap_data[rand_frame][40], self.mocap_data[rand_frame][41]]
        neckRotStart = (R.from_euler('ZYX', neckRotStart, degrees=True)).as_quat()

        headRotStart = [self.mocap_data[rand_frame][42], self.mocap_data[rand_frame][43], self.mocap_data[rand_frame][44]]
        headRotStart = (R.from_euler('ZYX', headRotStart, degrees=True)).as_quat()

        leftShoulderRotStart = [self.mocap_data[rand_frame][45], self.mocap_data[rand_frame][46], self.mocap_data[rand_frame][47]]
        leftShoulderRotStart = (R.from_euler('ZYX', leftShoulderRotStart, degrees=True)).as_quat()

        leftArmRotStart = [self.mocap_data[rand_frame][48], self.mocap_data[rand_frame][49], self.mocap_data[rand_frame][50]]
        leftArmRotStart = (R.from_euler('ZYX', leftArmRotStart, degrees=True)).as_quat()

        leftElbowRotStart = [self.mocap_data[rand_frame][51], self.mocap_data[rand_frame][52], self.mocap_data[rand_frame][53]]
        leftElbowRotStart = (R.from_euler('ZYX', leftElbowRotStart, degrees=True)).as_quat()

        leftWristRotStart = [self.mocap_data[rand_frame][54], self.mocap_data[rand_frame][55], self.mocap_data[rand_frame][56]]
        leftWristRotStart = (R.from_euler('ZYX', leftWristRotStart, degrees=True)).as_quat()

        rightShoulderRotStart = [self.mocap_data[rand_frame][57], self.mocap_data[rand_frame][58], self.mocap_data[rand_frame][59]]
        rightShoulderRotStart = (R.from_euler('ZYX', rightShoulderRotStart, degrees=True)).as_quat()

        rightArmRotStart = [self.mocap_data[rand_frame][60], self.mocap_data[rand_frame][61], self.mocap_data[rand_frame][62]]
        rightArmRotStart = (R.from_euler('ZYX', rightArmRotStart, degrees=True)).as_quat()

        rightElbowRotStart = [self.mocap_data[rand_frame][63], self.mocap_data[rand_frame][64], self.mocap_data[rand_frame][65]]
        rightElbowRotStart = (R.from_euler('ZYX', rightElbowRotStart, degrees=True)).as_quat()

        rightWristRotStart = [self.mocap_data[rand_frame][66], self.mocap_data[rand_frame][67], self.mocap_data[rand_frame][68]]
        rightWristRotStart = (R.from_euler('ZYX', rightWristRotStart, degrees=True)).as_quat()

        # import model
        flags=self.p0.URDF_MAINTAIN_LINK_ORDER+self.p0.URDF_USE_SELF_COLLISION 
        self.humanoid = self.p0.loadURDF("11s3_I.urdf",
                            basePosition=basePos1Start,
                            baseOrientation=baseOrn1Start,
                            globalScaling=0.01,
                            useFixedBase=False,
                            flags=flags)

        # # defaoult setup
        # startPose = [
        #     2, 0.847532, 0, 0.9986781045, 0.01410400148, -0.0006980000731, -0.04942300517, 0.9988133229,
        #     0.009485003066, -0.04756001538, -0.004475001447, 1, 0, 0, 0, 0.9649395871, 0.02436898957,
        #     -0.05755497537, 0.2549218909, -0.249116, 0.9993661511, 0.009952001505, 0.03265400494,
        #     0.01009800153, 0.9854981188, -0.06440700776, 0.09324301124, -0.1262970152, 0.170571,
        #     0.9927545808, -0.02090099117, 0.08882396249, -0.07817796699, -0.391532, 0.9828788495,
        #     0.1013909845, -0.05515999155, 0.143618978, 0.9659421276, 0.1884590249, -0.1422460188,
        #     0.105854014, 0.581348
        # ]
        # startVel = [
        #     1.235314324, -0.008525509087, 0.1515293946, -1.161516553, 0.1866449799, -0.1050802848, 0,
        #     0.935706195, 0.08277326387, 0.3002461862, 0, 0, 0, 0, 0, 1.114409628, 0.3618553952,
        #     -0.4505575061, 0, -1.725374735, -0.5052852598, -0.8555179722, -0.2221173515, 0, -0.1837617357,
        #     0.00171895706, 0.03912837591, 0, 0.147945294, 1.837653345, 0.1534535548, 1.491385941, 0,
        #     -4.632454387, -0.9111172777, -1.300648184, -1.345694622, 0, -1.084238535, 0.1313680236,
        #     -0.7236998534, 0, -0.5278312973
        # ]
        # self.p0.resetBasePositionAndOrientation(self.humanoid, basePos, self.p0.getQuaternionFromEuler([1.57, 0, 0]))
        
        # set pose to first frame    
        self.p0.resetJointStateMultiDof(self.humanoid, left_hip, leftHipRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, left_knee, leftKneeRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, left_ankle, leftAnkleRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, left_toe, leftToeRotStart)

        self.p0.resetJointStateMultiDof(self.humanoid, right_hip, rightHipRotStart) #RightUpLeg
        self.p0.resetJointStateMultiDof(self.humanoid, right_knee, rightKneeRotStart) #RightLeg
        self.p0.resetJointStateMultiDof(self.humanoid, right_ankle, rightAnkleRotStart) #RightFoot
        self.p0.resetJointStateMultiDof(self.humanoid, right_toe, rightToeRotStart) #RightToe

        self.p0.resetJointStateMultiDof(self.humanoid, spine, spineRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, spine1, spine1RotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, spine2, spine2RotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, neck, neckRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, head, headRotStart)
        
        self.p0.resetJointStateMultiDof(self.humanoid, left_shoulder, leftShoulderRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, left_arm, leftArmRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, left_elbow, leftElbowRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, left_wrist, leftWristRotStart)

        self.p0.resetJointStateMultiDof(self.humanoid, right_shoulder, rightShoulderRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, right_arm, rightArmRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, right_elbow, rightElbowRotStart)
        self.p0.resetJointStateMultiDof(self.humanoid, right_wrist, rightWristRotStart)
         

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
                # self.p0.changeDynamics(self.humanoid, j, mass=30)

    # def get_ids(self):
    #     return self.car, self.client

    
    def apply_action(self, targetPositions):
        # print('ACTION   ',targetPositions)
        self.p0.setJointMotorControlMultiDofArray(self.humanoid,
                                    indices,
                                    self.p0.POSITION_CONTROL, # applly stable PD controller to each joint.
                                    targetPositions=targetPositions, # PD target for each joint
                                    # targetVelocities=[[0,0,0]]*21,
                                    positionGains=kps, # specifies for each joint how strongly it will try to follow these desired positions
                                    # velocityGains=kds, # (control overshooting or oscillating)
                                    forces=maxForces # maximum allowable force the PD controller
                                    )

    def get_state(self):
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
