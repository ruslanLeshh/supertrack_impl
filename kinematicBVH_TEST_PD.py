import pybullet as p
import time
import csv 
import json
import math
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import numpy as np



def get_state(humanoid):
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
    # base_pos, base_ori = p.getBasePositionAndOrientation(self.humanoid, self.client)
    # # getBaseVelocity returns a list of two vector3 values (3 floats in a list) representing the linear velocity [x,y,z] and angular velocity [wx,wy,wz]
    # # in Cartesian worldspace coordinates.
    # base_lin_vel, base_ang_vel = p.getBaseVelocity(self.humanoid, self.client)
    
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
        link_state = p.getLinkState(humanoid, i, computeLinkVelocity=True, computeForwardKinematics=True)
        link_pos = link_state[4]        # vec3, list of 3 floats | world position of the URDF link frame
        link_ori = link_state[5]        # vec4, list of 4 floats | world orientation of the URDF link frame
        link_lin_vel = link_state[6]    # vec3, list of 3 floats | Cartesian world velocity
        link_ang_vel = link_state[7]    # vec3, list of 3 floats | Cartesian world velocity
        
        # Append link information to lists
        all_pos.append(link_pos)
        all_vel.append(link_lin_vel)
        all_rot.append(link_ori)
        all_rot_vel.append(link_ang_vel)
    
    # Format the final observation structure
    state = [
        all_pos,                 # List of [x, y, z] for each rigid body
        all_rot,                 # List of [x, y, z, w] quaternions for each rigid body
        all_vel,             # List of [vx, vy, vz] for each rigid body
        all_rot_vel          # List of [wx, wy, wz] for each rigid body
    ]
    
    return state





#                                   BULLET SETUP




useGUI = True

if useGUI:
  p.connect(p.GUI)
else:
 p.connect(p.DIRECT)

plane_id = p.loadURDF('Simple-Driving/simple_driving/resources/simpleplane.urdf', basePosition=[0, -0, 0],baseOrientation= p.getQuaternionFromEuler([1.57,0,0]), useFixedBase=False)

# from pdControllerExplicit import PDControllerExplicitMultiDof
# from pdControllerStable import PDControllerStableMultiDof

# explicitPD = PDControllerExplicitMultiDof(p) # PD for kinematic 
# stablePD = PDControllerStableMultiDof(p) # PD for simulated

p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
p.resetDebugVisualizerCamera(cameraDistance=3,
                             cameraYaw=180,
                             cameraPitch=-20,
                             cameraTargetPosition=[0, 0.1, 0])

t = 0.0
dt = 1./240. 

currentTime = time.time()
accumulator = 0.0

p.setPhysicsEngineParameter(numSolverIterations=100, fixedTimeStep=dt, numSubSteps=1) 


# p.setPhysicsEngineParameter(fixedTimeStep=timeStep) # simulation update
# p.setRealTimeSimulation(1)
# p.setPhysicsEngineParameter(fixedTimeStep=1. / 240.,
#                     solverResidualThreshold= -9, # velocity threshold
#                     numSolverIterations=30, # detection accuracy
#                     numSubSteps=2) # further TimeStep subdivision



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

jointTypes = ["JOINT_REVOLUTE", "JOINT_PRISMATIC", "JOINT_SPHERICAL", "JOINT_PLANAR", "JOINT_FIXED"]

startLocations = [[0, 0, 0]]

flags=p.URDF_MAINTAIN_LINK_ORDER+p.URDF_USE_SELF_COLLISION 
humanoid = p.loadURDF("11s3_I.urdf",
                      basePosition=startLocations[0],
                      globalScaling=0.01,
                      useFixedBase=False,
                      flags=flags)

start=0

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

        # Control the motor for the spherical joint
        p.setJointMotorControlMultiDof(humanoid, j, p.POSITION_CONTROL, targetPosition=[0, 0, 0, 1], targetVelocity=[0, 0, 0], positionGain=0, velocityGain=1, force=[0, 0, 0])

        # p.changeDynamics(humanoid, j,mass=30)


    # p.setJointMotorControl2(humanoid,
    #                         j,
    #                         p.POSITION_CONTROL,
    #                         targetPosition=0,
    #                         positionGain=0,
    #                         targetVelocity=0,
    #                         force=0)

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

p.setGravity(0, -9.8, 0)



# basePos1Start = [mocap_data[0][0]*0.01, mocap_data[0][1]*0.01, mocap_data[0][2]*0.01]

# # Orientation spherical linear interpolation
# baseOrn1Start = [mocap_data[0][5], mocap_data[0][4], mocap_data[0][3]]  # y x z
# baseOrn1Start = (R.from_euler('xyz',baseOrn1Start, degrees=True)).as_quat()


# p.resetBasePositionAndOrientation(humanoid, basePos1Start, baseOrn1Start)
# for j in range(p.getNumJoints(humanoid)):
#         p.resetJointStateMultiDof(humanoid, j, targetValue=[0,0,0], targetVelocity=[0,0,0])
# p.setTimeOut(1000)

basePos1Start = [mocap_data[0][0]*0.01, mocap_data[0][1]*0.01, mocap_data[0][2]*0.01]

# Orientation spherical linear interpolation
baseOrn1Start = [mocap_data[0][5], mocap_data[0][4], mocap_data[0][3]]  # y x z
baseOrn1Start = (R.from_euler('xyz',baseOrn1Start, degrees=True)).as_quat()

p.resetBasePositionAndOrientation(humanoid, basePos1Start, baseOrn1Start)

leftHipRotStart = [mocap_data[0][6], mocap_data[0][7], mocap_data[0][8]]
leftHipRotStart = (R.from_euler('ZYX', leftHipRotStart, degrees=True)).as_quat()

leftKneeRotStart = [mocap_data[0][9], mocap_data[0][10], mocap_data[0][11]]
leftKneeRotStart = (R.from_euler('ZYX', leftKneeRotStart, degrees=True)).as_quat()

leftAnkleRotStart = [mocap_data[0][12], mocap_data[0][13], mocap_data[0][14]]
leftAnkleRotStart = (R.from_euler('ZYX', leftAnkleRotStart, degrees=True)).as_quat()

leftToeRotStart = [mocap_data[0][15], mocap_data[0][16], mocap_data[0][17]]
leftToeRotStart = (R.from_euler('ZYX', leftToeRotStart, degrees=True)).as_quat()

rightHipRotStart = [mocap_data[0][18], mocap_data[0][19], mocap_data[0][20]]
rightHipRotStart = (R.from_euler('ZYX', rightHipRotStart, degrees=True)).as_quat()

rightKneeRotStart = [mocap_data[0][21], mocap_data[0][22], mocap_data[0][23]]
rightKneeRotStart = (R.from_euler('ZYX', rightKneeRotStart, degrees=True)).as_quat()

rightAnkleRotStart = [mocap_data[0][24], mocap_data[0][25], mocap_data[0][26]]
rightAnkleRotStart = (R.from_euler('ZYX', rightAnkleRotStart, degrees=True)).as_quat()

rightToeRotStart = [mocap_data[0][27], mocap_data[0][28], mocap_data[0][29]]
rightToeRotStart = (R.from_euler('ZYX', rightToeRotStart, degrees=True)).as_quat()

spineRotStart = [mocap_data[0][30], mocap_data[0][31], mocap_data[0][32]]
spineRotStart = (R.from_euler('ZYX', spineRotStart, degrees=True)).as_quat()

spine1RotStart = [mocap_data[0][33], mocap_data[0][34], mocap_data[0][35]]
spine1RotStart = (R.from_euler('ZYX', spine1RotStart, degrees=True)).as_quat()

spine2RotStart = [mocap_data[0][36], mocap_data[0][37], mocap_data[0][38]]
spine2RotStart = (R.from_euler('ZYX', spine2RotStart, degrees=True)).as_quat()

neckRotStart = [mocap_data[0][39], mocap_data[0][40], mocap_data[0][41]]
neckRotStart = (R.from_euler('ZYX', neckRotStart, degrees=True)).as_quat()

headRotStart = [mocap_data[0][42], mocap_data[0][43], mocap_data[0][44]]
headRotStart = (R.from_euler('ZYX', headRotStart, degrees=True)).as_quat()

leftShoulderRotStart = [mocap_data[0][45], mocap_data[0][46], mocap_data[0][47]]
leftShoulderRotStart = (R.from_euler('ZYX', leftShoulderRotStart, degrees=True)).as_quat()

leftArmRotStart = [mocap_data[0][48], mocap_data[0][49], mocap_data[0][50]]
leftArmRotStart = (R.from_euler('ZYX', leftArmRotStart, degrees=True)).as_quat()

leftElbowRotStart = [mocap_data[0][51], mocap_data[0][52], mocap_data[0][53]]
leftElbowRotStart = (R.from_euler('ZYX', leftElbowRotStart, degrees=True)).as_quat()

leftWristRotStart = [mocap_data[0][54], mocap_data[0][55], mocap_data[0][56]]
leftWristRotStart = (R.from_euler('ZYX', leftWristRotStart, degrees=True)).as_quat()

rightShoulderRotStart = [mocap_data[0][57], mocap_data[0][58], mocap_data[0][59]]
rightShoulderRotStart = (R.from_euler('ZYX', rightShoulderRotStart, degrees=True)).as_quat()

rightArmRotStart = [mocap_data[0][60], mocap_data[0][61], mocap_data[0][62]]
rightArmRotStart = (R.from_euler('ZYX', rightArmRotStart, degrees=True)).as_quat()

rightElbowRotStart = [mocap_data[0][63], mocap_data[0][64], mocap_data[0][65]]
rightElbowRotStart = (R.from_euler('ZYX', rightElbowRotStart, degrees=True)).as_quat()

rightWristRotStart = [mocap_data[0][66], mocap_data[0][67], mocap_data[0][68]]
rightWristRotStart = (R.from_euler('ZYX', rightWristRotStart, degrees=True)).as_quat()

p.resetJointStateMultiDof(humanoid, left_hip, leftHipRotStart)
p.resetJointStateMultiDof(humanoid, left_knee, leftKneeRotStart)
p.resetJointStateMultiDof(humanoid, left_ankle, leftAnkleRotStart)
p.resetJointStateMultiDof(humanoid, left_toe, leftToeRotStart)

p.resetJointStateMultiDof(humanoid, right_hip, rightHipRotStart) #RightUpLeg
p.resetJointStateMultiDof(humanoid, right_knee, rightKneeRotStart) #RightLeg
p.resetJointStateMultiDof(humanoid, right_ankle, rightAnkleRotStart) #RightFoot
p.resetJointStateMultiDof(humanoid, right_toe, rightToeRotStart) #RightToe

p.resetJointStateMultiDof(humanoid, spine, spineRotStart)
p.resetJointStateMultiDof(humanoid, spine1, spine1RotStart)
p.resetJointStateMultiDof(humanoid, spine2, spine2RotStart)
p.resetJointStateMultiDof(humanoid, neck, neckRotStart)
p.resetJointStateMultiDof(humanoid, head, headRotStart)

p.resetJointStateMultiDof(humanoid, left_shoulder, leftShoulderRotStart)
p.resetJointStateMultiDof(humanoid, left_arm, leftArmRotStart)
p.resetJointStateMultiDof(humanoid, left_elbow, leftElbowRotStart)
p.resetJointStateMultiDof(humanoid, left_wrist, leftWristRotStart)

p.resetJointStateMultiDof(humanoid, right_shoulder, rightShoulderRotStart)
p.resetJointStateMultiDof(humanoid, right_arm, rightArmRotStart)
p.resetJointStateMultiDof(humanoid, right_elbow, rightElbowRotStart)
p.resetJointStateMultiDof(humanoid, right_wrist, rightWristRotStart)
time.sleep(1)

def quaternion_conjugate(q):
    """Returns the conjugate of a quaternion."""
    x, y, z, w = q
    return np.array([-x, -y, -z, w])

def quaternion_multiply(quaternion1, quaternion0): # [x,y,z,w]
    # print("MULTIPLY\n\n",quaternion0)
    # print("MULTIPLY\n\n",quaternion1)
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,  # x component
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0, # y component
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,  # z component
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0  # w component
    ])
def rotate_vector_by_quaternion(quaternion, vector):
    """Rotates a vector by a quaternion."""
    # Convert vector to quaternion form (x, y, z, w=0)
    vector_quaternion = np.array([vector[0], vector[1], vector[2], 0])
    # Perform the rotation: q * v * q^-1 (->)
    q_conjugate = quaternion_conjugate(quaternion)
    rotated_quaternion = quaternion_multiply(quaternion_multiply(quaternion, vector_quaternion),q_conjugate)
    # Return the rotated vector
    return rotated_quaternion[:3]

def quaternion_multiply(quaternion1, quaternion0): # [x,y,z,w]
    # print("MULTIPLY\n\n",quaternion0)
    # print("MULTIPLY\n\n",quaternion1)
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,  # x component
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0, # y component
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,  # z component
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0  # w component
    ])
def quaternion_inverse(quaternion):
    """Return inverse of quaternion in [x, y, z, w] format."""
    quaternion = np.array(quaternion, dtype=np.float64)
    q = quaternion.copy()
    np.negative(q[:3], q[:3])
    return q / np.dot(quaternion, quaternion)

def buildHeadingTrans(rootOrn):
    # align root transform 'forward' with world-space x axis
    eul = p.getEulerFromQuaternion(rootOrn)
    refDir = [0, 1, 0]
    rotVec = p.rotateVector(rootOrn, refDir)
    heading = math.atan2(-rotVec[2], rotVec[1])
    # heading2 = eul[1]
    # print("heading=",heading)
    headingOrn = p.getQuaternionFromAxisAngle([0, 0, 1], -heading)
    return headingOrn

rootPos, rootOrn = p.getBasePositionAndOrientation(humanoid)
invRootPos = [-rootPos[0], 0, -rootPos[2]]
headingOrn = buildHeadingTrans(rootOrn)
invOrigTransPos, invOrigTransOrn = p.multiplyTransforms([0, 0, 0],
                                                        headingOrn,
                                                        invRootPos,
                                                        [0, 0, 0, 1])
                                                        
p.resetBasePositionAndOrientation(humanoid, rootPos, invOrigTransOrn)

basePos, baseOrn = p.getBasePositionAndOrientation(humanoid)

# rootPosRel, dummy = p.multiplyTransforms(invOrigTransPos, invOrigTransOrn,
#                                         basePos, [0, 0, 0, 1])
                                                                     
test = rotate_vector_by_quaternion(headingOrn,invRootPos)

ee_state = p.getLinkState(humanoid, 1, computeLinkVelocity=True, computeForwardKinematics=True)
link_state = p.getLinkState(humanoid, 0, computeLinkVelocity=True, computeForwardKinematics=True)
state = p.getLinkState(humanoid, 2, computeLinkVelocity=True, computeForwardKinematics=True)

# linkPosLocal, linkOrnLocal = p.multiplyTransforms(
#     invOrigTransPos, invOrigTransOrn, ee_state[4], ee_state[5])

# if (linkOrnLocal[3] < 0):
#     linkOrnLocal = [-linkOrnLocal[0], -linkOrnLocal[1], -linkOrnLocal[2], -linkOrnLocal[3]]
# linkPosLocal = [
#     linkPosLocal[0] - rootPosRel[0], linkPosLocal[1] - rootPosRel[1],
#     linkPosLocal[2] - rootPosRel[2]
# ]

inv_orig_p, inv_orig_r = p.multiplyTransforms([0, 0, 0],
                                            state[5],
                                            [0, 0, 0],
                                            ee_state[5])

print(inv_orig_r)
# inv_orig_p2 = rotate_vector_by_quaternion(ee_state[5],link_state[4])
# print(inv_orig_p2)
inv_orig_r2 = quaternion_multiply(state[5],ee_state[5])
print(inv_orig_r2)

# print("0    ", ee_state[4])
# print("1    ", )
# print("2    ", )
# print("3    ", quaternion_inverse(link_state[5]))

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

def get_rot_vel():
    rot_vel = []

    for joint_name in joint_indices:
        j = joint_indices[joint_name]
        jointState = p.getJointStateMultiDof(humanoid, j)
        rot_vel.append(jointState[1])
    return rot_vel


once = True
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
    # p.resetBasePositionAndOrientation(humanoid, basePos1, baseOrn1)

    # For each joint the code performs quaternion SLERP to smoothly interpolate
    # between joint orientations from the current frame to the next frame.
    
    leftHipRotStart = [frameData[6], frameData[7], frameData[8]]
    leftHipRotStart = (R.from_euler('ZYX', leftHipRotStart, degrees=True)).as_quat()
    leftHipRotEnd = [frameDataNext[6], frameDataNext[7], frameDataNext[8]]
    leftHipRotEnd = (R.from_euler('ZYX', leftHipRotEnd, degrees=True)).as_quat()
    leftHipRot = p.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd, frameFraction)

    leftKneeRotStart = [frameData[9], frameData[10], frameData[11]]
    leftKneeRotStart = (R.from_euler('ZYX', leftKneeRotStart, degrees=True)).as_quat()
    leftKneeRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11]]
    leftKneeRotEnd = (R.from_euler('ZYX', leftKneeRotEnd, degrees=True)).as_quat()
    leftKneeRot = p.getQuaternionSlerp(leftKneeRotStart, leftKneeRotEnd, frameFraction)

    leftAnkleRotStart = [frameData[12], frameData[13], frameData[14]]
    leftAnkleRotStart = (R.from_euler('ZYX', leftAnkleRotStart, degrees=True)).as_quat()
    leftAnkleRotEnd = [frameDataNext[12], frameDataNext[13], frameDataNext[14]]
    leftAnkleRotEnd = (R.from_euler('ZYX', leftAnkleRotEnd, degrees=True)).as_quat()
    leftAnkleRot = p.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd, frameFraction)

    leftToeRotStart = [frameData[15], frameData[16], frameData[17]]
    leftToeRotStart = (R.from_euler('ZYX', leftToeRotStart, degrees=True)).as_quat()
    leftToeRotEnd = [frameDataNext[15], frameDataNext[16], frameDataNext[17]]
    leftToeRotEnd = (R.from_euler('ZYX', leftToeRotEnd, degrees=True)).as_quat()
    leftToeRot = p.getQuaternionSlerp(leftToeRotStart, leftToeRotEnd, frameFraction)

    rightHipRotStart = [frameData[18], frameData[19], frameData[20]]
    rightHipRotStart = (R.from_euler('ZYX', rightHipRotStart, degrees=True)).as_quat()
    rightHipRotEnd = [frameDataNext[18], frameDataNext[19], frameDataNext[20]]
    rightHipRotEnd = (R.from_euler('ZYX', rightHipRotEnd, degrees=True)).as_quat()
    rightHipRot = p.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd, frameFraction)

    rightKneeRotStart = [frameData[21], frameData[22], frameData[23]]
    rightKneeRotStart = (R.from_euler('ZYX', rightKneeRotStart, degrees=True)).as_quat()
    rightKneeRotEnd = [frameDataNext[21], frameDataNext[22], frameDataNext[23]]
    rightKneeRotEnd = (R.from_euler('ZYX', rightKneeRotEnd, degrees=True)).as_quat()
    rightKneeRot = p.getQuaternionSlerp(rightKneeRotStart, rightKneeRotEnd, frameFraction)

    rightAnkleRotStart = [frameData[24], frameData[25], frameData[26]]
    rightAnkleRotStart = (R.from_euler('ZYX', rightAnkleRotStart, degrees=True)).as_quat()
    rightAnkleRotEnd = [frameDataNext[24], frameDataNext[25], frameDataNext[26]]
    rightAnkleRotEnd = (R.from_euler('ZYX', rightAnkleRotEnd, degrees=True)).as_quat()
    rightAnkleRot = p.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd, frameFraction)

    rightToeRotStart = [frameData[27], frameData[28], frameData[29]]
    rightToeRotStart = (R.from_euler('ZYX', rightToeRotStart, degrees=True)).as_quat()
    rightToeRotEnd = [frameDataNext[27], frameDataNext[28], frameDataNext[29]]
    rightToeRotEnd = (R.from_euler('ZYX', rightToeRotEnd, degrees=True)).as_quat()
    rightToeRot = p.getQuaternionSlerp(rightToeRotStart, rightToeRotEnd, frameFraction)

    spineRotStart = [frameData[30], frameData[31], frameData[32]]
    spineRotStart = (R.from_euler('ZYX', spineRotStart, degrees=True)).as_quat()
    spineRotEnd = [frameDataNext[30], frameDataNext[31], frameDataNext[32]]
    spineRotEnd = (R.from_euler('ZYX', spineRotEnd, degrees=True)).as_quat()
    spineRot = p.getQuaternionSlerp(spineRotStart, spineRotEnd, frameFraction)

    spine1RotStart = [frameData[33], frameData[34], frameData[35]]
    spine1RotStart = (R.from_euler('ZYX', spine1RotStart, degrees=True)).as_quat()
    spine1RotEnd = [frameDataNext[33], frameDataNext[34], frameDataNext[35]]
    spine1RotEnd = (R.from_euler('ZYX', spine1RotEnd, degrees=True)).as_quat()
    spine1Rot = p.getQuaternionSlerp(spine1RotStart, spine1RotEnd, frameFraction)

    spine2RotStart = [frameData[36], frameData[37], frameData[38]]
    spine2RotStart = (R.from_euler('ZYX', spine2RotStart, degrees=True)).as_quat()
    spine2RotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38]]
    spine2RotEnd = (R.from_euler('ZYX', spine2RotEnd, degrees=True)).as_quat()
    spine2Rot = p.getQuaternionSlerp(spine2RotStart, spine2RotEnd, frameFraction)

    neckRotStart = [frameData[39], frameData[40], frameData[41]]
    neckRotStart = (R.from_euler('ZYX', neckRotStart, degrees=True)).as_quat()
    neckRotEnd = [frameDataNext[39], frameDataNext[40], frameDataNext[41]]
    neckRotEnd = (R.from_euler('ZYX', neckRotEnd, degrees=True)).as_quat()
    neckRot = p.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)

    headRotStart = [frameData[42], frameData[43], frameData[44]]
    headRotStart = (R.from_euler('ZYX', headRotStart, degrees=True)).as_quat()
    headRotEnd = [frameDataNext[42], frameDataNext[43], frameDataNext[44]]
    headRotEnd = (R.from_euler('ZYX', headRotEnd, degrees=True)).as_quat()
    headRot = p.getQuaternionSlerp(headRotStart, headRotEnd, frameFraction)

    leftShoulderRotStart = [frameData[45], frameData[46], frameData[47]]
    leftShoulderRotStart = (R.from_euler('ZYX', leftShoulderRotStart, degrees=True)).as_quat()
    leftShoulderRotEnd = [frameDataNext[45], frameDataNext[46], frameDataNext[47]]
    leftShoulderRotEnd = (R.from_euler('ZYX', leftShoulderRotEnd, degrees=True)).as_quat()
    leftShoulderRot = p.getQuaternionSlerp(leftShoulderRotStart, leftShoulderRotEnd, frameFraction)

    leftArmRotStart = [frameData[48], frameData[49], frameData[50]]
    leftArmRotStart = (R.from_euler('ZYX', leftArmRotStart, degrees=True)).as_quat()
    leftArmRotEnd = [frameDataNext[48], frameDataNext[49], frameDataNext[50]]
    leftArmRotEnd = (R.from_euler('ZYX', leftArmRotEnd, degrees=True)).as_quat()
    leftArmRot = p.getQuaternionSlerp(leftArmRotStart, leftArmRotEnd, frameFraction)

    leftElbowRotStart = [frameData[51], frameData[52], frameData[53]]
    leftElbowRotStart = (R.from_euler('ZYX', leftElbowRotStart, degrees=True)).as_quat()
    leftElbowRotEnd = [frameDataNext[51], frameDataNext[52], frameDataNext[53]]
    leftElbowRotEnd = (R.from_euler('ZYX', leftElbowRotEnd, degrees=True)).as_quat()
    leftElbowRot = p.getQuaternionSlerp(leftElbowRotStart, leftElbowRotEnd, frameFraction)

    leftWristRotStart = [frameData[54], frameData[55], frameData[56]]
    leftWristRotStart = (R.from_euler('ZYX', leftWristRotStart, degrees=True)).as_quat()
    leftWristRotEnd = [frameDataNext[54], frameDataNext[55], frameDataNext[56]]
    leftWristRotEnd = (R.from_euler('ZYX', leftWristRotEnd, degrees=True)).as_quat()
    leftWristRot = p.getQuaternionSlerp(leftWristRotStart, leftWristRotEnd, frameFraction)

    rightShoulderRotStart = [frameData[57], frameData[58], frameData[59]]
    rightShoulderRotStart = (R.from_euler('ZYX', rightShoulderRotStart, degrees=True)).as_quat()
    rightShoulderRotEnd = [frameDataNext[57], frameDataNext[58], frameDataNext[59]]
    rightShoulderRotEnd = (R.from_euler('ZYX', rightShoulderRotEnd, degrees=True)).as_quat()
    rightShoulderRot = p.getQuaternionSlerp(rightShoulderRotStart, rightShoulderRotEnd, frameFraction)

    rightArmRotStart = [frameData[60], frameData[61], frameData[62]]
    rightArmRotStart = (R.from_euler('ZYX', rightArmRotStart, degrees=True)).as_quat()
    rightArmRotEnd = [frameDataNext[60], frameDataNext[61], frameDataNext[62]]
    rightArmRotEnd = (R.from_euler('ZYX', rightArmRotEnd, degrees=True)).as_quat()
    rightArmRot = p.getQuaternionSlerp(rightArmRotStart, rightArmRotEnd, frameFraction)

    rightElbowRotStart = [frameData[63], frameData[64], frameData[65]]
    rightElbowRotStart = (R.from_euler('ZYX', rightElbowRotStart, degrees=True)).as_quat()
    rightElbowRotEnd = [frameDataNext[63], frameDataNext[64], frameDataNext[65]]
    rightElbowRotEnd = (R.from_euler('ZYX', rightElbowRotEnd, degrees=True)).as_quat()
    rightElbowRot = p.getQuaternionSlerp(rightElbowRotStart, rightElbowRotEnd, frameFraction)

    rightWristRotStart = [frameData[66], frameData[67], frameData[68]]
    rightWristRotStart = (R.from_euler('ZYX', rightWristRotStart, degrees=True)).as_quat()
    rightWristRotEnd = [frameDataNext[66], frameDataNext[67], frameDataNext[68]]
    rightWristRotEnd = (R.from_euler('ZYX', rightWristRotEnd, degrees=True)).as_quat()
    rightWristRot = p.getQuaternionSlerp(rightWristRotStart, rightWristRotEnd, frameFraction)

    if (once):  #if (once):
       pass
        # p.resetBasePositionAndOrientation(humanoid, basePos1, baseOrn1)
        # p.resetJointStateMultiDof(humanoid, left_hip, leftHipRot)
        # p.resetJointStateMultiDof(humanoid, left_knee, leftKneeRot)
        # p.resetJointStateMultiDof(humanoid, left_ankle, leftAnkleRot)
        # p.resetJointStateMultiDof(humanoid, left_toe, leftToeRot)

        # p.resetJointStateMultiDof(humanoid, right_hip, rightHipRot) #RightUpLeg
        # p.resetJointStateMultiDof(humanoid, right_knee, rightKneeRot) #RightLeg
        # p.resetJointStateMultiDof(humanoid, right_ankle, rightAnkleRot) #RightFoot
        # p.resetJointStateMultiDof(humanoid, right_toe, rightToeRot) #RightToe

        # p.resetJointStateMultiDof(humanoid, spine, spineRot)
        # p.resetJointStateMultiDof(humanoid, spine1, spine1Rot)
        # p.resetJointStateMultiDof(humanoid, spine2, spine2Rot)
        # p.resetJointStateMultiDof(humanoid, neck, neckRot)
        # p.resetJointStateMultiDof(humanoid, head, headRot)
        
        # p.resetJointStateMultiDof(humanoid, left_shoulder, leftShoulderRot)
        # p.resetJointStateMultiDof(humanoid, left_arm, leftArmRot)
        # p.resetJointStateMultiDof(humanoid, left_elbow, leftElbowRot)
        # p.resetJointStateMultiDof(humanoid, left_wrist, leftWristRot)

        # p.resetJointStateMultiDof(humanoid, right_shoulder, rightShoulderRot)
        # p.resetJointStateMultiDof(humanoid, right_arm, rightArmRot)
        # p.resetJointStateMultiDof(humanoid, right_elbow, rightElbowRot)
        # p.resetJointStateMultiDof(humanoid, right_wrist, rightWristRot)
        # time.sleep(1)
    once = False



    #                              COMPUTE PD torques required to move each joint


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


    targetPositions = [
        leftHipRot,
        leftKneeRot,
        leftAnkleRot,
        leftToeRot,
        rightHipRot,
        rightKneeRot,
        rightAnkleRot,
        rightToeRot,
        spineRot,
        spine1Rot,
        spine2Rot,
        neckRot,
        headRot,
        leftShoulderRot,
        leftArmRot,
        leftElbowRot,
        leftWristRot,
        rightShoulderRot,
        rightArmRot,
        rightElbowRot,
        rightWristRot
    ]



    #                             Configures joint motor control
    kps = [1.0]*21
    # kds = [1]*21

    maxForce = 1000
    maxForces = [[maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
                 [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
                 [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],

                 [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
                 [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
                 [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
                 
                 [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce], [maxForce,maxForce,maxForce],
                 ]
    

    p.setJointMotorControlMultiDofArray(humanoid,
                                indices,
                                p.POSITION_CONTROL, # applly stable PD controller to each joint.
                                targetPositions=targetPositions, # PD target for each joint
                                positionGains=kps, # specifies for each joint how strongly it will try to follow these desired positions
                                # velocityGains=kds, # (control overshooting or oscillating)
                                forces=maxForces # maximum allowable force the PD controller
                                )
    # bp,_ = p.getBasePositionAndOrientation(humanoid)
    # print(bp)
    # time.sleep(0.05)

    # # print("ORN         ", baseOrn , "\n POS", basePos)
    # p.resetJointStateMultiDof(humanoid, left_hip, leftHipRot)
    # # print("LLLLLLLLL",leftHipRotStart)
    # p.resetJointStateMultiDof(humanoid, left_knee, leftKneeRot)
    # p.resetJointStateMultiDof(humanoid, left_ankle, leftAnkleRot)
    # p.resetJointStateMultiDof(humanoid, left_toe, leftToeRot)

    # p.resetJointStateMultiDof(humanoid, right_hip, rightHipRot) #RightUpLeg
    # # print(rightHipRotStart)
    # p.resetJointStateMultiDof(humanoid, right_knee, rightKneeRot) #RightLeg
    # p.resetJointStateMultiDof(humanoid, right_ankle, rightAnkleRot) #RightFoot
    # p.resetJointStateMultiDof(humanoid, right_toe, rightToeRot) #RightToe

    # p.resetJointStateMultiDof(humanoid, spine, spineRot)
    # p.resetJointStateMultiDof(humanoid, spine1, spine1Rot)
    # p.resetJointStateMultiDof(humanoid, spine2, spine2Rot)
    # p.resetJointStateMultiDof(humanoid, neck, neckRot)
    # p.resetJointStateMultiDof(humanoid, head, headRot)
    
    # p.resetJointStateMultiDof(humanoid, left_shoulder, leftShoulderRot)
    # p.resetJointStateMultiDof(humanoid, left_arm, leftArmRot)
    # p.resetJointStateMultiDof(humanoid, left_elbow, leftElbowRot)
    # p.resetJointStateMultiDof(humanoid, left_wrist, leftWristRot)

    # p.resetJointStateMultiDof(humanoid, right_shoulder, rightShoulderRot)
    # p.resetJointStateMultiDof(humanoid, right_arm, rightArmRot)
    # p.resetJointStateMultiDof(humanoid, right_elbow, rightElbowRot)
    # p.resetJointStateMultiDof(humanoid, right_wrist, rightWristRot)
    
    
    # state = get_state(humanoid)
    # print("TIME: {:.6f} seconds".format(time.time() - start))
    # p.stepSimulation()
    # start = time.time()
    # time.sleep(1. / 240.) 

    # print("\n",state[0])
    # print("\n",state[1])
    # print("\n",state[2])
    # print("\n",state[3])
    # time.sleep(1)
    # s = get_rot_vel()
    # print(s)
    #*-----
    # newTime = time.time()
    # frameTime = newTime - currentTime
    # currentTime = newTime
    # accumulator += frameTime
    # # print(accumulator)
    # while ( accumulator >= dt ):
    #     p.stepSimulation() # ( state, t, dt )
    #     accumulator -= dt
    #     t += dt
    #     # print('-', accumulator)
    #     i+=dt/(1./30.)
    #     if i >= numFrames:
    #         p.setTimeOut(1000)
    #         i=0
    #*-----
# Step simulation for 10 seconds 
        
    p.stepSimulation()
    # time.sleep(frameTime)
    i+=dt/(1/30)
    if i >= numFrames:
        p.setTimeOut(1000)
        i=0
    time.sleep(.01)
    # time.sleep(10000)
# p.disconnect()