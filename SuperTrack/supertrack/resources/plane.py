
import pybullet as p
import os

class Plane:
    def __init__(self, client):
        self.p0 = client
        f_name = os.path.join(os.path.dirname(__file__), 
                              'simpleplane.urdf')
        t_name = os.path.join(os.path.dirname(__file__), 
                        'cross-stitch-pattern2.jpg')
        self.plane = self.p0.loadURDF(fileName=f_name,
                   basePosition=[0, -0.01, 0],
                   baseOrientation= p.getQuaternionFromEuler([1.57,0,0]),
                   useFixedBase=False)
        texture = self.p0.loadTexture(t_name)
        self.p0.changeVisualShape(self.plane,linkIndex=-1,textureUniqueId=texture)
        # self.p0.getCameraImage(300,300,renderer=p.ER_TINY_RENDERER)