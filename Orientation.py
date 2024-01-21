import numpy as np


class Quaternion(np.ndarray):
    """
    Quaternion class
    ----------------
    Quaternion representation similar to AHRS libraries, but customized
    and simplified for use in EPPL CubeSat Attitude Control Platform. 
        
    Much deserved credit to the source material:
        
    https://ahrs.readthedocs.io/en/latest/quaternion/quaternion.html

    Parameters
    ----------
    q : array-like, default: None
        Array passed into class will be used to build the quaternion object.
        Quaternion objects will have applicable math operations overridden 
        to quaternion operations.
    DCM : array-like
        Array representing a Direction Cosine Matrix. Passing a 3x3 rotation
        matrix into the class will automatically convert it into a quaternion
        object.
    RPY : array-like
        Array representing the Roll-Pitch-Yaw angles. Passing a 1x3 array 
        into the class will automatically convert it into a quaternion
        object.
    313 : array-like
        Array representing a 313 euler rotation. Passing a 1x3 euler rotation
        array into the class will automatically convert it into a quaternion 
        object.

    Attributes
    ----------

    Q : numpy.ndarray
        Array of quaternion elements [w, x, y, z]

    """
    def __new__(cls, q: np.ndarray=None, **kwargs):
        if 'DCM' in kwargs:
            q = cls.DCM_convert(kwargs['DCM'])
        if 'RPY' in kwargs:
            q = cls.RPY_convert(kwargs['RPY'])
        if 'euler' in kwargs:
            q = cls.euler_convert(kwargs['euler'])
        
        q = np.array(q, dtype=float)

        obj = super(Quaternion, cls).__new__(cls, q.shape, float, q)
        obj.Q = q
        return obj
    
    @property
    def w(self):
        return self.Q[0]
    
    @property
    def x(self):
        return self.Q[1]
    
    @property
    def y(self):
        return self.Q[2]

    @property
    def z(self):
        return self.Q[3]

    def product(self, p:np.ndarray):
        
        pw, px, py, pz = p
        product = np.array([
            self.w * pw - self.x * px - self.y * py - self.z * pz,
            self.w * px + self.x * pw + self.y * pz - self.z * py,
            self.w * py + self.y * pw + self.z * px - self.x * pz,
            self.w * pz + self.z * pw + self.x * py - self.y * px
        ])
        
        return product
    
    def DCM_convert(dcm):
        q = np.empty(4)
        q[0] = 0.5 * np.sqrt(dcm[0,0] + dcm[1,1] + dcm[2,2] + 1)
        q[1] = (dcm[2,1] - dcm[1,2]) / (4 * q[0])
        q[2] = (dcm[0,2] - dcm[2,0]) / (4 * q[0])
        q[3] = (dcm[1,0] - dcm[0,1]) / (4 * q[0])
        return q
        
    def to_DCM(self):
        return np.array([
            [1.0 - 2.0 * (self.y ** 2 + self.z ** 2), 2.0 * (self.x * self. y - self.w * self.z), 2.0 * (self.x * self.z + self.w * self.y)],
            [2.0 * (self.x * self.y + self.w * self.z), 1.0 - 2.0 * (self.x ** 2 + self.z ** 2), 2.0 * (self.y * self.z - self.w * self.x)],
            [2.0 * (self.x * self.z - self.w * self.y), 2.0 * (self.w * self.x + self.y * self.z), 1.0 - 2.0 * (self.x ** 2 + self.y ** 2)]
        ])
    
    # def euler_convert(RPY):
    #     q = np.empty(4)
    #     q[0] = xxx
    #     q[1] = xxx
    #     q[2] = xxx
    #     q[3] = xxx
    #     return q
    
    def to_euler(self):
        phi = np.arctan2(2.0*(self.w*self.x + self.y*self.z), 1.0 - 2.0*(self.x**2 + self.y**2))
        theta = np.arcsin(2.0*(self.w*self.y - self.z*self.x))
        psi = np.arctan2(2.0*(self.w*self.z + self.x*self.y), 1.0 - 2.0*(self.y**2 + self.z**2))
        return np.array([phi, theta, psi])

    def ode(self, w: np.ndarray):
        F = np.array([
            [0.0, -w[0], -w[1], -w[2]],
            [w[0], 0.0, -w[2], w[1]],
            [w[1], w[2], 0.0, -w[0]],
            [w[2], -w[1], w[0], 0.0]
        ])
        return 0.5 * F @ self.Q
