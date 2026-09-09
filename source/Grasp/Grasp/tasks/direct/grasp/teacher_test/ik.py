from __future__ import annotations

from math import acos, asin, atan2, cos, pi, sin, sqrt

import numpy as np

def inv_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return np.vstack((np.hstack((rotation.T, -rotation.T @ translation[:, None])), [0.0, 0.0, 0.0, 1.0]))

def transform_dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    return np.array([[cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
                     [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                     [0.0, sin(alpha), cos(alpha), d], [0.0, 0.0, 0.0, 1.0]])

def transform_robot_parameter(theta: np.ndarray) -> np.ndarray:
    d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
    a = [0, -0.425, -0.39225, 0, 0, 0]
    alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
    result = np.eye(4)
    for i in range(6):
        result = result @ transform_dh(a[i], d[i], alpha[i], theta[i])
    return result

class InverseKinematicsUR5:
    def __init__(self):
        self.d = [0.089159, 0, 0, 0.10915, 0.09465, 0.0823]
        self.a = [0, -0.425, -0.39225, 0, 0, 0]
        self.alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]
        self.limit_min, self.limit_max = -2*pi, 2*pi
        self.joint_weights = np.ones(6)
        self.ee_offset = np.eye(4)
        self.debug = False

    def setJointLimits(self, limit_min: float, limit_max: float):
        self.limit_min, self.limit_max = limit_min, limit_max

    def setJointWeights(self, weights):
        self.joint_weights = np.asarray(weights)

    def normalize(self, value):
        while value > self.limit_max: value -= 2*pi
        while value < self.limit_min: value += 2*pi
        return value

    @staticmethod
    def _valid(numerator, denominator):
        return denominator != 0 and abs(numerator / denominator) < 1.01

    def solveIK(self, forward_kinematics):
        gd = forward_kinematics @ self.ee_offset
        p05 = gd @ np.array([0, 0, -self.d[5], 1]) - np.array([0, 0, 0, 1])
        psi = atan2(p05[1], p05[0]); length = sqrt(p05[0]**2 + p05[1]**2)
        valid1 = self._valid(self.d[3], length)
        length = max(abs(self.d[3]), length)
        theta1 = [self.normalize(psi + acos(self.d[3]/length) + pi/2), self.normalize(psi - acos(self.d[3]/length) + pi/2)]
        theta5 = np.zeros((2, 2)); valid5 = np.ones((2, 2), dtype=bool)
        for i in range(2):
            p16z = gd[0, 3]*sin(theta1[i]) - gd[1, 3]*cos(theta1[i])
            ratio = (p16z - self.d[3]) / self.d[5]
            valid5[i, :] = self._valid(p16z - self.d[3], self.d[5])
            theta5[i] = [acos(np.clip(ratio, -1, 1)), -acos(np.clip(ratio, -1, 1))]
        theta6 = np.zeros((2, 2))
        for i in range(2):
            t1 = transform_dh(self.a[0], self.d[0], self.alpha[0], theta1[i])
            t61 = inv_transform(inv_transform(t1) @ gd)
            for j in range(2):
                theta6[i, j] = 0 if sin(theta5[i, j]) == 0 else atan2(-t61[1, 2]/sin(theta5[i, j]), t61[0, 2]/sin(theta5[i, j]))
        theta2 = np.zeros((2, 2, 2)); theta3 = np.zeros((2, 2, 2)); theta4 = np.zeros((2, 2, 2)); valid3 = np.ones((2, 2, 2), dtype=bool)
        for i in range(2):
            t1 = transform_dh(self.a[0], self.d[0], self.alpha[0], theta1[i])
            t16 = inv_transform(t1) @ gd
            for j in range(2):
                t45 = transform_dh(self.a[4], self.d[4], self.alpha[4], theta5[i, j])
                t56 = transform_dh(self.a[5], self.d[5], self.alpha[5], theta6[i, j])
                t14 = t16 @ inv_transform(t45 @ t56)
                p13 = t14 @ np.array([0, -self.d[3], 0, 1]) - np.array([0, 0, 0, 1])
                value = (p13 @ p13.T - self.a[1]**2 - self.a[2]**2) / (2*self.a[1]*self.a[2])
                valid3[i, j, :] = self._valid(value, 1.0)
                value = np.clip(value, -1.0, 1.0)
                theta3[i, j, 0] = acos(value); theta3[i, j, 1] = -theta3[i, j, 0]
                for k in range(2):
                    theta2[i, j, k] = -atan2(p13[1], -p13[0]) + asin(np.clip(self.a[2]*sin(theta3[i,j,k])/np.linalg.norm(p13), -1, 1))
                for k in range(2):
                    t13 = transform_dh(self.a[1], self.d[1], self.alpha[1], theta2[i,j,k]) @ transform_dh(self.a[2], self.d[2], self.alpha[2], theta3[i,j,k])
                    t34 = inv_transform(t13) @ t14
                    theta4[i,j,k] = atan2(t34[1,0], t34[0,0])
        solutions = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if valid1 and valid5[i,j] and valid3[i,j,k]:
                        q = [theta1[i], theta2[i,j,k], theta3[i,j,k], theta4[i,j,k], theta5[i,j], theta6[i,j]]
                        solutions.append([self.normalize(v) for v in q])
        return np.asarray(solutions) if solutions else None

    def findClosestIK(self, forward_kinematics, current_joint_configuration):
        solutions = self.solveIK(forward_kinematics)
        if solutions is None: return None
        distances = np.sum(np.abs(solutions - np.asarray(current_joint_configuration)) * self.joint_weights, axis=1)
        return solutions[np.argmin(distances)]

__all__ = ["InverseKinematicsUR5", "transform_robot_parameter"]
