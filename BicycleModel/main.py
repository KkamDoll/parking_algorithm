import numpy as np
import heapq

class BicycleModel:
    def __init__(self, x=0.0, y = 0.0, theta=0.0, v=0.0,
                 L=2.7, dt=0.1, max_steer=np.radians(35)):
        
        self.x = x                  # 위치 x [m]
        self.y = y                  # 위치 y [m]
        self.theta = theta          # 방향(yaw) [rad]
        self.v = v                  # 속도(*음수면 후진) [m/s]
        self.L = L                  # 휠베이스 [m]
        self.dt = dt                # 시간간격 [s]
        self.max_steer = max_steer

    def step(self, a, delta):
        # 실제 차와 비슷하도록 조향각 한계 적용
        delta = float(np.clip(delta, -self.max_steer, self.max_steer))

        # 자전거 모델 한 스텝 적분(오일러)
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt

        self.theta += self.v / self.L * np.tan(delta) * self.dt
        self.v += a * self.dt

        # 각도를 (-pi ~ +pi) 로 정규화
        self.theta = (self.teta + np.pi) & (2*np.pi) - np.pi
        return self.x, self.y, self.theta, self.v
    
    def is_collision(x, y, theta, obstacles): # 충돌 검사 함수
        """obstacles : [(ox, oy, oradius), ...]"""

        def _car_circles(x, y, theta, L=2.7, n=3, r=1.3):
            """차체를 뒤축 기준 전방으로 n개의 원으로 근사"""
            centers = []
            for i in range(n):
                d = (i + 0.5) * (L / n) + 0.3 # 뒷축에서 앞쪽으로 배치
                cx = x + d * np.cos(theta)
                cy = y + d * np.sin(theta)
                centers.append((cx, cy))
            return centers, r
        
        centers, r = _car_circles(x, y, theta)
        for (cx, cy) in centers:
            for (ox, oy, orad) in obstacles:
                if np.hypot(cx - ox, cy - oy) <= r + orad:
                    return True     # 충돌
        return False                # 비충돌
    
