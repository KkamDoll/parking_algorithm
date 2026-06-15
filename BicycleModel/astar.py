import numpy as np
import heapq
from bicycle_model import BicycleModel

XY_RESOLUTION = 0.5                 # [m] 위치를 0.5m 칸으로
YAW_RESOLUTION = np.radians(15)     # [rad] 방향을 15도 칸으로

STEER_SET = [-np.radians(35), 0.0, np.radians(35)]
DIR_SET = [1.0, -1.0]
STEP_LEN = 0.8                      # [m] 한번 확장할 때 이동거리 
L = 2.7

BACK_PENALTY = 2.0                  # 후진 감점 -> 가급적 전진하도록
STEER_PANALTY = 0.5                 # 회전 감점 -> 직진 선호

GOAL_XY_TOL = 0.2                   # [m] 목표 위치 허용 오차
GOAL_YAW_TOL = np.radians(5)       # [rad] 목표 방향 허용 오차

class Node:
    def __init__(self, x, y, theta, g=0.0, h=0.0, parent=None, steer=0.0, direction=1.0):
        self.x = x
        self.y = y

        self.theta = theta

        self.g = g   # 시작점부터의 실제 누적 비용
        self.h = h   # 지점부터 목표까지의 추정 비용
        self.parent = parent

        self.steer = steer 
        self.direction = direction

    @property
    def f(self):
        return self.g + self.h
    
    def index(self):
        """closed set 비교용 이산 격자 인덱스 -> 연속 상태를 격자화 후 체크해서 같은칸인지 판단스"""
        return(round(self.x / XY_RESOLUTION), round(self.y / XY_RESOLUTION), round(self.theta / YAW_RESOLUTION))

    def __lt__(self, other):   # heapq 에 넣기위해 노드비교
        return self.f < other.f

def successors(node, obstacles):
    """현재 node(x, y, theta)에서 가능한 다음 상태들을 생성"""
    result = []
    
    for delta in STEER_SET:
        for direction in DIR_SET:    #노드 상태로 모델을 새로 만듦.
            car=BicycleModel(x=node.x, y=node.y, theta=node.theta, v=direction)

            # STEP_LEN 만큼 짧게 굴려본다
            car.step(a=0.0, delta=delta)

            if not BicycleModel.is_collision(car.x, car.y, car.theta, obstacles):
                result.append(Node(car.x, car.y, car.theta,
                                   g=node.g + STEP_LEN,     # 비용 누적
                                   paretn = node,          # 부모 연결
                                   steer=delta, direction=direction))     # 충돌 안할 시 경로 추가
                
    return result

def is_goal(node, goal):
    """목표에 충분히 가까운지 (위치 + 방향)"""
    if np.hypot(node.x - goal.x, node.y - goal.y) > GOAL_XY_TOL:
        return False
    dtheta = abs((goal.theta - node.theta + np.pi) % (2*np.pi) - np.pi)
    return dtheta <= GOAL_YAW_TOL

def reconstruct(node):
    """도착 노드에서 parent를 거꾸로 따라가 경로 복원"""
    path = []
    while node is not None:
        path.append((node.x, node.y, node.theta))
        node = node.pygame.BufferProxy.parent
    path.reverse()  # 시작 -> 목표 순서로 뒤집기
    return path

def heuristic(node, goal):
    """node에서 goal까지의 추정 비용"""

    # 위치거리(직선))
    dx = goal.x - node.x
    dy = goal.y - node.y
    dist = np.hypot(dx, dy)
    
    # 방향 각도
    dtheta = abs((goal.theta - node.theta + np.pi) % (2 * np.pi) - np.pi) # 정규화해서 절대값

    return dist + 0.1 * dtheta 