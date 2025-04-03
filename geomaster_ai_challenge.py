from collections import deque
import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import threading
import time
import matplotlib.pyplot as plt
from io import BytesIO

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GeoMaster AI Challenge - Next Shape Predictor")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)

# Font
font = pygame.font.SysFont("monospace", 20)


# Helper function
def to_int_point(point):
    if (
        not isinstance(point, (tuple, list))
        or len(point) != 2
        or not all(isinstance(coord, (int, float)) for coord in point)
    ):
        print(f"Invalid point detected: {point}")
        return (0, 0)
    return (int(point[0]), int(point[1]))


# Geometric Worlds and Tasks
WORLD_EUCLIDEAN = "Euclidean"
WORLD_SPHERICAL = "Spherical"
WORLD_HYPERBOLIC = "Hyperbolic"
WORLD_ELLIPTICAL = "Elliptical"
WORLD_PROJECTIVE = "Projective"
WORLD_FRACTAL = "Fractal"
worlds = [
    WORLD_EUCLIDEAN,
    WORLD_SPHERICAL,
    WORLD_HYPERBOLIC,
    WORLD_ELLIPTICAL,
    WORLD_PROJECTIVE,
    WORLD_FRACTAL,
]
current_world = WORLD_EUCLIDEAN

TASK_LINE = "Draw Line"
TASK_TRIANGLE = "Draw Triangle"
TASK_CIRCLE = "Draw Circle"
TASK_PENTAGON = "Draw Pentagon"
TASK_TESSELLATION = "Draw Tessellation"
tasks = [TASK_LINE, TASK_TRIANGLE, TASK_CIRCLE, TASK_PENTAGON, TASK_TESSELLATION]
current_task = TASK_LINE


# --- SumTree for Prioritized Experience Replay ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if v <= self.tree[left]:
                parent_idx = left
            else:
                v -= self.tree[left]
                parent_idx = right
        data_idx = parent_idx - self.capacity + 1
        return parent_idx, self.tree[parent_idx], self.data[data_idx]


# Neural Networks
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ForwardModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + 1, 128)  # Action is a scalar
        self.fc2 = nn.Linear(128, state_dim)

    def forward(self, state, action):
        try:
            if state.dim() != 2 or action.dim() != 2:
                raise ValueError(
                    f"Expected 2D tensors, got state shape {state.shape}, action shape {action.shape}"
                )
            if state.shape[1] != state_dim or action.shape[1] != 1:
                raise ValueError(
                    f"Unexpected feature dimensions: state {state.shape}, action {action.shape}"
                )
            action = action.view(-1, 1)
            x = torch.cat([state, action], dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
        except Exception as e:
            print(f"Error in ForwardModel forward: {e}")
            raise


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + 1, 128)
        self.fc2 = nn.Linear(128, state_dim)

    def forward(self, state, action):
        try:
            if state.dim() != 2 or action.dim() != 2:
                raise ValueError(
                    f"Expected 2D tensors, got state shape {state.shape}, action shape {action.shape}"
                )
            if state.shape[1] != state_dim or action.shape[1] != 1:
                raise ValueError(
                    f"Unexpected feature dimensions: state {state.shape}, action {action.shape}"
                )
            action = action.view(-1, 1)
            x = torch.cat([state, action], dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
        except Exception as e:
            print(f"Error in WorldModel forward: {e}")
            raise


class NextShapePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NextShapePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# DQN Parameters
state_dim = 10
action_dim = 6
predictor_input_dim = 13  # task_id, world_id, avg_reward, last 10 rewards
predictor_output_dim = len(tasks)
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 5000
epsilon = epsilon_start
batch_size = 64
memory_size = 10000
target_update = 10
n_step = 3
alpha = 0.7
beta = 0.5
base_lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=base_lr)

forward_model = ForwardModel(state_dim, action_dim).to(device)
optimizer_fm = optim.Adam(forward_model.parameters(), lr=base_lr)

world_model = WorldModel(state_dim, action_dim).to(device)
optimizer_wm = optim.Adam(world_model.parameters(), lr=base_lr)

shape_predictor = NextShapePredictor(predictor_input_dim, predictor_output_dim).to(
    device
)
optimizer_sp = optim.Adam(shape_predictor.parameters(), lr=0.001)

memory = SumTree(memory_size)
n_step_buffer = []

# Game State
start_point = None
end_point = (WIDTH - 50, HEIGHT - 50)
triangle_points = []
circle_center = None
circle_radius = 50
pentagon_points = []
tessellation_points = []
current_shape = []
ai_step = 0
max_steps = 50
game_state = "waiting"
running_state = "stopped"
rewards = deque(maxlen=100)
episode_rewards = deque(maxlen=100)
reward_history = deque(maxlen=10)
current_reward = 0
global_step = 0
episode_count = 0
tick_rate = 30
train_iters = 1

# Logging
log_file = "ai_calculations.csv"
file_exists = os.path.isfile(log_file)
with open(log_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(
            [
                "Episode",
                "Step",
                "State",
                "Action",
                "Reward",
                "Average Reward",
                "Learning Rate",
                "FPS",
                "Next Shape",
            ]
        )


# Geometric Projections
def stereographic_projection(point, R=100):
    x, y, z = point
    if z == 2 * R:
        return None
    factor = 2 * R / (2 * R - z)
    proj_x = factor * x + WIDTH / 2
    proj_y = factor * y + HEIGHT / 2
    return (proj_x, proj_y)


def poincare_disk_projection(point, R=100):
    x, y = point
    dist = np.sqrt(x**2 + y**2)
    if dist >= R:
        return None
    factor = R * np.tanh(dist / R)
    angle = np.arctan2(y, x)
    proj_x = factor * np.cos(angle) + WIDTH / 2
    proj_y = factor * np.sin(angle) + HEIGHT / 2
    return (proj_x, proj_y)


def elliptical_projection(point, a=150, b=100):
    x, y = point
    proj_x = (x / a) * (WIDTH / 2) + WIDTH / 2
    proj_y = (y / b) * (HEIGHT / 2) + HEIGHT / 2
    return (proj_x, proj_y)


def projective_projection(point, fov=90, near=1):
    x, y = point
    z = 100
    if z <= 0:
        return None
    proj_x = (x * near / z) * (WIDTH / 2) + WIDTH / 2
    proj_y = (y * near / z) * (HEIGHT / 2) + HEIGHT / 2
    return (proj_x, proj_y)


def fractal_projection(point, iterations=3):
    x, y = point
    x, y = x / WIDTH, y / HEIGHT
    for _ in range(iterations):
        r = np.random.randint(3)
        if r == 0:
            x, y = x / 2, y / 2
        elif r == 1:
            x, y = (x + 1) / 2, y / 2
        else:
            x, y = x / 2, (y + 1) / 2
    return (x * WIDTH, y * HEIGHT)


def project_point(point, world):
    x, y = point
    if world == WORLD_EUCLIDEAN:
        return (x + WIDTH / 2, y + HEIGHT / 2)
    elif world == WORLD_SPHERICAL:
        z = np.sqrt(max(0, 100**2 - x**2 - y**2))
        return stereographic_projection((x, y, z))
    elif world == WORLD_HYPERBOLIC:
        return poincare_disk_projection((x - WIDTH / 2, y - HEIGHT / 2))
    elif world == WORLD_ELLIPTICAL:
        return elliptical_projection((x, y))
    elif world == WORLD_PROJECTIVE:
        return projective_projection((x, y))
    elif world == WORLD_FRACTAL:
        return fractal_projection((x, y))
    return (x, y)


def hyperbolic_geodesic(start, end, R=100):
    start = (start[0] - WIDTH / 2, start[1] - HEIGHT / 2)
    end = (end[0] - WIDTH / 2, end[1] - HEIGHT / 2)
    start_r = np.sqrt(start[0] ** 2 + start[1] ** 2)
    end_r = np.sqrt(end[0] ** 2 + end[1] ** 2)
    if start_r >= R or end_r >= R:
        return []
    start_angle = np.arctan2(start[1], start[0])
    end_angle = np.arctan2(end[1], end[0])
    points = []
    num_points = 20
    for t in np.linspace(0, 1, num_points):
        angle = (1 - t) * start_angle + t * end_angle
        r = R * np.tanh((1 - t) * np.arctanh(start_r / R) + t * np.arctanh(end_r / R))
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        proj = project_point((x, y), WORLD_HYPERBOLIC)
        if proj:
            points.append(proj)
    return points


# Wireframe Visualization
animation_time = 0


def draw_wireframe(world):
    global animation_time
    animation_time += 0.05
    pulse = 0.5 + 0.5 * np.sin(animation_time)

    def gradient_color(t):
        r = int(0 * (1 - t) + 0 * t)
        g = int(0 * (1 - t) + 255 * t)
        b = int(255 * (1 - t) + 255 * t)
        return (r, g, b)

    if world == WORLD_EUCLIDEAN:
        for i, x in enumerate(range(-200, 201, 40)):
            t = i / 10
            color = gradient_color(t)
            start = project_point((x, -200), world)
            end = project_point((x, 200), world)
            if start and end:
                pygame.draw.line(
                    screen,
                    color,
                    to_int_point(start),
                    to_int_point(end),
                    int(1 + pulse),
                )
    elif world == WORLD_SPHERICAL:
        for i, theta in enumerate(np.linspace(0, 2 * np.pi, 20)):
            points = []
            t = i / 20
            color = gradient_color(t)
            for phi in np.linspace(0, np.pi, 20):
                x = 100 * np.sin(phi) * np.cos(theta)
                y = 100 * np.sin(phi) * np.sin(theta)
                z = 100 * np.cos(phi)
                proj = stereographic_projection((x, y, z))
                if proj:
                    points.append(proj)
            for j in range(len(points) - 1):
                pygame.draw.line(
                    screen,
                    color,
                    to_int_point(points[j]),
                    to_int_point(points[j + 1]),
                    int(1 + pulse),
                )
    elif world == WORLD_HYPERBOLIC:
        pygame.draw.circle(
            screen, WHITE, to_int_point((WIDTH // 2, HEIGHT // 2)), 100, int(1 + pulse)
        )
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 12)):
            points = []
            t = i / 12
            color = gradient_color(t)
            for r in np.linspace(0, 100, 20):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                proj = poincare_disk_projection((x, y))
                if proj:
                    points.append(proj)
            for j in range(len(points) - 1):
                pygame.draw.line(
                    screen,
                    color,
                    to_int_point(points[j]),
                    to_int_point(points[j + 1]),
                    int(1 + pulse),
                )
    elif world == WORLD_ELLIPTICAL:
        for i, x in enumerate(range(-200, 201, 40)):
            t = i / 10
            color = gradient_color(t)
            start = project_point((x, -200), world)
            end = project_point((x, 200), world)
            if start and end:
                pygame.draw.line(
                    screen,
                    color,
                    to_int_point(start),
                    to_int_point(end),
                    int(1 + pulse),
                )
    elif world == WORLD_PROJECTIVE:
        for i, x in enumerate(range(-200, 201, 40)):
            t = i / 10
            color = gradient_color(t)
            start = project_point((x, -200), world)
            end = project_point((x, 200), world)
            if start and end:
                pygame.draw.line(
                    screen,
                    color,
                    to_int_point(start),
                    to_int_point(end),
                    int(1 + pulse),
                )
    elif world == WORLD_FRACTAL:
        points = []
        for i in range(100):
            x, y = np.random.uniform(-200, 200), np.random.uniform(-200, 200)
            proj = project_point((x, y), world)
            if proj:
                points.append(proj)
        for j in range(len(points) - 1):
            pygame.draw.line(
                screen,
                CYAN,
                to_int_point(points[j]),
                to_int_point(points[j + 1]),
                int(1 + pulse),
            )


# Drawing Functions
def draw_line_segment(start, end, curvature, world):
    try:
        if not (isinstance(start, (tuple, list)) and isinstance(end, (tuple, list))):
            raise ValueError(f"Invalid start or end point: start={start}, end={end}")
        if len(start) != 2 or len(end) != 2:
            raise ValueError(
                f"Start and end must be 2D points: start={start}, end={end}"
            )
        if world == WORLD_HYPERBOLIC:
            return hyperbolic_geodesic(start, end)
        points = []
        num_points = 10
        for i in range(num_points + 1):
            t = i / num_points
            x = (1 - t) * start[0] + t * end[0]
            y = (1 - t) * start[1] + t * end[1]
            offset = curvature * 50 * np.sin(np.pi * t)
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
                x += offset * perp_x
                y += offset * perp_y
            proj = project_point((x - WIDTH / 2, y - HEIGHT / 2), world)
            if proj:
                points.append(proj)
        return points
    except Exception as e:
        print(f"Error in draw_line_segment: {e}")
        return []


def draw_triangle(points, angle_adjust, world):
    try:
        if len(points) < 3:
            return points
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in points):
            raise ValueError(f"Invalid points for triangle: {points}")
        new_points = points.copy()
        p1, p2, p3 = points
        dx = p3[0] - p2[0]
        dy = p3[1] - p2[1]
        angle = np.arctan2(dy, dx) + angle_adjust * np.pi / 180
        length = np.sqrt(dx**2 + dy**2)
        new_x = p2[0] + length * np.cos(angle)
        new_y = p2[1] + length * np.sin(angle)
        new_points[2] = (new_x, new_y)
        return [
            project_point((p[0] - WIDTH / 2, p[1] - HEIGHT / 2), world)
            for p in new_points
        ]
    except Exception as e:
        print(f"Error in draw_triangle: {e}")
        return points


def draw_circle(center, radius, radius_adjust, world):
    radius = max(10, radius + radius_adjust)
    points = []
    for theta in np.linspace(0, 2 * np.pi, 20):
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        proj = project_point((x - WIDTH / 2, y - HEIGHT / 2), world)
        if proj:
            points.append(proj)
    return points, radius


def draw_pentagon(points, angle_adjust, world):
    if len(points) < 5:
        return points
    new_points = points.copy()
    for i in range(2, 5):
        p1 = new_points[i - 1]
        p2 = new_points[i]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = np.arctan2(dy, dx) + angle_adjust * np.pi / 180
        length = np.sqrt(dx**2 + dy**2)
        new_x = p1[0] + length * np.cos(angle)
        new_y = p1[1] + length * np.sin(angle)
        new_points[i] = (new_x, new_y)
    return [
        project_point((p[0] - WIDTH / 2, p[1] - HEIGHT / 2), world) for p in new_points
    ]


def draw_tessellation(points, world):
    if len(points) < 3:
        return points
    base_triangle = points[:3]
    tessellation = [base_triangle]
    center_x = sum(p[0] for p in base_triangle) / 3
    center_y = sum(p[1] for p in base_triangle) / 3
    for i in range(3):
        p1 = base_triangle[i]
        p2 = base_triangle[(i + 1) % 3]
        new_point = (center_x + (center_x - p1[0]), center_y + (center_y - p1[1]))
        new_triangle = [p1, p2, new_point]
        tessellation.append(new_triangle)
    return [
        [project_point((p[0] - WIDTH / 2, p[1] - HEIGHT / 2), world) for p in triangle]
        for triangle in tessellation
    ]


# Macro Actions
def macro_action_draw_triangle(current_shape, world):
    actions = []
    if len(current_shape) < 3:
        for _ in range(3 - len(current_shape)):
            actions.append(3)
    return actions


def macro_action_draw_circle(center, radius, world):
    actions = []
    for _ in range(3):
        actions.append(4)
    return actions


# Helper Functions
def calculate_triangle_angle(points):
    if len(points) < 3:
        return 0
    p1, p2, p3 = points
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    angle = (
        np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        * 180
        / np.pi
    )
    return angle


def calculate_dist_to_close(points):
    if len(points) < 2:
        return 0
    p1, p_last = points[0], points[-1]
    return np.sqrt((p1[0] - p_last[0]) ** 2 + (p1[1] - p_last[1]) ** 2)


# Reward Function
def calculate_reward(shape, target, task, world):
    try:
        reward = 0
        if task == TASK_LINE:
            if not shape:
                return -10
            current_pos = shape[-1] if shape else (0, 0)
            if not (isinstance(current_pos, (tuple, list)) and len(current_pos) == 2):
                raise ValueError(f"Invalid current_pos: {current_pos}")
            if not (isinstance(target, (tuple, list)) and len(target) == 2):
                raise ValueError(f"Invalid target: {target}")
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            reward = -distance / 100.0
            if distance < 10:
                reward += 50
        elif task == TASK_TRIANGLE:
            if len(shape) != 3:
                return -10
            if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in shape):
                raise ValueError(f"Invalid shape for triangle: {shape}")
            angle = calculate_triangle_angle(shape)
            if world == WORLD_SPHERICAL:
                reward = (angle - 60) / 10 if angle > 60 else -10
            elif world == WORLD_HYPERBOLIC:
                reward = (60 - angle) / 10 if angle < 60 else -10
            else:
                reward = -abs(angle - 60) / 10
                if abs(angle - 60) < 10:
                    reward += 10
        elif task == TASK_CIRCLE:
            if not shape:
                return -10
            if not isinstance(target, (tuple, list)) or len(target) != 2:
                raise ValueError(f"Invalid target for circle: {target}")
            current_radius = target[1]
            ideal_radius = 50
            reward = -abs(current_radius - ideal_radius) / 10
            if abs(current_radius - ideal_radius) < 5:
                reward += 20
        elif task == TASK_PENTAGON:
            if len(shape) != 5:
                return -10
            if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in shape):
                raise ValueError(f"Invalid shape for pentagon: {shape}")
            dist = calculate_dist_to_close(shape)
            reward = -dist / 10
            if dist < 10:
                reward += 30
        elif task == TASK_TESSELLATION:
            if not shape:
                return -10
            num_triangles = len(shape)
            reward = num_triangles * 10
            for triangle in shape:
                if not (
                    isinstance(triangle, (list, tuple))
                    and len(triangle) == 3
                    and all(
                        isinstance(p, (tuple, list)) and len(p) == 2 for p in triangle
                    )
                ):
                    print(f"Invalid triangle in tessellation: {triangle}")
                    continue
                angle = calculate_triangle_angle(triangle)
                if world == WORLD_HYPERBOLIC:
                    if angle < 60:
                        reward += (60 - angle) / 10
                    else:
                        reward -= 5
                elif world == WORLD_SPHERICAL:
                    if angle > 60:
                        reward += (angle - 60) / 10
                    else:
                        reward -= 5
                else:
                    reward -= abs(angle - 60) / 10
                    if abs(angle - 60) < 10:
                        reward += 5
        return reward
    except Exception as e:
        print(f"Error in calculate_reward: {e}")
        return -10


# DQN Training
def optimize_model():
    if memory.n_entries < batch_size:
        return

    total_priority = memory.tree[0]
    batch = []
    idxs = []
    priorities = []
    segment = total_priority / batch_size

    for i in range(batch_size):
        a = segment * i
        b = segment * (i + 1)
        s = random.uniform(a, b)
        idx, priority, data = memory.get_leaf(s)
        batch.append(data)
        idxs.append(idx)
        priorities.append(priority)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(np.array(state_batch)).to(device)
    action_batch = torch.LongTensor(np.array(action_batch)).to(device)
    reward_batch = torch.FloatTensor(np.array(reward_batch)).to(device)
    next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(device)
    done_batch = torch.FloatTensor(np.array(done_batch)).to(device)

    q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

    td_errors = (q_values - expected_q_values).abs().detach().cpu().numpy()
    for i, idx in enumerate(idxs):
        memory.update(idx, (td_errors[i] + 1e-5) ** alpha)

    weights = torch.FloatTensor([(total_priority / p) ** -beta for p in priorities]).to(
        device
    )
    loss = (weights * (q_values - expected_q_values) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    action_batch_reshaped = action_batch.float().unsqueeze(1)
    predicted_next_state = forward_model(state_batch, action_batch_reshaped)
    fm_loss = nn.MSELoss()(predicted_next_state, next_state_batch)
    optimizer_fm.zero_grad()
    fm_loss.backward()
    optimizer_fm.step()


def store_transition(transition):
    n_step_buffer.append(transition)
    if len(n_step_buffer) < n_step:
        return
    cumulative_reward = sum([(gamma**i) * t[2] for i, t in enumerate(n_step_buffer)])
    state, action = n_step_buffer[0][0], n_step_buffer[0][1]
    next_state, done = n_step_buffer[-1][3], n_step_buffer[-1][4]
    memory.add(1.0, (state, action, cumulative_reward, next_state, done))
    n_step_buffer.pop(0)


def plan_actions(state_tensor, model, current_shape, target, task, world, steps=3):
    best_sequence = []
    best_reward = -float("inf")
    state = state_tensor.squeeze(0).cpu().numpy()

    for _ in range(10):
        sequence = [np.random.randint(action_dim) for _ in range(steps)]
        sim_state = state.copy()
        sim_shape = current_shape.copy()
        total_reward = 0

        for action in sequence:
            if task == TASK_LINE:
                curvature = 0
                if action == 0:
                    curvature = -1
                elif action == 1:
                    curvature = 1
                if sim_shape:
                    next_segment = draw_line_segment(
                        sim_shape[-1], target, curvature, world
                    )
                    sim_shape.extend(next_segment[1:])
            elif task == TASK_TRIANGLE:
                angle_adjust = 0
                if action == 2:
                    angle_adjust = -10
                elif action == 3:
                    angle_adjust = 10
                sim_shape = draw_triangle(sim_shape, angle_adjust, world)
            elif task == TASK_CIRCLE:
                radius_adjust = 0
                if action == 4:
                    radius_adjust = 5
                sim_shape, sim_radius = draw_circle(
                    target[0], target[1], radius_adjust, world
                )
                target = (target[0], sim_radius)
            elif task == TASK_PENTAGON:
                angle_adjust = 0
                if action == 2:
                    angle_adjust = -10
                elif action == 3:
                    angle_adjust = 10
                sim_shape = draw_pentagon(sim_shape, angle_adjust, world)
            elif task == TASK_TESSELLATION:
                if action == 5:
                    sim_shape = draw_tessellation(sim_shape, world)

            shape_progress = (
                len(sim_shape) / 5 if task != TASK_TESSELLATION else len(sim_shape)
            )
            num_vertices = (
                len(sim_shape)
                if task != TASK_TESSELLATION
                else sum(len(t) for t in sim_shape)
            )
            angle = 0
            dist_to_close = 0
            if task == TASK_LINE:
                next_pos = sim_shape[-1] if sim_shape else (0, 0)
                dx = target[0] - next_pos[0]
                dy = target[1] - next_pos[1]
            else:
                next_pos = (0, 0)
                dx = dy = 0
                if task == TASK_TRIANGLE and len(sim_shape) == 3:
                    angle = calculate_triangle_angle(sim_shape)
                elif task == TASK_PENTAGON and len(sim_shape) >= 2:
                    dist_to_close = calculate_dist_to_close(sim_shape)
                elif task == TASK_TESSELLATION and sim_shape:
                    angle = calculate_triangle_angle(sim_shape[0])
            task_id = tasks.index(task)
            world_id = worlds.index(world)
            sim_state = np.array(
                [
                    next_pos[0],
                    next_pos[1],
                    dx,
                    dy,
                    task_id,
                    world_id,
                    shape_progress,
                    num_vertices,
                    angle,
                    dist_to_close,
                ]
            )

            reward = calculate_reward(sim_shape, target, task, world)
            total_reward += reward

            sim_state_tensor = torch.FloatTensor(sim_state).unsqueeze(0).to(device)
            action_tensor = (
                torch.tensor([action], dtype=torch.float).unsqueeze(0).to(device)
            )
            sim_state_tensor = model(sim_state_tensor, action_tensor)
            sim_state = sim_state_tensor.detach().squeeze(0).cpu().numpy()

        if total_reward > best_reward:
            best_reward = total_reward
            best_sequence = sequence

    return best_sequence


# Console Input Handler
command = ""


def console_input():
    global command, running_state, tick_rate, train_iters
    while True:
        cmd = (
            input(
                "Enter command (start, pause, resume, quit, faster, slower, iters_up, iters_down): "
            )
            .strip()
            .lower()
        )
        with lock:
            command = cmd
            if cmd == "start" and running_state == "stopped":
                running_state = "running"
            elif cmd == "pause" and running_state == "running":
                running_state = "paused"
            elif cmd == "resume" and running_state == "paused":
                running_state = "running"
            elif cmd == "quit":
                running_state = "stopped"
                pygame.quit()
                os._exit(0)
            elif cmd == "faster":
                tick_rate = min(tick_rate + 10, 120)
                print(f"Tick rate increased to {tick_rate}")
            elif cmd == "slower":
                tick_rate = max(tick_rate - 10, 10)
                print(f"Tick rate decreased to {tick_rate}")
            elif cmd == "iters_up":
                train_iters = min(train_iters + 1, 10)
                print(f"Training iterations increased to {train_iters}")
            elif cmd == "iters_down":
                train_iters = max(train_iters - 1, 1)
                print(f"Training iterations decreased to {train_iters}")


lock = threading.Lock()
threading.Thread(target=console_input, daemon=True).start()


# Adaptive Learning Rate
def adjust_learning_rate(fps):
    global optimizer
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    current_lr = base_lr
    if avg_reward > 10:
        current_lr = min(base_lr * 2, 0.01)
    elif avg_reward < -5:
        current_lr = max(base_lr * 0.5, 0.0001)
    if fps < 20:
        current_lr *= 0.8
    elif fps > 60:
        current_lr *= 1.2
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return current_lr


# Predict Next Shape
def predict_next_shape():
    global current_task, reward_history
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    state = np.array(
        [tasks.index(current_task), worlds.index(current_world), avg_reward]
        + list(reward_history)
        + [0] * (10 - len(reward_history))
    )
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = shape_predictor(state_tensor).argmax().item()
    return tasks[pred]


# Reset Episode
def reset_episode():
    global start_point, triangle_points, circle_center, pentagon_points, tessellation_points, current_shape, game_state, ai_step
    if current_task == TASK_LINE:
        start_point = (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
        current_shape = [start_point]
    elif current_task == TASK_TRIANGLE:
        triangle_points = [
            (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
            for _ in range(3)
        ]
        current_shape = triangle_points.copy()
    elif current_task == TASK_CIRCLE:
        circle_center = (
            random.randint(50, WIDTH - 50),
            random.randint(50, HEIGHT - 50),
        )
        current_shape = []
    elif current_task == TASK_PENTAGON:
        pentagon_points = [
            (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
            for _ in range(5)
        ]
        current_shape = pentagon_points.copy()
    elif current_task == TASK_TESSELLATION:
        tessellation_points = [
            (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
            for _ in range(3)
        ]
        current_shape = [tessellation_points]
    game_state = "ai_drawing"
    ai_step = 0


# Matplotlib Plot
plot_surface = None


def update_reward_plot():
    global plot_surface, episode_rewards
    plt.figure(figsize=(4, 2))
    plt.plot(list(episode_rewards), label="Reward", color="cyan")
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.title("Reward Trend")
    plt.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = pygame.image.load(buf)
    plt.close()
    buf.close()
    return pygame.transform.scale(image, (300, 150))


# Main Game Loop
clock = pygame.time.Clock()
running = True
last_plot_update = 0
step_counter = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    with lock:
        if command == "quit":
            running = False

    if running_state == "running" and game_state == "waiting":
        reset_episode()

    if running_state == "running" and game_state == "ai_drawing":
        shape_progress = (
            len(current_shape) / 5
            if current_task != TASK_TESSELLATION
            else len(current_shape)
        )
        num_vertices = (
            len(current_shape)
            if current_task != TASK_TESSELLATION
            else sum(len(t) for t in current_shape)
        )
        angle = 0
        dist_to_close = 0
        if current_task == TASK_LINE:
            current_pos = current_shape[-1] if current_shape else (0, 0)
            dx = end_point[0] - current_pos[0]
            dy = end_point[1] - current_pos[1]
        else:
            current_pos = (0, 0)
            dx = dy = 0
            if current_task == TASK_TRIANGLE and len(current_shape) == 3:
                angle = calculate_triangle_angle(current_shape)
            elif current_task == TASK_PENTAGON and len(current_shape) >= 2:
                dist_to_close = calculate_dist_to_close(current_shape)
            elif current_task == TASK_TESSELLATION and current_shape:
                angle = calculate_triangle_angle(current_shape[0])
        task_id = tasks.index(current_task)
        world_id = worlds.index(current_world)
        state = np.array(
            [
                current_pos[0],
                current_pos[1],
                dx,
                dy,
                task_id,
                world_id,
                shape_progress,
                num_vertices,
                angle,
                dist_to_close,
            ]
        )
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        target = (
            end_point if current_task == TASK_LINE else (circle_center, circle_radius)
        )
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
            -global_step / epsilon_decay
        )

        actions = []
        if current_task == TASK_TRIANGLE and len(current_shape) < 3:
            actions = macro_action_draw_triangle(current_shape, current_world)
        elif current_task == TASK_CIRCLE and circle_center:
            actions = macro_action_draw_circle(
                circle_center, circle_radius, current_world
            )
        else:
            if random.random() < epsilon:
                actions = [random.randint(0, action_dim - 1)]
            else:
                actions = plan_actions(
                    state_tensor,
                    world_model,
                    current_shape,
                    target,
                    current_task,
                    current_world,
                )

        for action in actions:
            if current_task == TASK_LINE:
                curvature = 0
                if action == 0:
                    curvature = -1
                elif action == 1:
                    curvature = 1
                next_segment = draw_line_segment(
                    current_shape[-1], end_point, curvature, current_world
                )
                current_shape.extend(
                    [p for p in next_segment if p is not None]
                )  # Filter out None
            elif current_task == TASK_TRIANGLE:
                angle_adjust = 0
                if action == 2:
                    angle_adjust = -10
                elif action == 3:
                    angle_adjust = 10
                current_shape = draw_triangle(
                    current_shape, angle_adjust, current_world
                )
            elif current_task == TASK_CIRCLE:
                radius_adjust = 0
                if action == 4:
                    radius_adjust = 5
                current_shape, circle_radius = draw_circle(
                    circle_center, circle_radius, radius_adjust, current_world
                )
            elif current_task == TASK_PENTAGON:
                angle_adjust = 0
                if action == 2:
                    angle_adjust = -10
                elif action == 3:
                    angle_adjust = 10
                current_shape = draw_pentagon(
                    current_shape, angle_adjust, current_world
                )
            elif current_task == TASK_TESSELLATION:
                if action == 5:
                    current_shape = draw_tessellation(
                        tessellation_points, current_world
                    )

            shape_progress = (
                len(current_shape) / 5
                if current_task != TASK_TESSELLATION
                else len(current_shape)
            )
            num_vertices = (
                len(current_shape)
                if current_task != TASK_TESSELLATION
                else sum(len(t) for t in current_shape)
            )
            angle = 0
            dist_to_close = 0
            if current_task == TASK_LINE:
                next_pos = current_shape[-1] if current_shape else (0, 0)
                dx = end_point[0] - next_pos[0]
                dy = end_point[1] - next_pos[1]
            else:
                next_pos = (0, 0)
                dx = dy = 0
                if current_task == TASK_TRIANGLE and len(current_shape) == 3:
                    angle = calculate_triangle_angle(current_shape)
                elif current_task == TASK_PENTAGON and len(current_shape) >= 2:
                    dist_to_close = calculate_dist_to_close(current_shape)
                elif current_task == TASK_TESSELLATION and current_shape:
                    angle = calculate_triangle_angle(current_shape[0])
            next_state = np.array(
                [
                    next_pos[0],
                    next_pos[1],
                    dx,
                    dy,
                    task_id,
                    world_id,
                    shape_progress,
                    num_vertices,
                    angle,
                    dist_to_close,
                ]
            )

            extrinsic_reward = calculate_reward(
                current_shape, target, current_task, current_world
            )
            action_tensor = (
                torch.tensor([action], dtype=torch.float).unsqueeze(0).to(device)
            )
            predicted_next_state = forward_model(state_tensor, action_tensor)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            intrinsic_reward = torch.norm(
                predicted_next_state - next_state_tensor
            ).item()
            current_reward = extrinsic_reward + 0.3 * intrinsic_reward
            rewards.append(current_reward)
            reward_history.append(current_reward)

            done = 0
            if (
                ai_step >= max_steps
                or (
                    current_task == TASK_LINE
                    and np.sqrt(
                        (next_pos[0] - end_point[0]) ** 2
                        + (next_pos[1] - end_point[1]) ** 2
                    )
                    < 10
                )
                or (current_task == TASK_TESSELLATION and len(current_shape) >= 4)
            ):
                done = 1
                game_state = "waiting"
                episode_count += 1
                episode_rewards.append(sum(rewards) / len(rewards) if rewards else 0)
                current_task = predict_next_shape()
                current_world = worlds[(worlds.index(current_world) + 1) % len(worlds)]

            store_transition((state, action, current_reward, next_state, done))
            for _ in range(train_iters):
                optimize_model()

            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            fps = clock.get_fps()
            current_lr = adjust_learning_rate(fps)
            next_shape = predict_next_shape()
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        episode_count,
                        global_step,
                        state.tolist(),
                        action,
                        current_reward,
                        avg_reward,
                        current_lr,
                        fps,
                        next_shape,
                    ]
                )

            ai_step += 1
            step_counter += 1
            global_step += 1
            if step_counter % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    # Drawing
    screen.fill(BLACK)
    draw_wireframe(current_world)

    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    fps = clock.get_fps()
    info_text = [
        f"World: {current_world}",
        f"Task: {current_task}",
        f"Next Shape: {predict_next_shape()}",
        f"State: {running_state}",
        f"Episode: {episode_count}",
        f"Reward: {current_reward:.2f}",
        f"Avg Reward: {avg_reward:.2f}",
        f"LR: {optimizer.param_groups[0]['lr']:.6f}",
        f"FPS: {fps:.1f}",
        f"Tick: {tick_rate}, Iters: {train_iters}",
        "Cmds: start, pause, resume, quit, faster, slower, iters_up, iters_down",
    ]
    for i, line in enumerate(info_text):
        text = font.render(line, True, WHITE)
        screen.blit(text, (10, 10 + i * 30))

    if current_task == TASK_LINE:
        pygame.draw.circle(screen, RED, to_int_point(end_point), 5)
    elif current_task == TASK_TRIANGLE:
        for p in triangle_points:
            pygame.draw.circle(screen, RED, to_int_point(p), 5)
    elif current_task == TASK_CIRCLE and circle_center:
        pygame.draw.circle(screen, RED, to_int_point(circle_center), 5)
    elif current_task == TASK_PENTAGON:
        for p in pentagon_points:
            pygame.draw.circle(screen, RED, to_int_point(p), 5)
    elif current_task == TASK_TESSELLATION:
        for p in tessellation_points:
            pygame.draw.circle(screen, RED, to_int_point(p), 5)

    if current_shape:
        try:
            if current_task == TASK_LINE:
                pygame.draw.circle(screen, BLUE, to_int_point(start_point), 5)
                for i in range(len(current_shape) - 1):
                    start_pos = current_shape[i]
                    end_pos = current_shape[i + 1]
                    if (
                        start_pos
                        and end_pos
                        and isinstance(start_pos, (tuple, list))
                        and isinstance(end_pos, (tuple, list))
                    ):
                        pygame.draw.line(
                            screen,
                            CYAN,
                            to_int_point(start_pos),
                            to_int_point(end_pos),
                            2,
                        )
                    else:
                        print(
                            f"Invalid line segment at index {i}: start={start_pos}, end={end_pos}"
                        )
            elif current_task == TASK_TRIANGLE and len(current_shape) == 3:
                for i in range(3):
                    start_pos = current_shape[i]
                    end_pos = current_shape[(i + 1) % 3]
                    if (
                        start_pos
                        and end_pos
                        and isinstance(start_pos, (tuple, list))
                        and isinstance(end_pos, (tuple, list))
                    ):
                        pygame.draw.line(
                            screen,
                            CYAN,
                            to_int_point(start_pos),
                            to_int_point(end_pos),
                            2,
                        )
                    else:
                        print(
                            f"Invalid triangle segment at index {i}: start={start_pos}, end={end_pos}"
                        )
            elif current_task == TASK_CIRCLE:
                for i in range(len(current_shape) - 1):
                    start_pos = current_shape[i]
                    end_pos = current_shape[i + 1]
                    if (
                        start_pos
                        and end_pos
                        and isinstance(start_pos, (tuple, list))
                        and isinstance(end_pos, (tuple, list))
                    ):
                        pygame.draw.line(
                            screen,
                            CYAN,
                            to_int_point(start_pos),
                            to_int_point(end_pos),
                            2,
                        )
                    else:
                        print(
                            f"Invalid circle segment at index {i}: start={start_pos}, end={end_pos}"
                        )
            elif current_task == TASK_PENTAGON and len(current_shape) == 5:
                for i in range(5):
                    start_pos = current_shape[i]
                    end_pos = current_shape[(i + 1) % 5]
                    if (
                        start_pos
                        and end_pos
                        and isinstance(start_pos, (tuple, list))
                        and isinstance(end_pos, (tuple, list))
                    ):
                        pygame.draw.line(
                            screen,
                            CYAN,
                            to_int_point(start_pos),
                            to_int_point(end_pos),
                            2,
                        )
                    else:
                        print(
                            f"Invalid pentagon segment at index {i}: start={start_pos}, end={end_pos}"
                        )
            elif current_task == TASK_TESSELLATION:
                for triangle in current_shape:
                    if not isinstance(triangle, (list, tuple)) or len(triangle) != 3:
                        print(f"Invalid triangle in tessellation: {triangle}")
                        continue
                    for i in range(3):
                        start_pos = triangle[i]
                        end_pos = triangle[(i + 1) % 3]
                        if (
                            start_pos
                            and end_pos
                            and isinstance(start_pos, (tuple, list))
                            and isinstance(end_pos, (tuple, list))
                        ):
                            pygame.draw.line(
                                screen,
                                CYAN,
                                to_int_point(start_pos),
                                to_int_point(end_pos),
                                2,
                            )
                        else:
                            print(
                                f"Invalid tessellation segment: start={start_pos}, end={end_pos}"
                            )
        except Exception as e:
            print(f"Error in drawing shape: {e}, current_shape={current_shape}")

    if time.time() - last_plot_update > 1:
        plot_surface = update_reward_plot()
        last_plot_update = time.time()
    if plot_surface:
        screen.blit(plot_surface, (WIDTH - 310, 10))

    pygame.display.flip()
    clock.tick(tick_rate if running_state == "running" else 5)

pygame.quit()
