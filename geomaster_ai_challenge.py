from collections import deque
import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import time
import matplotlib.pyplot as plt
from io import BytesIO
from queue import Queue
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH = 1200  # Increase the width (e.g., from 1000 to 1200)
HEIGHT = 850
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

# Thread-safe logging
log_queue = Queue()
debug_mode = False

def log_message(message):
    if debug_mode:
        log_queue.put(message)

def process_logs():
    while not log_queue.empty():
        print(log_queue.get())

# Helper function
def to_int_point(point):
    if (
        not isinstance(point, (tuple, list))
        or len(point) != 2
        or not all(isinstance(coord, (int, float)) for coord in point)
    ):
        log_message(f"Invalid point detected: {repr(point)}")
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

# Constants
MAX_SCORE = 50000
NEGATIVE_REWARD_RETRY_LIMIT = 3
POSITIVE_REWARD_RETRY_LIMIT = 3  # Number of additional attempts after achieving positive rewards

# Curriculum learning parameters
curriculum_stage = 0
curriculum_threshold = 10  # Number of successful episodes to progress to the next stage
curriculum_progress = 0  # Tracks successful episodes in the current stage

# Define task progression for curriculum learning
curriculum_tasks = [
    [TASK_LINE],  # Stage 0: Only line tasks
    [TASK_LINE, TASK_TRIANGLE],  # Stage 1: Line and triangle tasks
    [TASK_LINE, TASK_TRIANGLE, TASK_CIRCLE],  # Stage 2: Add circle tasks
    [TASK_LINE, TASK_TRIANGLE, TASK_CIRCLE, TASK_PENTAGON],  # Stage 3: Add pentagon tasks
    tasks,  # Stage 4: All tasks
]

curriculum_worlds = [
    [WORLD_EUCLIDEAN],  # Stage 0: Only Euclidean world
    [WORLD_EUCLIDEAN, WORLD_SPHERICAL],  # Stage 1: Add spherical world
    [WORLD_EUCLIDEAN, WORLD_SPHERICAL, WORLD_HYPERBOLIC],  # Stage 2: Add hyperbolic world
    [WORLD_EUCLIDEAN, WORLD_SPHERICAL, WORLD_HYPERBOLIC, WORLD_ELLIPTICAL],  # Stage 3: Add elliptical world
    worlds,  # Stage 4: All worlds
]

def update_curriculum():
    """Update the curriculum stage based on performance."""
    global curriculum_stage, curriculum_progress
    if curriculum_progress >= curriculum_threshold and curriculum_stage < len(curriculum_tasks) - 1:
        curriculum_stage += 1
        curriculum_progress = 0
        log_message(f"Curriculum advanced to stage {curriculum_stage}. Tasks: {curriculum_tasks[curriculum_stage]}, Worlds: {curriculum_worlds[curriculum_stage]}")

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

    def sample(self, batch_size, alpha=0.6):
        """Sample a batch of experiences with prioritization."""
        batch = []
        idxs = []
        priorities = []
        segment = self.tree[0] / batch_size  # Total priority divided into segments

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)  # Randomly sample within the segment
            idx, priority, data = self.get_leaf(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        sampling_probabilities = np.array(priorities) / self.tree[0]
        is_weights = np.power(len(self.data) * sampling_probabilities, -alpha)
        is_weights /= is_weights.max()  # Normalize importance-sampling weights

        return batch, idxs, is_weights

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
        self.state_dim = state_dim  # Store state_dim as an instance variable
        self.fc1 = nn.Linear(state_dim + 1, 128)
        self.fc2 = nn.Linear(128, state_dim)

    def forward(self, state, action):
        try:
            log_message(f"ForwardModel: state_dim={self.state_dim}, state.shape={state.shape}, action.shape={action.shape}")
            if state.dim() != 2 or action.dim() != 2:
                raise ValueError(
                    f"Expected 2D tensors, got state shape {state.shape}, action shape {action.shape}"
                )
            if state.shape[1] != self.state_dim or action.shape[1] != 1:
                raise ValueError(
                    f"Unexpected feature dimensions: state {state.shape}, action {action.shape}"
                )
            action = action.view(-1, 1)
            x = torch.cat([state, action], dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
        except Exception as e:
            log_message(f"Error in ForwardModel forward: {repr(e)}")
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
                    f"Expected 2D tensors, got state shape {state.shape}, action shape {state.shape}"
                )
            if state.shape[1] != state_dim or action.shape[1] != 1:
                raise ValueError(
                    f"Unexpected feature dimensions: state {state.shape}, action {state.shape}"
                )
            action = action.view(-1, 1)
            x = torch.cat([state, action], dim=1)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
        except Exception as e:
            log_message(f"Error in WorldModel forward: {repr(e)}")
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
state_dim_2d = 10
state_dim_3d = 11  # Add z or dz for 3D
state_dim = state_dim_2d  # Default to 2D
action_dim = 6
predictor_input_dim = 13
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

# PyTorch Lightning Model
class GeoMasterAIModel(pl.LightningModule):
    def __init__(self, state_dim, action_dim, gamma, base_lr):
        super().__init__()
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.gamma = gamma
        self.base_lr = base_lr
        self.loss_fn = nn.MSELoss()

    def forward(self, state):
        return self.policy_net(state)

    def training_step(self, batch, batch_idx):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = self.loss_fn(q_values, expected_q_values)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.base_lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)  # Reduce LR every 10 epochs
        return [optimizer], [scheduler]

# DataLoader for PyTorch Lightning
class GeoMasterDataModule(pl.LightningDataModule):
    def __init__(self, memory, batch_size):
        super().__init__()
        self.memory = memory
        self.batch_size = batch_size

    def train_dataloader(self):
        dataset = self.memory_to_dataset()
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def memory_to_dataset(self):
        transitions = [self.memory.data[i] for i in range(self.memory.n_entries)]
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        return torch.utils.data.TensorDataset(states, actions, rewards, next_states, dones)

# Initialize PyTorch Lightning components
geo_model = GeoMasterAIModel(state_dim, action_dim, gamma, base_lr)
memory = SumTree(memory_size)
geo_data = GeoMasterDataModule(memory, batch_size)

# Replace the training loop with PyTorch Lightning Trainer
trainer = Trainer(max_epochs=100, gpus=1 if torch.cuda.is_available() else 0)

# Train the model
trainer.fit(geo_model, geo_data)

# Update the optimize_model function to include scheduler step
def optimize_model():
    """Optimization is now handled by PyTorch Lightning."""
    trainer.fit(geo_model, geo_data)
    for scheduler in trainer.optimizers[0].schedulers:
        scheduler.step()  # Step the scheduler after each epoch

# Neural Networks
policy_net = geo_model.policy_net.to(device)
target_net = geo_model.target_net.to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=base_lr)

forward_model = ForwardModel(state_dim, action_dim).to(device)
optimizer_fm = optim.Adam(forward_model.parameters(), lr=base_lr)

world_model = WorldModel(state_dim, action_dim).to(device)
optimizer_wm = optim.Adam(world_model.parameters(), lr=base_lr)

shape_predictor = NextShapePredictor(predictor_input_dim, predictor_output_dim).to(device)
optimizer_sp = optim.Adam(shape_predictor.parameters(), lr=0.001)

n_step_buffer = []

# Adaptive exploration parameters
epsilon_noise_scale = 0.1  # Scale of noise added to actions during exploration

def select_action(state_tensor, epsilon, action_dim):
    """Select an action using epsilon-greedy with noise."""
    if random.random() < epsilon:
        # Exploration: Add noise to a random action
        action = random.randint(0, action_dim - 1)
        noise = np.random.normal(0, epsilon_noise_scale, size=action_dim)
        noisy_action = np.clip(action + noise[action], 0, action_dim - 1)
        return int(noisy_action)
    else:
        # Exploitation: Choose the best action
        with torch.no_grad():
            q_values = policy_net(state_tensor).cpu().numpy().flatten()
            return int(np.argmax(q_values))

# Game State
start_point = None
end_point = (WIDTH - 50, HEIGHT - 50)
triangle_points = []
circle_center = None

# Initialize circle_radius with a default value to avoid undefined reference
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
negative_reward_attempts = 0  # Track retries for negative rewards
positive_reward_attempts = 0  # Track retries for positive rewards

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
        return (0, 0)
    factor = 2 * R / (2 * R - z)
    proj_x = factor * x + WIDTH / 2
    proj_y = factor * y + HEIGHT / 2
    return (proj_x, proj_y)

def poincare_disk_projection(point, R=100):
    x, y = point
    dist = np.sqrt(x**2 + y**2)
    if dist >= R:
        return (WIDTH / 2, HEIGHT / 2)
    factor = R * np.tanh(dist / R)
    angle = np.arctan2(y, x)
    proj_x = factor * np.cos(angle) + WIDTH / 2
    proj_y = factor * np.sin(angle) + WIDTH / 2
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
        return (WIDTH / 2, HEIGHT / 2)
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
        proj = (x + WIDTH / 2, y + HEIGHT / 2)
    elif world == WORLD_SPHERICAL:
        z = np.sqrt(max(0, 100**2 - x**2 - y**2))
        proj = stereographic_projection((x, y, z))
    elif world == WORLD_HYPERBOLIC:
        proj = poincare_disk_projection((x - WIDTH / 2, y - HEIGHT / 2))
    elif world == WORLD_ELLIPTICAL:
        proj = elliptical_projection((x, y))
    elif world == WORLD_PROJECTIVE:
        proj = projective_projection((x, y))
    elif world == WORLD_FRACTAL:
        proj = fractal_projection((x, y))
    else:
        proj = (x, y)
    return (max(0, min(proj[0], WIDTH)), max(0, min(proj[1], HEIGHT)))

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

# 3D Projections
def perspective_projection(point, fov=90, aspect_ratio=WIDTH / HEIGHT, near=1, far=1000):
    x, y, z = point
    if z <= near:
        return None
    scale = 1 / np.tan(np.radians(fov) / 2)
    proj_x = (scale * x / z) * WIDTH / 2 + WIDTH / 2
    proj_y = (scale * y / z) * HEIGHT / 2 + HEIGHT / 2
    return (proj_x, proj_y)

def orthographic_projection(point):
    x, y, z = point
    proj_x = x + WIDTH / 2
    proj_y = y + HEIGHT / 2
    return (proj_x, proj_y)

# Extend project_point to handle 3D
def project_point_3d(point, world, projection="perspective"):
    if len(point) == 3:
        if projection == "perspective":
            return perspective_projection(point)
        elif projection == "orthographic":
            return orthographic_projection(point)
    return project_point(point, world)

# Toggle between 2D and 3D modes
is_3d_mode = False
projection_mode = "perspective"  # Options: "perspective", "orthographic"

def update_network_dimensions():
    global policy_net, target_net, forward_model, world_model, state_dim
    state_dim = state_dim_3d if is_3d_mode else state_dim_2d
    log_message(f"Updating network dimensions: state_dim={state_dim}, is_3d_mode={is_3d_mode}")
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    forward_model = ForwardModel(state_dim, action_dim).to(device)  # Re-initialize with updated state_dim
    world_model = WorldModel(state_dim, action_dim).to(device)
    global optimizer, optimizer_fm, optimizer_wm
    optimizer = optim.Adam(policy_net.parameters(), lr=base_lr)
    optimizer_fm = optim.Adam(forward_model.parameters(), lr=base_lr)
    optimizer_wm = optim.Adam(world_model.parameters(), lr=base_lr)

def toggle_3d_mode():
    global is_3d_mode
    is_3d_mode = not is_3d_mode
    update_network_dimensions()
    log_message(f"3D mode {'enabled' if is_3d_mode else 'disabled'}")

# Extend tasks to 3D
def draw_line_segment_3d(start, end, curvature, world):
    try:
        if not (isinstance(start, (tuple, list)) and isinstance(end, (tuple, list))):
            raise ValueError(f"Invalid start or end point: start={repr(start)}, end={repr(end)}")
        if len(start) != 3 or len(end) != 3:
            raise ValueError(f"Start and end must be 3D points: start={repr(start)}, end={repr(end)}")
        
        points = []
        num_points = 10
        for i in range(num_points + 1):
            t = i / num_points
            x = (1 - t) * start[0] + t * end[0]
            y = (1 - t) * start[1] + t * end[1]
            z = (1 - t) * start[2] + t * end[2]
            offset = curvature * 50 * np.sin(np.pi * t)
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            dz = end[2] - start[2]
            length = np.sqrt(dx**2 + dy**2 + dz**2)
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
                x += offset * perp_x
                y += offset * perp_y
            proj = project_point_3d((x, y, z), world)
            if proj:
                points.append(proj)
        return points
    except Exception as e:
        log_message(f"Error in draw_line_segment_3d: {repr(e)}")
        return [start]

def draw_triangle_3d(points, angle_adjust, world):
    try:
        if len(points) != 3:
            return points if len(points) < 3 else points[:3]
        if not all(isinstance(p, (tuple, list)) and len(p) == 3 for p in points):
            raise ValueError(f"Invalid points for triangle: {repr(points)}")
        
        new_points = points.copy()
        p1, p2, p3 = points
        dx = p3[0] - p2[0]
        dy = p3[1] - p2[1]
        dz = p3[2] - p2[2]
        angle = np.arctan2(dy, dx) + angle_adjust * np.pi / 180
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        new_x = p2[0] + length * np.cos(angle)
        new_y = p2[1] + length * np.sin(angle)
        new_z = p2[2]  # Keep z constant for simplicity
        new_points[2] = (new_x, new_y, new_z)
        return [
            project_point_3d((p[0], p[1], p[2]), world) for p in new_points
        ]
    except Exception as e:
        log_message(f"Error in draw_triangle_3d: {repr(e)}")
        return points

def draw_circle_3d(center, radius, radius_adjust, world):
    try:
        radius = max(10, radius + radius_adjust)
        points = []
        for theta in np.linspace(0, 2 * np.pi, 20):
            for phi in np.linspace(0, np.pi, 10):  # Add depth for 3D
                x = center[0] + radius * np.sin(phi) * np.cos(theta)
                y = center[1] + radius * np.sin(phi) * np.sin(theta)
                z = center[2] + radius * np.cos(phi)
                proj = project_point_3d((x, y, z), world)
                if proj:
                    points.append(proj)
        return points, radius
    except Exception as e:
        log_message(f"Error in draw_circle_3d: {repr(e)}")
        return [], radius

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
            raise ValueError(f"Invalid start or end point: start={repr(start)}, end={repr(end)}")
        if len(start) != 2 or len(end) != 2:
            raise ValueError(
                f"Start and end must be 2D points: start={repr(start)}, end={repr(end)}"
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
            points.append(proj)
        return points
    except Exception as e:
        log_message(f"Error in draw_line_segment: {repr(e)}")
        return [start]

def draw_triangle(points, angle_adjust, world):
    try:
        if len(points) != 3:
            return points if len(points) < 3 else points[:3]
        if not all(isinstance(p, (tuple, list)) and len(p) == 2 for p in points):
            raise ValueError(f"Invalid points for triangle: {repr(points)}")
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
        log_message(f"Error in draw_triangle: {repr(e)}")
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
    if len(points) != 5:
        return points if len(points) < 5 else points[:5]
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
        return [points]
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
    return [[project_point((p[0] - WIDTH / 2, p[1] - HEIGHT / 2), world) for p in triangle] for triangle in tessellation]

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
    try:
        if len(points) < 3:
            return 0
        # Flatten if nested (e.g., from tessellation)
        flat_points = []
        for p in points[:3]:
            if isinstance(p, (tuple, list)) and len(p) >= 2:
                flat_points.append((float(p[0]), float(p[1])))
            else:
                raise ValueError(f"Invalid point format: {repr(p)}")
        if len(flat_points) != 3:
            raise ValueError(f"Expected 3 points, got {len(flat_points)}")
        p1, p2, p3 = flat_points
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            return 0
        dot_product = np.dot(v1, v2)
        norms_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norms_product == 0 or not (-1 <= dot_product / norms_product <= 1):
            return 0
        angle = (
            np.arccos(dot_product / norms_product)
            * 180
            / np.pi
        )
        return angle
    except Exception as e:
        log_message(f"Error in calculate_triangle_angle: {repr(e)}")
        return 0

def calculate_dist_to_close(points):
    try:
        if not points or len(points) < 2:
            return 0
        if all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in points):
            p1 = points[0]
            p_last = points[-1]
            return np.sqrt((p1[0] - p_last[0]) ** 2 + (p1[1] - p_last[1]) ** 2)
        elif all(isinstance(t, (list, tuple)) and len(t) == 3 for t in points):
            if not points[0] or not points[-1]:
                return 0
            p1 = points[0][0]
            p_last = points[-1][-1]
            return np.sqrt((p1[0] - p_last[0]) ** 2 + (p1[1] - p_last[1]) ** 2)
        else:
            log_message(f"Invalid points structure for dist_to_close: {repr(points)}")
            return 0
    except Exception as e:
        log_message(f"Error in calculate_dist_to_close: {repr(e)}")
        return 0

# Reward normalization constants
REWARD_MIN = -50  # Minimum possible reward
REWARD_MAX = 50   # Maximum possible reward

def normalize_reward(reward):
    """Normalize reward to a range of [-1, 1]."""
    return 2 * (reward - REWARD_MIN) / (REWARD_MAX - REWARD_MIN) - 1

# Advanced Reward System
def calculate_advanced_reward(shape, target, task, world):
    try:
        reward = 0
        efficiency_factor = 1.0
        creativity_factor = 1.0
        constraint_penalty = 0

        if task == TASK_LINE:
            if not shape:
                return normalize_reward(-10)
            current_pos = shape[-1] if shape else (0, 0)
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            reward = -distance / 100.0
            if distance < 10:
                reward += 50
            efficiency_factor = max(0.5, 1 - len(shape) / 50)  # Penalize longer paths

        elif task == TASK_TRIANGLE:
            if len(shape) != 3:
                return normalize_reward(-10)
            angle = calculate_triangle_angle(shape)
            if world == WORLD_SPHERICAL:
                reward = (angle - 60) / 10 if angle > 60 else -10
            elif world == WORLD_HYPERBOLIC:
                reward = (60 - angle) / 10 if angle < 60 else -10
            else:
                reward = -abs(angle - 60) / 10
                if abs(angle - 60) < 10:
                    reward += 10
            creativity_factor = 1 + (angle / 180)  # Reward diverse angles

        elif task == TASK_CIRCLE:
            if not shape:
                return normalize_reward(-10)
            current_radius = target[1]
            ideal_radius = 50
            reward = -abs(current_radius - ideal_radius) / 10
            if abs(current_radius - ideal_radius) < 5:
                reward += 20
            efficiency_factor = max(0.5, 1 - abs(current_radius - ideal_radius) / 50)

        elif task == TASK_PENTAGON:
            if len(shape) != 5:
                return normalize_reward(-10)
            dist = calculate_dist_to_close(shape)
            reward = -dist / 10
            if dist < 10:
                reward += 30
            creativity_factor = 1 + (len(set(shape)) / 5)  # Reward unique vertices

        elif task == TASK_TESSELLATION:
            if not shape or not isinstance(shape, (list, tuple)):
                return normalize_reward(-10)
            num_triangles = len(shape)
            reward = num_triangles * 10
            for triangle in shape:
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
            efficiency_factor = max(0.5, 1 - num_triangles / 10)

        # Apply penalties for constraint violations
        if task == TASK_TRIANGLE and len(shape) > 3:
            constraint_penalty = -5 * (len(shape) - 3)

        # Combine factors into the final reward
        reward = reward * efficiency_factor * creativity_factor + constraint_penalty

        # Enforce maximum score limit
        if sum(rewards) + reward > MAX_SCORE:
            reward = MAX_SCORE - sum(rewards)
            log_message(f"Score capped at {MAX_SCORE}.")

        # Normalize the reward
        return normalize_reward(reward)
    except Exception as e:
        log_message(f"Error in calculate_advanced_reward: {repr(e)}")
        return normalize_reward(-10)

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

# Modify reset_episode to use curriculum learning
def reset_episode():
    global start_point, triangle_points, circle_center, pentagon_points, tessellation_points, current_shape
    global game_state, ai_step, current_task, end_point, circle_radius, negative_reward_attempts, positive_reward_attempts
    global curriculum_progress

    # Initialize circle_radius with a default value
    circle_radius = 50

    if negative_reward_attempts >= NEGATIVE_REWARD_RETRY_LIMIT:
        log_message("Max retries for negative rewards reached. Moving to next task.")
        negative_reward_attempts = 0  # Reset retry counter
        positive_reward_attempts = 0  # Reset positive retry counter
    elif sum(rewards) < 0:
        log_message(f"Negative reward detected. Retrying task. Attempt {negative_reward_attempts + 1}/{NEGATIVE_REWARD_RETRY_LIMIT}.")
        negative_reward_attempts += 1
        return  # Retry the current task

    if sum(rewards) > 0 and positive_reward_attempts < POSITIVE_REWARD_RETRY_LIMIT:
        log_message(f"Positive reward achieved. Reattempting task for additional training. Attempt {positive_reward_attempts + 1}/{POSITIVE_REWARD_RETRY_LIMIT}.")
        positive_reward_attempts += 1
        save_model_weights(current_task, current_world)  # Save progress
        return  # Retry the current task

    # Update curriculum progress
    if sum(rewards) > 0:
        curriculum_progress += 1
        update_curriculum()

    # Reset counters for the next task
    negative_reward_attempts = 0
    positive_reward_attempts = 0

    # Generate a new task based on the current curriculum stage
    dynamic_task = generate_dynamic_task(curriculum_tasks[curriculum_stage], curriculum_worlds[curriculum_stage])
    current_task = dynamic_task["type"]

    if current_task == TASK_LINE:
        start_point = dynamic_task["start"]
        end_point = dynamic_task["end"]
        if is_3d_mode:  # Ensure `end_point` is a 3D point in 3D mode
            start_point = (start_point[0], start_point[1], random.randint(-100, 100))
            end_point = (end_point[0], end_point[1], random.randint(-100, 100))
        current_shape = [start_point]
    elif current_task == TASK_TRIANGLE:
        triangle_points = dynamic_task["points"]
        if is_3d_mode:  # Ensure points are 3D in 3D mode
            triangle_points = [(p[0], p[1], random.randint(-100, 100)) for p in triangle_points]
        current_shape = triangle_points.copy()
    elif current_task == TASK_CIRCLE:
        circle_center = dynamic_task["center"]
        circle_radius = dynamic_task["radius"]
        if is_3d_mode:  # Ensure `circle_center` is a 3D point in 3D mode
            circle_center = (circle_center[0], circle_center[1], random.randint(-100, 100))
        current_shape = []
    elif current_task == TASK_PENTAGON:
        pentagon_points = dynamic_task["points"]
        if is_3d_mode:  # Ensure points are 3D in 3D mode
            pentagon_points = [(p[0], p[1], random.randint(-100, 100)) for p in pentagon_points]
        current_shape = pentagon_points.copy()
    elif current_task == TASK_TESSELLATION:
        tessellation_points = dynamic_task["base_points"]
        if is_3d_mode:  # Ensure points are 3D in 3D mode
            tessellation_points = [(p[0], p[1], random.randint(-100, 100)) for p in tessellation_points]
        current_shape = [tessellation_points.copy()]
    else:
        log_message(f"Unknown task type: {current_task}")

    game_state = "ai_drawing"
    ai_step = 0
    rewards.clear()  # Reset rewards for the new episode

# Modify generate_dynamic_task to accept task and world constraints
def generate_dynamic_task(allowed_tasks, allowed_worlds):
    """Generate a new dynamic task with random parameters within the allowed tasks and worlds."""
    task_type = random.choice(allowed_tasks)
    world = random.choice(allowed_worlds)
    if task_type == TASK_LINE:
        start = (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
        end = (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
        return {"type": TASK_LINE, "start": start, "end": end, "world": world}
    elif task_type == TASK_TRIANGLE:
        points = [(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)) for _ in range(3)]
        return {"type": TASK_TRIANGLE, "points": points, "world": world}
    elif task_type == TASK_CIRCLE:
        center = (random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50))
        radius = random.randint(20, 100)
        return {"type": TASK_CIRCLE, "center": center, "radius": radius, "world": world}
    elif task_type == TASK_PENTAGON:
        points = [(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)) for _ in range(5)]
        return {"type": TASK_PENTAGON, "points": points, "world": world}
    elif task_type == TASK_TESSELLATION:
        base_points = [(random.randint(50, WIDTH - 50), random.randint(50, HEIGHT - 50)) for _ in range(3)]
        return {"type": TASK_TESSELLATION, "base_points": base_points, "world": world}
    else:
        return {"type": "Unknown", "world": world}

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

# Real-time Dashboard
dashboard_surface = None

def update_dashboard():
    """Update the real-time dashboard with rewards, losses, and task progression."""
    global dashboard_surface, episode_rewards, reward_history, curriculum_stage

    # Create a new figure for the dashboard
    plt.figure(figsize=(10, 4))

    # Subplot 1: Reward Trend
    plt.subplot(1, 3, 1)
    plt.plot(list(episode_rewards), label="Avg Reward", color="cyan")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward Trend")
    plt.legend()

    # Subplot 2: Recent Rewards
    plt.subplot(1, 3, 2)
    if len(reward_history) > 0:
        plt.plot(list(reward_history), label="Recent Rewards", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Recent Rewards")
    plt.legend()

    # Subplot 3: Curriculum Progression
    plt.subplot(1, 3, 3)
    stages = [f"Stage {i}" for i in range(len(curriculum_tasks))]
    progression = [1 if i <= curriculum_stage else 0 for i in range(len(curriculum_tasks))]
    plt.bar(stages, progression, color="green")
    plt.xlabel("Curriculum Stages")
    plt.ylabel("Progress")
    plt.title("Curriculum Progression")

    # Save the dashboard to a buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    dashboard_surface = pygame.image.load(buf)
    plt.close()
    buf.close()

    # Scale the dashboard for display
    dashboard_surface = pygame.transform.scale(dashboard_surface, (600, 200))

# Transfer Learning
def save_model_weights(task, world):
    """Save the model weights for a specific task and world."""
    filename = f"model_{task}_{world}.pth"
    torch.save(policy_net.state_dict(), filename)
    log_message(f"Model weights saved to {filename}")

def load_model_weights(task, world):
    """Load pretrained model weights for a specific task and world."""
    filename = f"model_{task}_{world}.pth"
    if os.path.exists(filename):
        policy_net.load_state_dict(torch.load(filename))
        target_net.load_state_dict(policy_net.state_dict())
        log_message(f"Model weights loaded from {filename}")
    else:
        log_message(f"No saved weights found for {task} in {world}")

def fine_tune_model(task, world, fine_tune_steps=1000):
    """Fine-tune the model on a new task or environment."""
    global circle_radius  # Ensure circle_radius is accessible and initialized
    circle_radius = 50  # Initialize circle_radius with a default value

    log_message(f"Starting fine-tuning for task: {task}, world: {world}")
    load_model_weights(task, world)  # Load pretrained weights if available

    for step in range(fine_tune_steps):
        reset_episode()  # Reset the environment for the task
        for _ in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = select_action(state_tensor, epsilon, action_dim)

            # Simulate the action
            actions = [action]
            for action in actions:
                if current_task == TASK_LINE:
                    curvature = 0
                    next_segment = draw_line_segment(
                        current_shape[-1], end_point, curvature, current_world
                    )
                    current_shape.extend([p for p in next_segment if p is not None])
                elif current_task == TASK_TRIANGLE:
                    angle_adjust = 0
                    current_shape = draw_triangle(
                        current_shape, angle_adjust, current_world
                    )
                elif current_task == TASK_CIRCLE:
                    radius_adjust = 0
                    current_shape, circle_radius = draw_circle(
                        circle_center, circle_radius, radius_adjust, current_world
                    )
                elif current_task == TASK_PENTAGON:
                    angle_adjust = 0
                    current_shape = draw_pentagon(
                        current_shape, angle_adjust, current_world
                    )
                elif current_task == TASK_TESSELLATION:
                    base_points = current_shape[0] if current_shape else tessellation_points
                    current_shape = draw_tessellation(base_points, current_world)

            # Calculate reward
            extrinsic_reward = calculate_advanced_reward(
                current_shape, end_point, current_task, current_world
            )
            current_reward = extrinsic_reward
            rewards.append(current_reward)

            # Store transition and optimize model
            next_state = state  # Assume state transition logic is handled elsewhere
            done = 0  # Assume task completion logic is handled elsewhere
            store_transition((state, action, current_reward, next_state, done))
            optimize_model()

            if done:
                break

        # Save fine-tuned weights periodically
        if step % 100 == 0:
            save_model_weights(task, world)
            log_message(f"Fine-tuning progress: {step}/{fine_tune_steps} steps completed.")

    log_message(f"Fine-tuning completed for task: {task}, world: {world}")

# Explainable AI (XAI)
def explain_decision(state_tensor):
    """Provide insights into decision-making by analyzing Q-values and action probabilities."""
    with torch.no_grad():
        q_values = policy_net(state_tensor).cpu().numpy().flatten()
    # Numerically stable softmax for action probabilities
    q_values_exp = np.exp(q_values - np.max(q_values))
    action_probabilities = q_values_exp / np.sum(q_values_exp)
    explanation = {
        "action_probabilities": action_probabilities.tolist(),
        "q_values": q_values.tolist(),
    }
    log_message(f"Decision explanation: {explanation}")
    return explanation

def visualize_q_values(q_values):
    """Visualize Q-values for each action."""
    plt.figure(figsize=(6, 4))
    actions = [f"Action {i}" for i in range(len(q_values))]
    plt.bar(actions, q_values, color="blue")
    plt.xlabel("Actions")
    plt.ylabel("Q-Values")
    plt.title("Q-Values for Actions")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = pygame.image.load(buf)
    plt.close()
    buf.close()
    return pygame.transform.scale(image, (300, 200))

def visualize_action_probabilities(action_probabilities):
    """Visualize action probabilities as a pie chart."""
    plt.figure(figsize=(6, 4))
    actions = [f"Action {i}" for i in range(len(action_probabilities))]
    plt.pie(action_probabilities, labels=actions, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Action Probabilities")
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = pygame.image.load(buf)
    plt.close()
    buf.close()
    return pygame.transform.scale(image, (300, 200))

# Model Evaluation
def evaluate_model(num_episodes=10):
    """Evaluate the model's performance on predefined tasks."""
    global current_task, current_world, game_state, ai_step, current_shape, circle_radius
    evaluation_rewards = []
    success_count = 0
    total_steps = 0

    predefined_tasks = [
        {"task": TASK_LINE, "world": WORLD_EUCLIDEAN},
        {"task": TASK_TRIANGLE, "world": WORLD_SPHERICAL},
        {"task": TASK_CIRCLE, "world": WORLD_HYPERBOLIC},
        {"task": TASK_PENTAGON, "world": WORLD_ELLIPTICAL},
        {"task": TASK_TESSELLATION, "world": WORLD_PROJECTIVE},
    ]

    for episode in range(num_episodes):
        for task_config in predefined_tasks:
            current_task = task_config["task"]
            current_world = task_config["world"]
            reset_episode()  # Reset the environment for the task

            # Initialize circle_radius to avoid undefined reference
            if current_task == TASK_CIRCLE:
                circle_radius = 50

            episode_reward = 0
            for step in range(max_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                if random.random() < epsilon_end:
                    action = random.randint(0, action_dim - 1)
                else:
                    with torch.no_grad():
                        action = policy_net(state_tensor).argmax().item()

                # Simulate the action
                actions = [action]
                for action in actions:
                    if current_task == TASK_LINE:
                        curvature = 0
                        next_segment = draw_line_segment(
                            current_shape[-1], end_point, curvature, current_world
                        )
                        current_shape.extend([p for p in next_segment if p is not None])
                    elif current_task == TASK_TRIANGLE:
                        angle_adjust = 0
                        current_shape = draw_triangle(
                            current_shape, angle_adjust, current_world
                        )
                    elif current_task == TASK_CIRCLE:
                        radius_adjust = 0
                        current_shape, circle_radius = draw_circle(
                            circle_center, circle_radius, radius_adjust, current_world
                        )
                    elif current_task == TASK_PENTAGON:
                        angle_adjust = 0
                        current_shape = draw_pentagon(
                            current_shape, angle_adjust, current_world
                        )
                    elif current_task == TASK_TESSELLATION:
                        base_points = current_shape[0] if current_shape else tessellation_points
                        current_shape = draw_tessellation(base_points, current_world)

                # Calculate reward
                extrinsic_reward = calculate_advanced_reward(
                    current_shape, end_point, current_task, current_world
                )
                episode_reward += extrinsic_reward

                # Check for task completion
                if (
                    current_task == TASK_LINE
                    and np.sqrt(
                        (current_shape[-1][0] - end_point[0]) ** 2
                        + (current_shape[-1][1] - end_point[1]) ** 2
                    )
                    < 10
                ):
                    success_count += 1
                    break

            evaluation_rewards.append(episode_reward)
            total_steps += step + 1

    avg_reward = sum(evaluation_rewards) / len(evaluation_rewards)
    success_rate = success_count / (num_episodes * len(predefined_tasks))
    avg_steps = total_steps / (num_episodes * len(predefined_tasks))

    log_message(f"Evaluation Results: Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2%}, Avg Steps: {avg_steps:.2f}")
    return {"avg_reward": avg_reward, "success_rate": success_rate, "avg_steps": avg_steps}

# Hyperparameter Tuning
def tune_hyperparameters(avg_reward, fps):
    """Dynamically adjust hyperparameters based on performance."""
    global base_lr, epsilon_decay, REWARD_MIN, REWARD_MAX

    # Adjust learning rate based on average reward
    if avg_reward > 10:
        base_lr = min(base_lr * 1.1, 0.01)  # Increase learning rate
    elif avg_reward < -5:
        base_lr = max(base_lr * 0.9, 0.0001)  # Decrease learning rate

    # Adjust epsilon decay based on FPS
    if fps < 20:
        epsilon_decay = max(epsilon_decay * 1.1, 1000)  # Slow down decay
    elif fps > 60:
        epsilon_decay = max(epsilon_decay * 0.9, 100)  # Speed up decay

    # Adjust reward scaling dynamically
    if avg_reward > 20:
        REWARD_MAX = min(REWARD_MAX + 5, 100)
    elif avg_reward < -10:
        REWARD_MIN = max(REWARD_MIN - 5, -100)

    # Update optimizer learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr

    log_message(f"Hyperparameters tuned: base_lr={base_lr:.6f}, epsilon_decay={epsilon_decay}, REWARD_MIN={REWARD_MIN}, REWARD_MAX={REWARD_MAX}")

# Main Game Loop
clock = pygame.time.Clock()
running = True
last_plot_update = 0
step_counter = 0

# Command handling via Pygame keys
command_keys = {
    pygame.K_s: "start",
    pygame.K_p: "pause",
    pygame.K_r: "resume",
    pygame.K_q: "quit",
    pygame.K_f: "faster",
    pygame.K_l: "slower",
    pygame.K_u: "iters_up",
    pygame.K_d: "iters_down",
    pygame.K_b: "toggle_debug",
    pygame.K_3: "toggle_3d_mode",
}

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            command = command_keys.get(event.key)
            if command == "start" and running_state == "stopped":
                running_state = "running"
            elif command == "pause" and running_state == "running":
                running_state = "paused"
            elif command == "resume" and running_state == "paused":
                running_state = "running"
            elif command == "quit":
                running = False
            elif command == "faster":
                tick_rate = min(tick_rate + 10, 120)
                log_message(f"Tick rate increased to {tick_rate}")
            elif command == "slower":
                tick_rate = max(tick_rate - 10, 10)
                log_message(f"Tick rate decreased to {tick_rate}")
            elif command == "iters_up":
                train_iters = min(train_iters + 1, 10)
                log_message(f"Training iterations increased to {train_iters}")
            elif command == "iters_down":
                train_iters = max(train_iters - 1, 1)
                log_message(f"Training iterations decreased to {train_iters}")
            elif command == "toggle_debug":
                debug_mode = not debug_mode
                log_message(f"Debug mode {'enabled' if debug_mode else 'disabled'}")
            elif command == "toggle_3d_mode":
                toggle_3d_mode()

    if running_state == "running" and game_state == "waiting":
        reset_episode()
        log_message(f"New dynamic task generated: {current_task}")

    if running_state == "running" and game_state == "ai_drawing":
        shape_progress = (
            len(current_shape) / 5
            if current_task != TASK_TESSELLATION
            else len(current_shape)
        )
        num_vertices = (
            len(current_shape)
            if current_task != TASK_TESSELLATION
            else sum(len(t) for t in current_shape if isinstance(t, (list, tuple)))
        )
        angle = 0
        dist_to_close = 0
        if is_3d_mode:
            current_pos = current_shape[-1] if current_shape else (0, 0, 0)  # Default to (0, 0, 0) if empty
            if not isinstance(current_pos, (tuple, list)) or len(current_pos) < 2:
                current_pos = (0, 0, 0)  # Ensure valid default for 3D
            if len(current_pos) == 2:
                current_pos = (current_pos[0], current_pos[1], 0)
            task_id = tasks.index(current_task)
            world_id = worlds.index(current_world)
            if current_task == TASK_LINE:
                dx = end_point[0] - current_pos[0]
                dy = end_point[1] - current_pos[1]
                dz = end_point[2] - current_pos[2]
                state = np.array([
                    current_pos[0], current_pos[1], dx, dy, task_id,
                    world_id, shape_progress, num_vertices, angle, dz
                ])
            else:
                state = np.array([
                    0, 0, 0, 0, task_id,
                    world_id, shape_progress, num_vertices, angle, 0
                ])
        else:
            current_pos = current_shape[-1] if current_shape else (0, 0)  # Default to (0, 0) if empty
            if not isinstance(current_pos, (tuple, list)) or len(current_pos) < 2:
                current_pos = (0, 0)  # Ensure valid default for 2D
            if isinstance(current_pos[0], (tuple, list)):
                current_pos = (current_pos[0][0], current_pos[0][1])
            if current_task == TASK_LINE:
                dx = end_point[0] - current_pos[0]
                dy = end_point[1] - current_pos[1]
            else:
                dx = dy = 0
                if current_task == TASK_TRIANGLE and len(current_shape) == 3:
                    angle = calculate_triangle_angle(current_shape)
                elif current_task == TASK_PENTAGON and len(current_shape) >= 2:
                    if all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in current_shape):
                        dist_to_close = calculate_dist_to_close(current_shape)
                    else:
                        log_message(f"Invalid current_shape for dist_to_close: {repr(current_shape)}")
                        dist_to_close = 0
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
        if state_tensor.shape[1] != state_dim:
            log_message(f"State tensor dimension mismatch: expected {state_dim}, got {state_tensor.shape[1]}")
            state_tensor = torch.cat([state_tensor, torch.zeros(1, state_dim - state_tensor.shape[1]).to(device)], dim=1)

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
                actions = [select_action(state_tensor, epsilon, action_dim)]
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
            if is_3d_mode:
                if current_task == TASK_LINE:
                    curvature = 0  # Initialize curvature
                    next_segment = draw_line_segment_3d(
                        current_shape[-1], end_point, curvature, current_world
                    )
                    current_shape.extend([p for p in next_segment if p is not None])
                elif current_task == TASK_TRIANGLE:
                    angle_adjust = 0  # Initialize angle_adjust
                    current_shape = draw_triangle_3d(current_shape, angle_adjust, current_world)
                elif current_task == TASK_CIRCLE:
                    radius_adjust = 0  # Initialize radius_adjust
                    current_shape, circle_radius = draw_circle_3d(
                        circle_center, circle_radius, radius_adjust, current_world
                    )
            else:
                if current_task == TASK_LINE:
                    curvature = 0  # Initialize curvature
                    if action == 0:
                        curvature = -1
                    elif action == 1:
                        curvature = 1
                    next_segment = draw_line_segment(
                        current_shape[-1], end_point, curvature, current_world
                    )
                    current_shape.extend([p for p in next_segment if p is not None])
                elif current_task == TASK_TRIANGLE:
                    angle_adjust = 0  # Initialize angle_adjust
                    if action == 2:
                        angle_adjust = -10
                    elif action == 3:
                        angle_adjust = 10
                    current_shape = draw_triangle(
                        current_shape, angle_adjust, current_world
                    )
                elif current_task == TASK_CIRCLE:
                    radius_adjust = 0  # Initialize radius_adjust
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
                        base_points = current_shape[0] if current_shape and isinstance(current_shape[0], (list, tuple)) else tessellation_points
                        current_shape = draw_tessellation(base_points, current_world)

            shape_progress = (
                len(current_shape) / 5
                if current_task != TASK_TESSELLATION
                else len(current_shape)
            )
            num_vertices = (
                len(current_shape)
                if current_task != TASK_TESSELLATION
                else sum(len(t) for t in current_shape if isinstance(t, (list, tuple)))
            )
            angle = 0
            dist_to_close = 0
            if current_task == TASK_LINE:
                next_pos = current_shape[-1] if current_shape else (0, 0)
                # Flatten next_pos if it contains nested tuples
                if isinstance(next_pos[0], (tuple, list)):
                    next_pos = (next_pos[0][0], next_pos[0][1])
                dx = end_point[0] - next_pos[0]
                dy = end_point[1] - next_pos[1]
            else:
                next_pos = (0, 0)
                dx = dy = 0
                if current_task == TASK_TRIANGLE and len(current_shape) == 3:
                    angle = calculate_triangle_angle(current_shape)
                elif current_task == TASK_PENTAGON and len(current_shape) >= 2:
                    if all(isinstance(p, (tuple, list)) and len(p) >= 2 for p in current_shape):
                        dist_to_close = calculate_dist_to_close(current_shape)
                    else:
                        log_message(f"Invalid current_shape for dist_to_close: {repr(current_shape)}")
                        dist_to_close = 0
                elif current_task == TASK_TESSELLATION and current_shape:
                    angle = calculate_triangle_angle(current_shape[0])
            task_id = tasks.index(current_task)
            world_id = worlds.index(current_world)
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

            extrinsic_reward = calculate_advanced_reward(
                current_shape, target, current_task, current_world
            )
            action_tensor = (
                torch.tensor([action], dtype=torch.float).unsqueeze(0).to(device)
            )
            predicted_next_state = forward_model(state_tensor, action_tensor)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            if next_state_tensor.shape[1] != state_dim:
                log_message(f"Next state tensor dimension mismatch: expected {state_dim}, got {next_state_tensor.shape[1]}")
                next_state_tensor = torch.cat([next_state_tensor, torch.zeros(1, state_dim - next_state_tensor.shape[1]).to(device)], dim=1)
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
            if global_step % 5 == 0:  # Train every 5 steps
                for _ in range(train_iters):
                    optimize_model()

            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            fps = clock.get_fps()

            # Call the hyperparameter tuning function
            tune_hyperparameters(avg_reward, fps)

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

            if global_step % 1000 == 0:
                torch.save(policy_net.state_dict(), "policy_net.pth")
                torch.save(shape_predictor.state_dict(), "shape_predictor.pth")
                log_message(f"Models saved at step {global_step}")

            if global_step % 100 == 0:  # Save weights periodically
                save_model_weights(current_task, current_world)

            if global_step % 50 == 0:  # Explain decisions periodically
                explanation = explain_decision(state_tensor)
                q_values_image = visualize_q_values(explanation["q_values"])
                action_probabilities_image = visualize_action_probabilities(explanation["action_probabilities"])
                screen.blit(q_values_image, (WIDTH - 310, HEIGHT - 420))
                screen.blit(action_probabilities_image, (WIDTH - 310, HEIGHT - 220))

            state = next_state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            if state_tensor.shape[1] != state_dim:
                log_message(f"State tensor dimension mismatch: expected {state_dim}, got {state_tensor.shape[1]}")
                state_tensor = torch.cat([state_tensor, torch.zeros(1, state_dim - state_tensor.shape[1]).to(device)], dim=1)

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
        "Keys: S:start, P:pause, R:resume, Q:quit, F:faster, L:slower, U:iters_up, D:iters_down, B:toggle_debug, 3:toggle_3d_mode",
    ]
    for i, line in enumerate(info_text):
        text = font.render(line, True, WHITE)
        screen.blit(text, (10, 10 + i * 30))

    if current_task == TASK_LINE:
        if is_3d_mode:
            pygame.draw.circle(screen, RED, to_int_point(project_point_3d(end_point, current_world)), 5)
        else:
            pygame.draw.circle(screen, RED, to_int_point(end_point), 5)
    elif current_task == TASK_TRIANGLE:
        for p in triangle_points:
            if is_3d_mode:
                pygame.draw.circle(screen, RED, to_int_point(project_point_3d(p, current_world)), 5)
            else:
                pygame.draw.circle(screen, RED, to_int_point(p), 5)
    elif current_task == TASK_CIRCLE and circle_center:
        if is_3d_mode:
            pygame.draw.circle(screen, RED, to_int_point(project_point_3d(circle_center, current_world)), 5)
        else:
            pygame.draw.circle(screen, RED, to_int_point(circle_center), 5)
    elif current_task == TASK_PENTAGON:
        for p in pentagon_points:
            if is_3d_mode:
                pygame.draw.circle(screen, RED, to_int_point(project_point_3d(p, current_world)), 5)
            else:
                pygame.draw.circle(screen, RED, to_int_point(p), 5)
    elif current_task == TASK_TESSELLATION:
        for p in tessellation_points:
            if is_3d_mode:
                pygame.draw.circle(screen, RED, to_int_point(project_point_3d(p, current_world)), 5)
            else:
                pygame.draw.circle(screen, RED, to_int_point(p), 5)

    if current_shape:
        try:
            if current_task == TASK_LINE:
                if is_3d_mode:
                    pygame.draw.circle(screen, BLUE, to_int_point(project_point_3d(start_point, current_world)), 5)
                    for i in range(len(current_shape) - 1):
                        start_pos = current_shape[i]
                        end_pos = current_shape[i + 1]
                        proj_start = project_point_3d(start_pos, current_world)
                        proj_end = project_point_3d(end_pos, current_world)
                        if proj_start and proj_end:
                            pygame.draw.line(screen, CYAN, to_int_point(proj_start), to_int_point(proj_end), 2)
                else:
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
                            log_message(
                                f"Invalid line segment at index {i}: start={repr(start_pos)}, end={repr(end_pos)}"
                            )
            elif current_task == TASK_TRIANGLE and len(current_shape) == 3:
                if is_3d_mode:
                    for i in range(3):
                        start_pos = project_point_3d(current_shape[i], current_world)
                        end_pos = project_point_3d(current_shape[(i + 1) % 3], current_world)
                        if start_pos and end_pos:
                            pygame.draw.line(screen, CYAN, to_int_point(start_pos), to_int_point(end_pos), 2)
                else:
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
                            log_message(
                                f"Invalid triangle segment at index {i}: start={repr(start_pos)}, end={repr(end_pos)}"
                            )
            elif current_task == TASK_CIRCLE:
                if is_3d_mode:
                    for i in range(len(current_shape) - 1):
                        start_pos = current_shape[i]
                        end_pos = current_shape[i + 1]
                        proj_start = project_point_3d(start_pos, current_world)
                        proj_end = project_point_3d(end_pos, current_world)
                        if proj_start and proj_end:
                            pygame.draw.line(screen, CYAN, to_int_point(proj_start), to_int_point(proj_end), 2)
                else:
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
                            log_message(
                                f"Invalid circle segment at index {i}: start={repr(start_pos)}, end={repr(end_pos)}"
                            )
            elif current_task == TASK_PENTAGON and len(current_shape) == 5:
                if is_3d_mode:
                    for i in range(5):
                        start_pos = project_point_3d(current_shape[i], current_world)
                        end_pos = project_point_3d(current_shape[(i + 1) % 5], current_world)
                        if start_pos and end_pos:
                            pygame.draw.line(screen, CYAN, to_int_point(start_pos), to_int_point(end_pos), 2)
                else:
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
                            log_message(
                                f"Invalid pentagon segment at index {i}: start={repr(start_pos)}, end={repr(end_pos)}"
                            )
            elif current_task == TASK_TESSELLATION:
                for triangle in current_shape:
                    if not isinstance(triangle, (list, tuple)) or len(triangle) != 3:
                        log_message(f"Invalid triangle in tessellation: {repr(triangle)}")
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
                            log_message(
                                f"Invalid tessellation segment: start={repr(start_pos)}, end={repr(end_pos)}"
                            )
        except Exception as e:
            log_message(f"Error in drawing shape: {repr(e)}, current_shape={repr(current_shape)}")

    process_logs()
    if time.time() - last_plot_update > 1:
        plot_surface = update_reward_plot()
        update_dashboard()  # Update the dashboard every second
        last_plot_update = time.time()
    if plot_surface:
        screen.blit(plot_surface, (WIDTH - 310, 10))
    if dashboard_surface:
        screen.blit(dashboard_surface, (WIDTH - 610, HEIGHT - 220))  # Display the dashboard

    pygame.display.flip()
    clock.tick(tick_rate if running_state == "running" else 5)

pygame.quit()

def store_transition(transition):
    """Store a transition in the replay memory."""
    n_step_buffer.append(transition)
    if len(n_step_buffer) < n_step:
        return
    cumulative_reward = sum([(gamma**i) * t[2] for i, t in enumerate(n_step_buffer)])
    state, action = n_step_buffer[0][0], n_step_buffer[0][1]
    next_state, done = n_step_buffer[-1][3], n_step_buffer[-1][4]
    memory.add(1.0, (state, action, cumulative_reward, next_state, done))
    n_step_buffer.pop(0)

def plan_actions(state_tensor, model, current_shape, target, task, world, steps=3):
    """Plan a sequence of actions using a simple simulation."""
    best_sequence = []
    best_reward = -float("inf")

    for _ in range(10):  # Try 10 random sequences
        sequence = [np.random.randint(action_dim) for _ in range(steps)]
        sim_shape = current_shape.copy() if isinstance(current_shape, list) else []
        total_reward = 0

        for action in sequence:
            # Simulate the action
            if task == TASK_LINE:
                curvature = 0
                if action == 0:
                    curvature = -1
                elif action == 1:
                    curvature = 1
                if sim_shape:
                    next_segment = draw_line_segment(sim_shape[-1], target, curvature, world)
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
                sim_shape, sim_radius = draw_circle(target[0], target[1], radius_adjust, world)
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
                    base_points = sim_shape[0] if sim_shape and isinstance(sim_shape[0], (list, tuple)) else [(0, 0), (50, 50), (100, 0)]
                    sim_shape = draw_tessellation(base_points, world)

            # Calculate reward
            reward = calculate_advanced_reward(sim_shape, target, task, world)
            total_reward += reward

        if total_reward > best_reward:
            best_reward = total_reward
            best_sequence = sequence

    return best_sequence
