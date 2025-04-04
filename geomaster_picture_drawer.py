import pygame
import numpy as np
import torch
import cv2
from geomaster_ai_challenge import (
    project_point,
    WORLD_EUCLIDEAN,
    WORLD_HYPERBOLIC,
    WORLD_FRACTAL,
    GeoMasterAIModel,
    DQN,
)
import os
import csv
import time
import torch.nn as nn
import torch.optim as optim

# Initialize Pygame
pygame.init()

# Screen settings
WIDTH = 500
HEIGHT = 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("GeoMaster Picture Drawer")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
CYAN = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)

# Font
font = pygame.font.SysFont("monospace", 20)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Action space and state
action_dim = 36  # Expanded: 4 dirs (no draw) + 4 dirs x 4 colors x 3 stroke sizes
state_dim = 10  # Match 2D state from geomaster_ai_challenge.py

# Load pre-trained GeoMasterAIModel
geo_model = GeoMasterAIModel(state_dim, action_dim, gamma=0.99, base_lr=0.001).to(
    device
)
weights_path = "policy_net.pth"
if os.path.exists(weights_path):
    try:
        geo_model.policy_net.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path} into policy_net")
    except RuntimeError as e:
        print(
            f"Error loading weights: {e}. Action dim mismatch? Initializing randomly."
        )
else:
    print(f"No weights found at {weights_path}; initializing with random weights")
geo_model.eval()

# Initialize DQN model
dqn_model = DQN(state_dim, action_dim).to(device)
dqn_weights_path = "dqn_policy_net.pth"
if os.path.exists(dqn_weights_path):
    try:
        dqn_model.load_state_dict(torch.load(dqn_weights_path))
        print(f"Loaded DQN weights from {dqn_weights_path}")
    except RuntimeError as e:
        print(f"Error loading DQN weights: {e}. Initializing randomly.")
else:
    print(
        f"No DQN weights found at {dqn_weights_path}; initializing with random weights"
    )
dqn_model.eval()

# Geometric worlds
worlds = [WORLD_EUCLIDEAN, WORLD_HYPERBOLIC, WORLD_FRACTAL]


# Target images (RGB)
def create_stick_figure():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 20), 10, (255, 255, 0), -1)  # Yellow head
    cv2.line(img, (50, 30), (50, 60), (0, 255, 0), 2)  # Green body
    cv2.line(img, (50, 40), (30, 50), (0, 0, 255), 2)  # Red left arm
    cv2.line(img, (50, 40), (70, 50), (0, 0, 255), 2)  # Red right arm
    cv2.line(img, (50, 60), (40, 80), (255, 0, 0), 2)  # Blue left leg
    cv2.line(img, (50, 60), (60, 80), (255, 0, 0), 2)  # Blue right leg
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


def create_tree():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (45, 60), (55, 100), (0, 165, 255), -1)  # Brown trunk
    cv2.circle(img, (50, 40), 20, (0, 255, 0), -1)  # Green crown
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


def load_png(filename):
    if os.path.exists(filename):
        img = cv2.imread(filename)  # BGR
        if img is None:
            raise ValueError(f"Failed to load {filename}")
        return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    else:
        print(f"Warning: {filename} not found. Skipping this image.")
        return None


def create_placeholder_image():
    """Create a placeholder image if a target image is missing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.putText(img, "MISSING", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


target_images = [
    ("stick_figure", create_stick_figure()),
    ("tree", create_tree()),
]

# Attempt to load custom_star only if the file exists, otherwise use a placeholder
custom_star = load_png("star.png")
if custom_star is None:
    print("Using placeholder for missing 'star.png'")
    custom_star = create_placeholder_image()
target_images.append(("custom_star", custom_star))

# Logging
log_file = "picture_drawer_log.csv"
file_exists = os.path.isfile(log_file)
with open(log_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(
            [
                "Image",
                "World",
                "Step",
                "Pen_X",
                "Pen_Y",
                "Action",
                "Reward",
                "Similarity",
                "Color_R",
                "Color_G",
                "Color_B",
                "Stroke_Size",
            ]
        )


# State function
def get_state(canvas, pen_pos, target_img):
    patch_size = 20
    x, y = int(pen_pos[0]), int(pen_pos[1])
    x_min, x_max = max(0, x - patch_size), min(WIDTH, x + patch_size)
    y_min, y_max = max(0, y - patch_size), min(HEIGHT, y + patch_size)
    canvas_patch = canvas[y_min:y_max, x_min:x_max]
    target_patch = target_img[y_min:y_max, x_min:x_max]
    similarity = -np.mean((canvas_patch - target_patch) ** 2) / 255.0
    state = np.array(
        [
            pen_pos[0] / WIDTH,
            pen_pos[1] / HEIGHT,
            similarity,
            np.mean(canvas_patch[:, :, 0]) / 255,
            np.mean(canvas_patch[:, :, 1]) / 255,
            np.mean(canvas_patch[:, :, 2]) / 255,
            np.mean(target_patch[:, :, 0]) / 255,
            np.mean(target_patch[:, :, 1]) / 255,
            np.mean(target_patch[:, :, 2]) / 255,
            np.std(canvas_patch) / 255,
        ]
    )
    return state


# Action map
action_map = {
    0: (-5, 0, False, (0, 0, 0), 1),
    1: (5, 0, False, (0, 0, 0), 1),
    2: (0, -5, False, (0, 0, 0), 1),
    3: (0, 5, False, (0, 0, 0), 1),
    4: (-5, 0, True, CYAN, 1),
    5: (5, 0, True, CYAN, 1),
    6: (0, -5, True, CYAN, 1),
    7: (0, 5, True, CYAN, 1),
    8: (-5, 0, True, CYAN, 3),
    9: (5, 0, True, CYAN, 3),
    10: (0, -5, True, CYAN, 3),
    11: (0, 5, True, CYAN, 3),
    12: (-5, 0, True, CYAN, 5),
    13: (5, 0, True, CYAN, 5),
    14: (0, -5, True, CYAN, 5),
    15: (0, 5, True, CYAN, 5),
    16: (-5, 0, True, GREEN, 1),
    17: (5, 0, True, GREEN, 1),
    18: (0, -5, True, GREEN, 1),
    19: (0, 5, True, GREEN, 1),
    20: (-5, 0, True, GREEN, 3),
    21: (5, 0, True, GREEN, 3),
    22: (0, -5, True, GREEN, 3),
    23: (0, 5, True, GREEN, 3),
    24: (-5, 0, True, GREEN, 5),
    25: (5, 0, True, GREEN, 5),
    26: (0, -5, True, GREEN, 5),
    27: (0, 5, True, GREEN, 5),
    28: (-5, 0, True, RED, 1),
    29: (5, 0, True, RED, 1),
    30: (0, -5, True, RED, 1),
    31: (0, 5, True, RED, 1),
    32: (-5, 0, True, RED, 3),
    33: (5, 0, True, RED, 3),
    34: (0, -5, True, RED, 3),
    35: (0, 5, True, RED, 3),
}


# Reward function
def calculate_reward(canvas, target_img):
    mse = np.mean((canvas - target_img) ** 2) / 255.0
    return -mse


# Load previous data for action bias
def load_previous_data(image_name, world):
    action_rewards = {i: [] for i in range(action_dim)}
    previous_data = []
    if os.path.exists(log_file):
        with open(log_file, mode="r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Image"] == image_name and row["World"] == world:
                    action = int(row["Action"])
                    reward = float(row["Reward"])
                    previous_data.append(
                        {
                            "Pen_X": float(row["Pen_X"]),
                            "Pen_Y": float(row["Pen_Y"]),
                            "Action": action,
                            "Reward": reward,
                        }
                    )
                    action_rewards[action].append(reward)
    action_bias = {
        i: np.mean(rewards) if rewards else 0 for i, rewards in action_rewards.items()
    }
    return previous_data, action_bias


# Drawing function
def apply_action(pen_pos, action, canvas, world):
    dx, dy, draw, color, stroke_size = action_map[action]
    new_x = max(0, min(WIDTH - 1, pen_pos[0] + dx))
    new_y = max(0, min(HEIGHT - 1, pen_pos[1] + dy))
    new_pos = [new_x, new_y]

    proj_start = project_point((pen_pos[0] - WIDTH / 2, pen_pos[1] - HEIGHT / 2), world)
    proj_end = project_point((new_x - WIDTH / 2, new_y - HEIGHT / 2), world)

    if draw:
        pygame.draw.line(screen, color, proj_start, proj_end, stroke_size)
        cv2.line(
            canvas,
            (int(pen_pos[0]), int(pen_pos[1])),
            (int(new_x), int(new_y)),
            color[::-1],
            stroke_size,
        )

    return new_pos, color if draw else (0, 0, 0), stroke_size


# Image Generator
class ImageGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output [0, 1]
        return x


# Initialize ImageGenerator
input_dim = state_dim
output_dim = WIDTH * HEIGHT * 3
image_generator = ImageGenerator(input_dim, output_dim).to(device)


# Training loop for ImageGenerator
def train_image_generator(generator, target_images, epochs=100, lr=0.001):
    optimizer = optim.Adam(generator.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Training ImageGenerator...")
    for epoch in range(epochs):
        total_loss = 0
        for img_name, target_img in target_images:
            # Generate random state-like input (simulating pen position and canvas state)
            pen_pos = [np.random.uniform(0, WIDTH), np.random.uniform(0, HEIGHT)]
            canvas = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
            state = get_state(canvas, pen_pos, target_img)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Target image as ground truth (ensure same shape as input)
            target_tensor = torch.FloatTensor(target_img.flatten() / 255.0).unsqueeze(0).to(device)

            # Forward pass
            optimizer.zero_grad()
            output = generator(state_tensor)
            loss = criterion(output, target_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(target_images)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.6f}")

    # Save trained model
    torch.save(generator.state_dict(), "image_generator.pth")
    print("ImageGenerator training complete. Weights saved to 'image_generator.pth'")


# Load or train ImageGenerator
generator_weights_path = "image_generator.pth"
if os.path.exists(generator_weights_path):
    image_generator.load_state_dict(torch.load(generator_weights_path))
    print(f"Loaded ImageGenerator weights from {generator_weights_path}")
else:
    train_image_generator(image_generator, target_images)
image_generator.eval()


# Generate initial canvas
def generate_initial_canvas(state):
    input_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = image_generator(input_tensor)
    output_image = output_tensor.cpu().numpy().reshape((HEIGHT, WIDTH, 3)) * 255
    return output_image.astype(np.uint8)


# Action selection
def select_action(state, model, action_bias, bias_strength):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor).cpu().numpy().flatten()
        for action, bias in action_bias.items():
            q_values[action] += bias * bias_strength
    return np.argmax(q_values)


if __name__ == '__main__':
    clock = pygame.time.Clock()
    running = True
    max_steps = 1000
    bias_strength = 0.5
    use_dqn = True  # Default to DQN model

    for img_name, target_img in target_images:
        for current_world in worlds:
            if not running:
                break

            # Reset for new image/world
            canvas = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
            pen_pos = [WIDTH // 2, HEIGHT // 2]
            target_surface = pygame.surfarray.make_surface(
                cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            )
            step = 0
            total_reward = 0

            # Load previous data and bias
            previous_data, action_bias = load_previous_data(img_name, current_world)
            if previous_data:
                avg_reward = np.mean([d["Reward"] for d in previous_data])
                if avg_reward > -50:
                    pen_pos = [previous_data[0]["Pen_X"], previous_data[0]["Pen_Y"]]
                    print(f"Reusing starting position for {img_name} in {current_world}")

            # Generate initial canvas
            initial_state = get_state(canvas, pen_pos, target_img)
            canvas = generate_initial_canvas(initial_state)

            while step < max_steps and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_q
                    ):
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_u:
                            bias_strength = min(1.0, bias_strength + 0.1)
                            print(f"Bias strength increased to {bias_strength:.1f}")
                        if event.key == pygame.K_d:
                            bias_strength = max(0.1, bias_strength - 0.1)
                            print(f"Bias strength decreased to {bias_strength:.1f}")
                        if event.key == pygame.K_m:
                            use_dqn = not use_dqn
                            model_name = "DQN" if use_dqn else "PolicyNet"
                            print(f"Switched to {model_name} for action selection")

                # Get state
                state = get_state(canvas, pen_pos, target_img)

                # Choose action based on selected model
                model = dqn_model if use_dqn else geo_model.policy_net
                action = select_action(state, model, action_bias, bias_strength)

                # Apply action
                pen_pos, color, stroke_size = apply_action(
                    pen_pos, action, canvas, current_world
                )
                reward = calculate_reward(canvas, target_img)
                total_reward += reward

                # Log data
                similarity = -np.mean((canvas - target_img) ** 2) / 255.0
                with open(log_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            img_name,
                            current_world,
                            step,
                            pen_pos[0],
                            pen_pos[1],
                            action,
                            reward,
                            similarity,
                            *color,
                            stroke_size,
                        ]
                    )

                # Drawing
                screen.fill(BLACK)
                screen.blit(target_surface, (0, 0))
                canvas_surface = pygame.surfarray.make_surface(
                    cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                )
                screen.blit(canvas_surface, (0, 0))

                # Info
                model_name = "DQN" if use_dqn else "PolicyNet"
                info_text = [
                    f"Image: {img_name}",
                    f"World: {current_world}",
                    f"Step: {step}/{max_steps}",
                    f"Reward: {reward:.2f}",
                    f"Total Reward: {total_reward:.2f}",
                    f"Bias Strength: {bias_strength:.1f}",
                    f"Model: {model_name}",
                ]
                for i, line in enumerate(info_text):
                    text = font.render(line, True, WHITE)
                    screen.blit(text, (10, 10 + i * 30))

                pygame.display.flip()
                clock.tick(60)
                step += 1

            # Save output
            if running:
                output_filename = f"{img_name}_{current_world}_{int(time.time())}.png"
                cv2.imwrite(output_filename, canvas)
                print(f"Saved drawing to {output_filename}")

    pygame.quit()
