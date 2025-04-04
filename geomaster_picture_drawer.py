import pygame
import numpy as np
import torch
import cv2
from geomaster_ai_challenge import project_point, WORLD_EUCLIDEAN, WORLD_HYPERBOLIC, WORLD_FRACTAL, GeoMasterAIModel
import os
import csv
import time
import torch.nn as nn

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

# Load pre-trained GeoMasterAIModel
state_dim = 10  # Match 2D state from geomaster_ai_challenge.py
action_dim = 36  # Expanded: 4 dirs (no draw) + 4 dirs x 4 colors x 3 stroke sizes
geo_model = GeoMasterAIModel(state_dim, action_dim, gamma=0.99, base_lr=0.001).to(device)
weights_path = "policy_net.pth"
if os.path.exists(weights_path):
    try:
        # Load weights into policy_net directly
        geo_model.policy_net.load_state_dict(torch.load(weights_path))
        print(f"Loaded weights from {weights_path} into policy_net")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        print("Initializing policy_net with random weights instead.")
else:
    print(f"No weights found at {weights_path}; initializing with random weights")
geo_model.eval()
policy_net = geo_model.policy_net  # Use the policy_net from GeoMasterAIModel

# Geometric worlds (subset from geomaster_ai_challenge.py)
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
        raise FileNotFoundError(f"{filename} not found")

target_images = [
    ("stick_figure", create_stick_figure()),
    ("tree", create_tree()),
    ("custom_star", load_png("star.png"))
]

# Logging
log_file = "picture_drawer_log.csv"
file_exists = os.path.isfile(log_file)
with open(log_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Image", "World", "Step", "Pen_X", "Pen_Y", "Action", "Reward", "Similarity", "Color_R", "Color_G", "Color_B", "Stroke_Size"])

# State and actions
def get_state(canvas, pen_pos, target_img):
    patch_size = 20
    x, y = int(pen_pos[0]), int(pen_pos[1])
    x_min, x_max = max(0, x - patch_size), min(WIDTH, x + patch_size)
    y_min, y_max = max(0, y - patch_size), min(HEIGHT, y + patch_size)
    canvas_patch = canvas[y_min:y_max, x_min:x_max]
    target_patch = target_img[y_min:y_max, x_min:x_max]
    similarity = -np.mean((canvas_patch - target_patch) ** 2) / 255.0
    state = np.array([
        pen_pos[0] / WIDTH, pen_pos[1] / HEIGHT,
        similarity, 0, 0,
        0, 0, 0, 0, 0  # Padded to match state_dim=10
    ])
    return state

action_map = {
    # No draw (4 actions)
    0: (-5, 0, False, (0, 0, 0), 1),  1: (5, 0, False, (0, 0, 0), 1),
    2: (0, -5, False, (0, 0, 0), 1),  3: (0, 5, False, (0, 0, 0), 1),
    
    # Cyan (1, 3, 5)
    4: (-5, 0, True, CYAN, 1),  5: (5, 0, True, CYAN, 1),
    6: (0, -5, True, CYAN, 1),  7: (0, 5, True, CYAN, 1),
    8: (-5, 0, True, CYAN, 3),  9: (5, 0, True, CYAN, 3),
    10: (0, -5, True, CYAN, 3), 11: (0, 5, True, CYAN, 3),
    12: (-5, 0, True, CYAN, 5), 13: (5, 0, True, CYAN, 5),
    14: (0, -5, True, CYAN, 5), 15: (0, 5, True, CYAN, 5),
    
    # Green (1, 3, 5)
    16: (-5, 0, True, GREEN, 1), 17: (5, 0, True, GREEN, 1),
    18: (0, -5, True, GREEN, 1), 19: (0, 5, True, GREEN, 1),
    20: (-5, 0, True, GREEN, 3), 21: (5, 0, True, GREEN, 3),
    22: (0, -5, True, GREEN, 3), 23: (0, 5, True, GREEN, 3),
    24: (-5, 0, True, GREEN, 5), 25: (5, 0, True, GREEN, 5),
    26: (0, -5, True, GREEN, 5), 27: (0, 5, True, GREEN, 5),
    
    # Red (1, 3, 5)
    28: (-5, 0, True, RED, 1),  29: (5, 0, True, RED, 1),
    30: (0, -5, True, RED, 1),  31: (0, 5, True, RED, 1),
    32: (-5, 0, True, RED, 3),  33: (5, 0, True, RED, 3),
    34: (0, -5, True, RED, 3),  35: (0, 5, True, RED, 3),
    # Red stroke 5 and Blue actions can be added by expanding action_dim to 48
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
                    previous_data.append({
                        "Pen_X": float(row["Pen_X"]), "Pen_Y": float(row["Pen_Y"]),
                        "Action": action, "Reward": reward
                    })
                    action_rewards[action].append(reward)
    action_bias = {i: np.mean(rewards) if rewards else 0 for i, rewards in action_rewards.items()}
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
        cv2.line(canvas, (int(pen_pos[0]), int(pen_pos[1])), 
                 (int(new_x), int(new_y)), color[::-1], stroke_size)
    
    return new_pos, color if draw else (0, 0, 0), stroke_size

# Define a simple neural network for image generation
class ImageGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageGenerator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output values between 0 and 1
        return x

# Initialize the image generator
input_dim = 10  # Example input dimension
output_dim = WIDTH * HEIGHT * 3  # Output dimension for an RGB image
image_generator = ImageGenerator(input_dim, output_dim).to(device)

# Generate an image using the neural network
def generate_image(input_vector):
    input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(device)
    with torch.no_grad():
        output_tensor = image_generator(input_tensor)
    output_image = output_tensor.cpu().numpy().reshape((HEIGHT, WIDTH, 3)) * 255
    return output_image.astype(np.uint8)

# Example usage of the image generator
input_vector = np.random.rand(input_dim)  # Random input vector
generated_image = generate_image(input_vector)

# Save the generated image
output_filename = "generated_image.png"
cv2.imwrite(output_filename, generated_image)
print(f"Generated image saved to {output_filename}")

# Main loop
clock = pygame.time.Clock()
running = True
max_steps = 1000
bias_strength = 0.5  # Starting value

for img_name, target_img in target_images:
    for current_world in worlds:
        if not running:
            break
        
        # Reset for new image/world
        canvas = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
        pen_pos = [WIDTH // 2, HEIGHT // 2]
        target_surface = pygame.surfarray.make_surface(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
        step = 0
        total_reward = 0
        
        # Load previous data and bias actions
        previous_data, action_bias = load_previous_data(img_name, current_world)
        if previous_data:
            avg_reward = np.mean([d["Reward"] for d in previous_data])
            if avg_reward > -50:
                pen_pos = [previous_data[0]["Pen_X"], previous_data[0]["Pen_Y"]]
                print(f"Reusing starting position for {img_name} in {current_world}")
        
        while step < max_steps and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_u:
                        bias_strength = min(1.0, bias_strength + 0.1)
                        print(f"Bias strength increased to {bias_strength:.1f}")
                    if event.key == pygame.K_d:
                        bias_strength = max(0.1, bias_strength - 0.1)
                        print(f"Bias strength decreased to {bias_strength:.1f}")

            # Get state
            state = get_state(canvas, pen_pos, target_img)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            # Choose action with bias
            with torch.no_grad():
                q_values = policy_net(state_tensor).cpu().numpy().flatten()
                for action, bias in action_bias.items():
                    q_values[action] += bias * bias_strength
                action = np.argmax(q_values)

            # Apply action
            pen_pos, color, stroke_size = apply_action(pen_pos, action, canvas, current_world)
            reward = calculate_reward(canvas, target_img)
            total_reward += reward

            # Log data
            similarity = -np.mean((canvas - target_img) ** 2) / 255.0
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([img_name, current_world, step, pen_pos[0], pen_pos[1], action, reward, similarity, *color, stroke_size])

            # Drawing
            screen.fill(BLACK)
            screen.blit(target_surface, (0, 0))
            canvas_surface = pygame.surfarray.make_surface(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            screen.blit(canvas_surface, (0, 0))

            # Info
            info_text = [
                f"Image: {img_name}",
                f"World: {current_world}",
                f"Step: {step}/{max_steps}",
                f"Reward: {reward:.2f}",
                f"Total Reward: {total_reward:.2f}",
                f"Bias Strength: {bias_strength:.1f}"
            ]
            for i, line in enumerate(info_text):
                text = font.render(line, True, WHITE)
                screen.blit(text, (10, 10 + i * 30))

            pygame.display.flip()
            clock.tick(60)
            step += 1

        # Save output as PNG
        if running:
            output_filename = f"{img_name}_{current_world}_{int(time.time())}.png"
            cv2.imwrite(output_filename, canvas)  # Save the canvas as an image file
            print(f"Saved drawing to {output_filename}")

pygame.quit()