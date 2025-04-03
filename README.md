# GeoMaster AI Challenge

## Purpose
The **GeoMaster AI Challenge** is an interactive Python application designed to explore the intersection of artificial intelligence, reinforcement learning, and non-Euclidean geometry. The primary purpose of this project is to create an AI agent that learns to draw geometric shapes (e.g., lines, triangles, circles, pentagons, and tessellations) in various geometric worlds (Euclidean, Spherical, Hyperbolic, Elliptical, Projective, and Fractal) using a Deep Q-Network (DQN) enhanced with advanced techniques like Prioritized Experience Replay, N-Step Returns, and curiosity-driven exploration via a forward model.

This project serves as both an educational tool and a research platform:
- **Educational Tool**: It visually demonstrates how AI can adapt to different geometric constraints and tasks, making it accessible for learning about reinforcement learning and geometry.
- **Research Platform**: It provides a framework to experiment with advanced RL techniques, such as model-based planning, macro actions, and intrinsic motivation, in a visually engaging environment.

The AI can operate in two modes—**Autonomous** (fully AI-driven) and **Guidance** (player-assisted)—allowing users to either observe the AI's learning process or guide it interactively. The project logs the AI's performance to a CSV file for analysis, making it suitable for studying learning dynamics over time.

---

## Features
- **Geometric Worlds**: Supports drawing in six distinct geometric spaces with unique projections (e.g., stereographic for Spherical, Poincaré disk for Hyperbolic).
- **Tasks**: The AI can draw lines, triangles, circles, pentagons, and tessellations, each with world-specific reward functions.
- **AI Techniques**:
  - Deep Q-Network (DQN) with a target network.
  - Prioritized Experience Replay using a SumTree for efficient sampling.
  - N-Step Returns for improved reward propagation.
  - Curiosity-driven exploration with a forward model.
  - Model-based planning for multi-step action sequences.
  - Macro actions for complex tasks like triangles and circles.
- **Visualization**: Animated wireframes with gradient colors to represent each geometric world.
- **Modes**: Autonomous mode for pure AI operation and Guidance mode for player interaction.
- **Logging**: Saves AI decisions, rewards, and learning progress to a CSV file (`ai_calculations.csv`).

---

## Requirements
To run this project, you need the following Python libraries:
- `pygame` (>= 2.0.0): For rendering the game window and handling user input.
- `numpy` (>= 1.19.0): For numerical computations and array operations.
- `torch` (>= 1.9.0): For building and training neural networks (DQN, Forward Model, World Model).
- `csv`: Standard library for logging data (included with Python).
- `os`: Standard library for file operations (included with Python).

Install the required packages using pip:
```bash
pip install pygame numpy torch
```

---

## File Structure
- `geomaster_ai_challenge.py`: The main script containing all game logic, AI models, and rendering code. It includes:
  - **Game Logic**: Handles user input, task progression, and geometric world transitions.
  - **AI Models**: Implements the DQN, Forward Model, and World Model for reinforcement learning and planning.
  - **Rendering**: Visualizes geometric shapes and AI actions in real-time using `pygame`.
- `ai_calculations.csv`: Generated log file storing AI steps, states, actions, rewards, and learning messages (created on first run).

---

## How to Run
1. **Clone or Download**: Obtain the source code by cloning this repository or downloading the script.
2. **Install Dependencies**: Ensure all required libraries are installed (see Requirements).
3. **Run the Script**: Execute the Python script in your terminal or IDE:
   ```bash
   python geomaster_ai_challenge.py
   ```
4. **Interact with the Game**:
   - **Set Points**: Click the mouse to define starting points based on the current task (e.g., 1 click for a line, 3 for a triangle).
   - **Guidance Mode**: Use arrow keys (Left, Right, Up, Down), Space, or 'V' to provide hints to the AI.
   - **Reset**: Press 'R' when the AI finishes a task to reset and cycle to the next task and world.
   - **Quit**: Close the window or press the 'X' button to exit.

---

## Usage
### Game Rules
1. **Set Points**: Click to place points based on the task:
   - Line: 1 point (AI draws to a fixed endpoint).
   - Triangle: 3 points.
   - Circle: 1 center point.
   - Pentagon: 5 points.
   - Tessellation: 3 points to start.
2. **AI Drawing**: The AI attempts to complete the shape in the current geometric world.
3. **Guidance Mode**: Use keyboard inputs to assist the AI:
   - Left/Right Arrows: Adjust curvature.
   - Up/Down Arrows: Adjust angles.
   - Space: Adjust radius.
   - V: Add a vertex.
4. **Progress**: Monitor the AI's performance via on-screen reward metrics and the CSV log.

---

## Technical Details
### AI Components
- **DQN**: A neural network with three fully connected layers (input: 10 state features, output: 6 actions).
- **State Space**: 10-dimensional vector including position, direction, task/world IDs, shape progress, vertices, angle, and distance to close.
- **Action Space**: 6 actions (curvature left/right, angle adjust left/right, radius adjust, add vertex).
- **Reward Function**: Task-specific rewards encouraging accurate shape completion, adjusted for each geometric world.
- **Forward Model**: Predicts the next state for curiosity-driven intrinsic rewards.
- **World Model**: Used for planning multi-step action sequences.

### Geometric Projections
- **Euclidean**: Simple translation to screen coordinates.
- **Spherical**: Stereographic projection from a 3D sphere.
- **Hyperbolic**: Poincaré disk projection with geodesic paths.
- **Elliptical**: Scaled elliptical mapping.
- **Projective**: Perspective projection with a fixed focal point.
- **Fractal**: Iterative random transformations for a fractal pattern.

---

## Limitations
- **Performance**: Requires a decent CPU/GPU for smooth operation, especially with CUDA-enabled PyTorch.
- **Complexity**: The AI may struggle with complex tasks (e.g., tessellations) without extensive training.
- **User Input**: Limited to mouse clicks and basic keyboard controls in Guidance mode.

---

## Future Improvements
- Add a menu to manually select tasks and worlds.
- Implement a pre-trained model option for faster demonstration.
- Enhance visualization with more detailed shape rendering or 3D views.
- Expand the action space for finer control over shapes.
- Integrate a GUI for real-time parameter tuning (e.g., learning rate, epsilon decay).

---

## License
This project is open-source and available under the [MIT License](LICENSE). Feel free to modify and distribute it as needed.

---

## Acknowledgments
- Built with inspiration from reinforcement learning research and geometric visualization techniques.