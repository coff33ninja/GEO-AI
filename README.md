# GeoMaster AI Challenge

## Purpose
The **GeoMaster AI Challenge** is an interactive Python application that explores the intersection of artificial intelligence, reinforcement learning, and non-Euclidean geometry. Its primary goal is to develop an AI agent that learns to draw geometric shapes—lines, triangles, circles, pentagons, and tessellations—across various geometric worlds: Euclidean, Spherical, Hyperbolic, Elliptical, Projective, and Fractal. The AI uses a Deep Q-Network (DQN) enhanced with techniques like Prioritized Experience Replay, N-Step Returns, and curiosity-driven exploration via a forward model. The project also includes a Next Shape Predictor to anticipate the next task based on performance history.

This project began as an idea and was developed with significant AI assistance from tools like OpenAI, GROK, and VSCode’s OpenAI o4 free tier, with my role being to expand functionality and experiment with enhancements. It remains a work in progress as I push the boundaries of everyday AI capabilities.

It serves two main purposes:
- **Educational Tool**: Visually demonstrates AI adaptation to geometric constraints, making it a resource for learning reinforcement learning and geometry.
- **Research Platform**: Provides a testbed for experimenting with advanced RL techniques in a dynamic, visual environment.

The AI currently operates in an **Autonomous** mode, with basic player interaction limited to starting, pausing, and adjusting parameters via keyboard commands. Performance is logged to a CSV file for analysis.

---

## Features
- **Geometric Worlds**: Supports six geometric spaces with unique projections:
  - Euclidean: Standard 2D/3D coordinates.
  - Spherical: Stereographic projection from a 3D sphere.
  - Hyperbolic: Poincaré disk projection with geodesic paths (for lines).
  - Elliptical: Scaled elliptical mapping.
  - Projective: Perspective projection.
  - Fractal: Iterative random transformations.
- **Tasks**: The AI draws:
  - Lines (to a fixed endpoint).
  - Triangles (three points).
  - Circles (around a center with adjustable radius).
  - Pentagons (five points).
  - Tessellations (multiple triangles from three base points).
- **AI Techniques**:
  - Deep Q-Network (DQN) with a target network.
  - Prioritized Experience Replay using a `SumTree`.
  - N-Step Returns (n=3) for improved reward propagation.
  - Curiosity-driven exploration via a `ForwardModel`.
  - Model-based planning with a `WorldModel` for multi-step action sequences.
  - Macro actions for tasks like triangles and circles.
  - Next Shape Predictor (`NextShapePredictor`) to forecast the next task.
  - Adaptive learning rate based on FPS and average reward.
- **Visualization**: Animated wireframes with gradient colors for each world, plus a real-time reward trend plot.
- **Modes**: Autonomous AI drawing with basic control (start, pause, resume, speed adjustments).
- **Logging**: Saves episode data (state, action, reward, etc.) to `ai_calculations.csv`.
- **3D Mode**: Toggleable 2D/3D drawing with perspective or orthographic projections.
- **Explainable AI (XAI)**: Visualizes action probabilities every 50 steps.

---

## Requirements
To run this project, you need:
- `pygame` (>= 2.0.0): For rendering and input handling.
- `numpy` (>= 1.19.0): For numerical operations.
- `torch` (>= 1.9.0): For neural network training.
- `matplotlib` (>= 3.0.0): For reward plotting.
- `csv` and `os`: Standard libraries (included with Python).

Install dependencies with:
```bash
pip install pygame numpy torch matplotlib
```

---

## File Structure
- `geomaster_ai_challenge.py`: Main script with game logic, AI models, and rendering.
- `ai_calculations.csv`: Log file for AI performance data (created on first run).
- Saved model files (e.g., `policy_net.pth`, `model_<task>_<world>.pth`): Generated during training.

---

## How to Run
1. **Clone or Download**: Obtain the source code.
2. **Install Dependencies**: Run the pip command above.
3. **Run the Script**: Execute:
   ```bash
   python geomaster_ai_challenge.py
   ```
4. **Interact with the Game**:
   - **S**: Start the AI.
   - **P**: Pause.
   - **R**: Resume.
   - **Q**: Quit.
   - **F**: Increase tick rate (faster).
   - **L**: Decrease tick rate (slower).
   - **U**: Increase training iterations.
   - **D**: Decrease training iterations.
   - **B**: Toggle debug mode.
   - **3**: Toggle 2D/3D mode.

The AI automatically cycles through tasks and worlds upon completion.

---

## Usage
### Game Rules
1. **Tasks Begin**: The AI starts drawing based on randomly set points for the current task.
2. **AI Drawing**: It attempts to complete the shape in the current world, adjusting curvature, angles, or radius as needed.
3. **Progress**: Monitor performance via on-screen metrics (reward, FPS) and the CSV log.

---

## Technical Details
### AI Components
- **DQN**: Three-layer network (input: 10 for 2D, 11 for 3D; output: 6 actions).
- **State Space**: 2D (10D) or 3D (11D) vector with position, direction, task/world IDs, shape progress, vertices, angle, and distance (or z-depth in 3D).
- **Action Space**: 6 actions (curvature left/right, angle adjust left/right, radius adjust, add vertex).
- **Reward Function**: Task-specific with efficiency and creativity factors, adjusted for world constraints.
- **Forward Model**: Predicts next state for intrinsic rewards.
- **World Model**: Plans multi-step actions.
- **Next Shape Predictor**: Predicts the next task using a 13D input (task, world, rewards).

### Geometric Projections
- **2D Projections**: As listed in Features.
- **3D Projections**: Perspective (default) or orthographic, toggleable with key `3`.

---

## Limitations
- **Performance**: Requires a decent CPU/GPU, especially in 3D mode or with CUDA.
- **Complexity**: Tessellations and 3D shapes may lack precision without extended training.
- **User Input**: Limited to basic controls; full Guidance mode (e.g., arrow keys) is not implemented.
- **Stability**: Some projections (e.g., Fractal) are simplistic and may produce unexpected results.

---

## Future Improvements
- Implement full Guidance mode with interactive controls (e.g., arrow keys for curvature/angle).
- Enhance 3D visualization with wireframe depth cues or shading.
- Refine tessellation logic for more complex patterns.
- Add a menu for task/world selection.
- Integrate pre-trained models for instant demonstrations.
- **Text-Based AI Enhancement**: Leverage text-based AI to provide real-time measurements (e.g., side lengths, angles, radii) or properties for shapes, tailored to the geometric world’s rules (e.g., geodesic distances in Hyperbolic space).

---

## License
This project is open-source under the [MIT License](LICENSE).

---

## Acknowledgments
- Inspired by RL research and geometric visualization techniques, with assistance from OpenAI, GROK, and VSCode AI tools.

---

“For future planning, I intend to integrate a text-based AI (e.g., a language model) to dynamically calculate and provide measurements—such as side lengths, angles, radii, or other geometric properties—for the shapes being drawn, customized to the specific geometric world (e.g., Euclidean distances, Hyperbolic geodesic lengths, or Spherical arc lengths). This would enhance user understanding and interaction.”
