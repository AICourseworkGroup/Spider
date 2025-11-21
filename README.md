# Spider Gait Generation Project/Coursework

This project uses genetic algorithms and neural networks to generate walking gaits for an 8-legged spider robot. It compares two neural network implementations: a custom implementation from scratch and PyTorch.

## Project Structure

```
Spider/
├── main.py                          # Main program orchestrating all steps
├── genetic_algorithm.py             # GA implementation for pose generation
├── neural_network_from_prac5.py     # Custom neural network from scratch - this was based off the code in AI Practical 5
├── pytorch.py                       # PyTorch neural network implementation
├── forward_leg_kinematics.py        # 3D forward kinematics for spider legs - given to us as part of the assignment in matlab. was translated to python using gemini 2.5
├── plot_spider_pose.py              # 3D visualization of spider poses - given to us as part of the assignment in matlab. was translated to python using gemini 2.5
└── README.md                        # This file
```

## Requirements

```bash
# Core dependencies
numpy
matplotlib
torch
pytorch-lightning
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AICourseworkGroup/Spider.git
cd Spider
```

2. Create and activate a conda environment:
```bash
conda create -n spider python=3.11
conda activate spider
```

3. Install dependencies:
```bash
pip install numpy matplotlib torch pytorch-lightning
```

## Usage

### Run the Full Pipeline

```bash
python main.py
```

This will execute the following steps:

1. **Display Target Chromosomes**: Shows the two key poses (standing and mid-walk)
2. **Ask to Animate Target Chromosomes**: User chooses whether to view the full 300-frame walk cycle
3. **Run Genetic Algorithm**: Prompts the user for parameters (generations, population size, mutation rate), then evolves poses to match targets
4. **Animate GA Poses**: Displays the GA-generated walk cycle
5. **Train Custom Neural Network**: Trains the from-scratch implementation
6. **Display Custom NN Input**: Shows a random test pose
7. **Display Custom NN Output**: Shows the network's prediction
8. **Train PyTorch Neural Network**: Trains the PyTorch implementation
9. **Display PyTorch NN Input**: Shows a random test pose
10. **Display PyTorch NN Output**: Shows the network's prediction

## Credits

- **Forward kinematics (forward_leg_kinematics.py) and plotting functions (plot_spider_pose.py)**: Given to us as part of the assignment in MATLAB, translated to Python using Gemini 2.5 Pro
- **Custom neural network (neural_network_from_prac5.py)**: Based on code from AI Practical 5
- **GitHub Copilot**: Used for syntax help and code autocomplete throughout development

## Repository

https://github.com/AICourseworkGroup/Spider

## Assignment Related Documentation

https://portdotacdotuk-my.sharepoint.com/:w:/g/personal/up2125866_myport_ac_uk/IQBYUY9xRLXxSqQvPgOamz4JAZE6rQXyJORW1xC-d_sptI8?e=dftlG4