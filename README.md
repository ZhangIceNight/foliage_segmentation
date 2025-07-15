# Dynamic Hypergraph-guided Mamba for TLS Point Cloud Foliage Separation (TGRS-2025)

Brief introduction about the project background, research goals, and significance.

---


## Model Architecture

[overall-architecture](figures/overall-architecture.pdf)

---

## Experimental Results

- Experimental rusults on three single-scale datasets

| Method (metric: mIoU)    | Tropical           |                   | Mixed Forest      |                   | ForestSemantic    |                   |
|--------------------------|--------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|                          | Mean               | (±std, p)         | Mean              | (±std, p)         | Mean              | (±std, p)         |
| LeWoS                    | 0.7321             | ±1.348, 7.92e-4   | 0.7182            | ±1.457, 8.15e-4   | 0.6194            | ±1.552, 8.23e-4   |
| PointNeXt                | 0.8134             | ±1.243, 1.63e-3   | 0.7893            | ±1.354, 1.28e-3   | 0.6783            | ±1.445, 1.15e-3   |
| PointNet++               | 0.7947             | ±1.342, 1.42e-3   | 0.7682            | ±1.453, 1.87e-3   | 0.6592            | ±1.544, 1.69e-3   |
| PointTransformer         | 0.8212             | ±1.234, 1.21e-3   | 0.7843            | ±1.345, 1.38e-3   | 0.6853            | ±1.448, 1.27e-3   |
| PointTransformerV2       | 0.8423             | ±1.241, 1.52e-3   | 0.8052            | ±1.332, 1.79e-3   | 0.7064            | ±1.443, 1.58e-3   |
| PointCloudMamba          | 0.8329             | ±1.212, 1.04e-3   | 0.8104            | ±1.351, 1.33e-3   | 0.7022            | ±1.445, 1.66e-3   |
| Mamba3D                  | 0.8501             | ±1.522, 1.84e-3   | 0.8155            | ±1.209, 1.88e-3   | 0.7092            | ±1.254, 1.93e-3   |
| Sen-Net                  | 0.8289             | ±1.417, 1.64e-3   | 0.7921            | ±1.527, 1.62e-3   | 0.6744            | ±1.329, 1.83e-3   |
| **DHMamba (Ours)**       | **0.8546**†        | ±1.195, -         | **0.8237**†       | ±1.304, -         | **0.7273**†       | ±1.348, -         |

- Experimental results on four plot-scale datasets

| Method               | Birch           |                | Larch           |                | CST             |                | Evo MLS         |                |
|----------------------|-----------------|----------------|-----------------|----------------|-----------------|----------------|-----------------|----------------|
|                      | Mean            | (±std, p)      | Mean            | (±std, p)      | Mean            | (±std, p)      | Mean            | (±std, p)      |
| LeWoS                | 0.4252          | ±1.543, 7.92e-4| 0.4374          | ±1.552, 8.15e-4| 0.4193          | ±1.544, 7.89e-4| 0.4534          | ±1.647, 8.24e-4|
| PointNeXt            | 0.5143          | ±1.448, 1.18e-3| 0.5262          | ±1.445, 1.27e-3| 0.5012          | ±1.435, 1.16e-3| 0.5443          | ±1.541, 1.22e-3|
| PointNet++           | 0.4854          | ±1.442, 1.32e-3| 0.4973          | ±1.447, 1.48e-3| 0.4723          | ±1.443, 1.35e-3| 0.5162          | ±1.534, 1.39e-3|
| PCT                  | 0.5013          | ±1.435, 1.29e-3| 0.5132          | ±1.444, 1.38e-3| 0.4884          | ±1.442, 1.26e-3| 0.5324          | ±1.538, 1.34e-3|
| PCTV2                | 0.5423          | ±1.441, 1.42e-3| 0.5542          | ±1.446, 1.57e-3| 0.5292          | ±1.434, 1.45e-3| 0.5732          | ±1.535, 1.49e-3|
| PointCloudMamba      | 0.5446          | ±1.610, 1.52e-3| 0.5478          | ±1.523, 2.31e-3| 0.5211          | ±1.677, 3.84e-2| 0.5706          | ±1.734, 1.10e-2|
| Mamba3D              | 0.5581          | ±1.588, 2.73e-3| 0.5520          | ±1.465, 1.90e-3| 0.5312          | ±1.641, 2.91e-2| 0.5846          | ±1.702, 4.12e-2|
| Sen-Net              | 0.5304          | ±1.721, 7.65e-4| 0.5215          | ±1.805, 1.33e-3| 0.5661          | ±1.643, 4.77e-2| 0.5719          | ±1.790, 3.54e-2|
| **DHMamba (Ours)**   | **0.5711**†     | ±1.384, -      | **0.5852**†     | ±1.386, -      | **0.5342**†     | ±1.392, -      | **0.6061**†     | ±1.448, -      |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yourproject.git
cd yourproject

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
