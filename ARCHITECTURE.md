# EMAP: 3D Neural Edge Reconstruction - Architecture

## Overview
EMAP (3D Neural Edge Reconstruction) is a framework designed to reconstruct 3D edges from multi-view 2D edge maps. It learns an Unsigned Distance Field (UDF) representation of 3D edges through volume rendering, allowing for the extraction of parametric 3D edges (lines and curves).

## Directory Structure

- `main.py`: The entry point for training and extracting edges. It parses configurations and initializes the appropriate runner.
- `confs/`: Configuration files in HOCON format defining dataset, model, and training hyperparameters.
- `scripts/`: Shell scripts for data downloading and running experiments on various datasets (ABC, DTU, Replica).
- `src/`: The core source code directory.
  - `dataset/`: Data loading and ray generation utilities.
  - `edge_extraction/`: Code for extracting point clouds from UDFs and fitting parametric curves/lines.
  - `eval/`: Evaluation scripts.
  - `models/`: Neural network definitions for UDF and volume rendering.
  - `runner/`: Training and inference loops.
  - `utils/`: Assorted math, plotting, and visualization utilities.

## Core Components

### 1. Data Processing (`src/dataset/`)
The `Dataset` class handles loading multi-view images, camera intrinsics/extrinsics, and pre-extracted 2D edge maps (using DexiNed or PidiNet). It generates rays for volume rendering during training and validation.

### 2. Network Models (`src/models/`)
The underlying representation is an Unsigned Distance Field (UDF).
- **`UDFNetwork` (`udf_model.py`)**: An MLP that takes 3D point coordinates (optionally with positional encoding) and predicts the UDF value and geometric features. 
- **`BetaNetwork` & `SingleVarianceNetwork` (`udf_model.py`)**: Auxiliary networks used to learn parameters (variance, beta, gamma) that map UDF values to occlusion probabilities and volume densities.
- **`RenderingNetwork`**: Predicts colors or features (if color rendering is needed).
- **`UDFRendererBlending` (`udf_renderer_blending.py`)**: The core volume rendering engine. It implements the conversion from UDF values to alpha/density values. It uses unbiased upsampling near the surface and integrates the density along rays to render 2D edge maps.

### 3. Training and Inference (`src/runner/`)
The execution logic is managed by `Runner_UDF`, which inherits from `Runner`.
- **Training (`train_udf`)**: Optimizes the UDF network by rendering 2D edges and minimizing the discrepancy with ground truth 2D edge maps. It incorporates an Eikonal loss (`gradient_error_loss`) to enforce valid distance fields and a sparsity regularization.
- **Validation (`validate`)**: Periodically renders depth, edges, and normals to visual formats for monitoring.

### 4. Edge Extraction (`src/edge_extraction/`)
Once the UDF is trained, 3D edges are extracted in two stages:
- **Point Cloud Extraction (`extract_pointcloud.py`)**: Evaluates the UDF on a dense 3D grid. Points where the UDF value falls below a specific threshold are selected. Points are iteratively shifted towards the local UDF zero-level set using gradients.
- **Parametric Edge Fitting (`extract_parametric_edge.py`)**: Processes the dense point cloud to fit 3D line segments and Bezier curves. It includes an edge merging strategy and optional visibility checking (projecting edges back to 2D views to filter out occluded or spurious edges).

## Execution Flow
1. **Configuration**: `main.py` reads `confs/*.conf` parameters.
2. **Initialization**: Creates dataset, initializes `UDFNetwork` and `UDFRendererBlending`, and sets up the Adam optimizer.
3. **Training Loop**: For each iteration, rays are sampled, UDF is queried, edges are rendered, and the networks are updated based on the edge loss and geometric regularizations.
4. **Extraction**: When called with `--mode extract_edge`, it loads the trained checkpoint, generates the edge point cloud, fits parametric lines/curves, and saves them as `.ply` and `.json` files.
