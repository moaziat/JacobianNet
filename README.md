# Neural Network for Arakawa Jacobian Computation

A PyTorch-based implementation for learning and computing Arakawa Jacobians using neural networks. This project includes data generation, model training, and inference capabilities with a focus on conservation properties.

## Features

- Generate diverse fluid dynamics training data:
  - Turbulent fields
  - Vortex dipoles
  - Kelvin-Helmholtz instability
  - Mixed field combinations
- Memory-efficient dataset handling with HDF5
- Custom conservation-aware loss function
- Configurable neural network architecture
- Comprehensive training pipeline with:
  - Early stopping
  - Learning rate scheduling
  - Model checkpointing
  - Training visualization
  - Memory optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

# Install required packages
pip install torch h5py matplotlib numpy psutil
```

## Usage

### 1. Generate Training Data

```bash
python compute_jacobian.py
```

This script will:
- Generate diverse field pairs (turbulent, vortex, shear, mixed)
- Compute corresponding Arakawa Jacobians
- Save the dataset in HDF5 format

### 2. Train the Model

```bash
python model.py --data_path path/to/your/data.h5 \
                --save_dir model_checkpoints \
                --subset_size 49000 \
                --batch_size 32 \
                --num_epochs 50 \
                --learning_rate 0.001
```

Training features:
- Automatic train/validation split
- Progress monitoring
- Checkpoint saving
- Loss visualization
- Early stopping
- Learning rate scheduling

### 3. Inference

```python
from model import load_model_for_inference

# Load the model
model, stats = load_model_for_inference('path/to/model_checkpoints/inference_model.pt')

# Make predictions
with torch.no_grad():
    prediction = model(f_input, g_input)
```

## Model Architecture

The neural network (`JacobianNet`) consists of:
- Input: Two 2D fields (f and g)
- Multiple convolutional layers with batch normalization and dropout
- Conservation-aware loss function based on Arakawa scheme properties

## Training Details

The training process includes:
- Data normalization
- Memory-efficient batch processing
- Conservation-based loss function incorporating:
  - Mean squared error
  - Mean vorticity conservation
  - Energy conservation
  - Square vorticity conservation
- Early stopping based on validation loss
- Learning rate reduction on plateau

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```
Arakawa's Method Is a Finite-Element Method
by DENNIS C. JESPERSEN
PII: 0021-9991(74)90047-3
```