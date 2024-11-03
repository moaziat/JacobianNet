import numpy as np
import torch 
import os
import h5py

def cartesian_arakawa_jacobian(f, g, dx, dy):
    """
    Computing the arakawa jacobian.
    """
    J = torch.zeros_like(f)

    J1 = (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) * \
         (torch.roll(g, 1, dims=1) - torch.roll(g, -1, dims=1))
    
    J2 = (torch.roll(f, 1, dims=1) - torch.roll(f, -1, dims=1)) * \
         (torch.roll(g, -1, dims=0) - torch.roll(g, 1, dims=0))
    
    J3 = (torch.roll(f, -1, dims=0) - torch.roll(f, -1, dims=1)) * \
         (torch.roll(g, 1, dims=1) - torch.roll(g, -1, dims=1))
    
    J4 = (torch.roll(f, -1, dims=1) - torch.roll(f, -1, dims=0)) * \
         (torch.roll(g, -1, dims=0) - torch.roll(g, 1, dims=0))
    
    J = (J1 - J2 + J3 - J4) / (12 * dx * dy)
    return J

def generate_turbulent_field(size, k0=5, decay=2.0):
    """
    Generate a turbulent-like field using Fourier synthesis.
    """
    kx = torch.fft.fftfreq(size) * size
    ky = torch.fft.fftfreq(size) * size
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    K = torch.sqrt(Kx**2 + Ky**2)
    
    spectrum = torch.exp(-(K - k0)**2 / (2 * k0**2)) * (K + 1e-10)**(-decay)
    spectrum[0, 0] = 0
    
    phases = 2 * np.pi * torch.rand(size, size)
    fourier_components = spectrum * (torch.cos(phases) + 1j * torch.sin(phases))
    
    field = torch.fft.ifft2(fourier_components).real
    return (field - field.mean()) / field.std()

def generate_vortex_dipole(size, x1, y1, x2, y2, strength=1.0, radius=0.1):
    """
    Generate a vortex dipole field.
    """
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    r1 = torch.sqrt((X - x1)**2 + (Y - y1)**2)
    r2 = torch.sqrt((X - x2)**2 + (Y - y2)**2)
    
    field = strength * (torch.exp(-r1**2 / radius**2) - torch.exp(-r2**2 / radius**2))
    return field

def generate_kelvin_helmholtz(size, amplitude=0.1, wavelength=0.25, thickness=0.1):
    """
    Generate a Kelvin-Helmholtz like shear layer.
    """
    x = torch.linspace(0, 2*np.pi, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    base_flow = torch.tanh(Y / thickness)
    perturbation = amplitude * torch.sin(X / wavelength) * torch.exp(-Y**2 / thickness)
    
    return base_flow + perturbation

def generate_field_pair(size=128, field_type=None):
    """
    Generate a pair of fields with specified characteristics.
    """
    if field_type is None:
        field_type = np.random.choice(['turbulent', 'vortex', 'shear', 'mixed'])
        
    if field_type == 'turbulent':
        f = generate_turbulent_field(size, k0=torch.randint(3, 8, (1,)).item())
        g = generate_turbulent_field(size, k0=torch.randint(3, 8, (1,)).item())
        
    elif field_type == 'vortex':
        f = torch.zeros(size, size)
        g = torch.zeros(size, size)
        
        for _ in range(3):
            x1, y1 = torch.rand(2) * 1.6 - 0.8
            x2, y2 = x1 + 0.2, y1 + 0.2
            strength = torch.rand(1).item() * 0.5 + 0.5
            radius = torch.rand(1).item() * 0.1 + 0.05
            
            f += generate_vortex_dipole(size, x1, y1, x2, y2, strength, radius)
            g += generate_vortex_dipole(size, y1, x1, y2, x2, strength*0.8, radius*1.2)
            
    elif field_type == 'shear':
        f = generate_kelvin_helmholtz(size, 
                                    amplitude=torch.rand(1).item()*0.1,
                                    wavelength=torch.rand(1).item()*0.2 + 0.2)
        g = generate_kelvin_helmholtz(size, 
                                    amplitude=torch.rand(1).item()*0.1,
                                    wavelength=torch.rand(1).item()*0.2 + 0.2)
        
    else:  # mixed
        f = (generate_turbulent_field(size, k0=4) * 0.4 + 
             generate_kelvin_helmholtz(size) * 0.3 +
             generate_vortex_dipole(size, 0.3, 0.3, -0.3, -0.3) * 0.3)
        
        g = (generate_turbulent_field(size, k0=5) * 0.4 +
             generate_kelvin_helmholtz(size) * 0.3 +
             generate_vortex_dipole(size, -0.3, 0.3, 0.3, -0.3) * 0.3)
    
    return f, g

def generate_and_save_dataset(num_samples, grid_size, batch_size, save_dir, file_name, device='cpu'):
    """
    Generate and immediately save each batch to disk using HDF5 format.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    full_path = os.path.join(save_dir, file_name)
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Setup coordinate grid
    x = torch.linspace(-20, 20, grid_size, device=device)
    y = torch.linspace(-20, 20, grid_size, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    dx = dy = x[1] - x[0]
    
    field_types = ['turbulent', 'vortex', 'shear', 'mixed']
    
    with h5py.File(full_path, 'w') as f:
        f.create_dataset('f_data', shape=(num_samples, grid_size, grid_size), dtype='float32')
        f.create_dataset('g_data', shape=(num_samples, grid_size, grid_size), dtype='float32')
        f.create_dataset('jacobian_data', shape=(num_samples, grid_size, grid_size), dtype='float32')
        
        start_idx = 0
        for batch in range(num_batches):
            current_batch_size = min(batch_size, num_samples - batch * batch_size)
            end_idx = start_idx + current_batch_size
            
            print(f"Generating batch {batch + 1}/{num_batches} (samples {start_idx} to {end_idx-1})")
            
            f_batch = []
            g_batch = []
            j_batch = []
            
            for i in range(current_batch_size):
                field_type = np.random.choice(field_types)
                F, G = generate_field_pair(size=grid_size, field_type=field_type)
                J = cartesian_arakawa_jacobian(F, G, dx, dy)
                
                f_batch.append(F.cpu().numpy())
                g_batch.append(G.cpu().numpy())
                j_batch.append(J.cpu().numpy())
            
            f_batch = np.array(f_batch)
            g_batch = np.array(g_batch)
            j_batch = np.array(j_batch)
            
            f['f_data'][start_idx:end_idx] = f_batch
            f['g_data'][start_idx:end_idx] = g_batch
            f['jacobian_data'][start_idx:end_idx] = j_batch
            
            start_idx = end_idx
            
            del f_batch, g_batch, j_batch
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        print(f"Dataset saved to {full_path}")

def load_dataset(file_path, start_idx=0, end_idx=None):
    """
    Load a portion of the dataset from HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        if end_idx is None:
            end_idx = f['f_data'].shape[0]
            
        f_data = torch.from_numpy(f['f_data'][start_idx:end_idx])
        g_data = torch.from_numpy(f['g_data'][start_idx:end_idx])
        j_data = torch.from_numpy(f['jacobian_data'][start_idx:end_idx])
        
    return f_data, g_data, j_data

if __name__ == "__main__":
    num_samples = 50000
    grid_size = 128
    batch_size = 50
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Generating dataset on {device}")
    
    save_dir = os.path.join(os.path.dirname(os.getcwd()), "Arakawa_Datasets")
    file_name = "diverse_fields.h5"
    
    generate_and_save_dataset(
        num_samples=num_samples,
        grid_size=grid_size,
        batch_size=batch_size,
        save_dir=save_dir,
        file_name=file_name,
        device=device
    )
    
    print("\nTesting data loading...")
    f_test, g_test, j_test = load_dataset(
        os.path.join(save_dir, file_name), 
        start_idx=0, 
        end_idx=5
    )
    print(f"Loaded test batch shapes: {f_test.shape}, {g_test.shape}, {j_test.shape}")