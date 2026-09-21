import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
from datetime import datetime

# ==========================================
# 1. THE 4D NEURAL BRAIN (REPNET)
# ==========================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SIREN_RepNet4D(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        # Massively upgraded neural network for 4D topology (X, Y, Z, Time)
        self.net = nn.Sequential(
            SineLayer(4, hidden, is_first=True),
            SineLayer(hidden, hidden),
            SineLayer(hidden, hidden),
            nn.Linear(hidden, 1)
        )
        
    def forward(self, x, y, z, t):
        # Concatenate into a 4D input tensor
        xyzt = torch.cat([x, y, z, t], dim=1)
        # Predict the 4D geometric invariant Z
        Z_manifold = self.net(xyzt)
        return Z_manifold

# ==========================================
# 2. THE 3D PHYSICS REGISTRY
# ==========================================
class PhysicsRegistry3D:
    @staticmethod
    def generate_navier_stokes_proxy():
        """
        A chaotic 3D fluid vortex system serving as a proxy for Navier-Stokes complexity.
        Returns the velocity vector field (dx/dt, dy/dt, dz/dt) given (x,y,z,t).
        """
        a = round(random.uniform(-1.0, 1.0), 2)
        b = round(random.uniform(-1.0, 1.0), 2)
        c = round(random.uniform(0.1, 1.0), 2)
        
        eq_str = f"3D Chaotic Vortex: dx/dt={a}y*sin(t), dy/dt={b}x*cos(t), dz/dt=-{c}z + x*y"
        
        def vector_field(x, y, z, t):
            dx_dt = a * y * torch.sin(t)
            dy_dt = b * x * torch.cos(t)
            dz_dt_spatial = -c * z + (x * y)
            return dx_dt, dy_dt, dz_dt_spatial
            
        return eq_str, vector_field, {"type": "3D_Vortex", "a": a, "b": b, "c": c}

    @staticmethod
    def generate_true_navier_stokes_abc_flow():
        """
        The Arnold-Beltrami-Childress (ABC) Flow: An exact 3D solution to the 
        Euler and Navier-Stokes equations used to study turbulent chaos.
        u_x = A*sin(z) + C*cos(y)
        u_y = B*sin(x) + A*cos(z)
        u_z = C*sin(y) + B*cos(x)
        """
        A = round(random.uniform(0.5, 1.5), 2)
        B = round(random.uniform(0.5, 1.5), 2)
        C = round(random.uniform(0.5, 1.5), 2)
        
        eq_str = f"True Navier-Stokes (ABC Flow): dx/dt={A}sin(z)+{C}cos(y), dy/dt={B}sin(x)+{A}cos(z), dz/dt={C}sin(y)+{B}cos(x)"
        
        def vector_field(x, y, z, t):
            # ABC flow is primarily spatial, but we inject a time-decay factor for viscosity
            decay = torch.exp(-0.1 * t) 
            dx_dt = (A * torch.sin(z) + C * torch.cos(y)) * decay
            dy_dt = (B * torch.sin(x) + A * torch.cos(z)) * decay
            dz_dt = (C * torch.sin(y) + B * torch.cos(x)) * decay
            return dx_dt, dy_dt, dz_dt
            
        return eq_str, vector_field, {"type": "Navier_Stokes_ABC", "A": A, "B": B, "C": C}

    @staticmethod
    def generate_hurricane_lorenz_flow():
        """
        The Lorenz Attractor: The standard mathematical model for atmospheric 
        convection and hurricane chaos (Weather prediction).
        dx/dt = sigma*(y - x)
        dy/dt = x*(rho - z) - y
        dz/dt = x*y - beta*z
        """
        sigma = round(random.uniform(8.0, 12.0), 1)  # Prandtl number
        rho = round(random.uniform(25.0, 30.0), 1)   # Rayleigh number
        beta = round(random.uniform(2.0, 3.0), 2)    # Geometric factor
        
        eq_str = f"Real-World Hurricane (Lorenz): dx/dt={sigma}(y-x), dy/dt=x({rho}-z)-y, dz/dt=xy-{beta}z"
        
        def vector_field(x, y, z, t):
            # Scale down slightly to prevent blowing up the neural network gradients
            scale = 0.1
            dx_dt = sigma * (y - x) * scale
            dy_dt = (x * (rho - z) - y) * scale
            dz_dt = (x * y - beta * z) * scale
            return dx_dt, dy_dt, dz_dt
            
        return eq_str, vector_field, {"type": "Hurricane_Lorenz", "sigma": sigma, "rho": rho, "beta": beta}

    @staticmethod
    def generate_aneurysm_blood_flow():
        """
        A 3D pulsating turbulent flow representing blood recirculating inside a heart aneurysm.
        Uses Womersley-like pulsatile boundary conditions and rotational chaotic shear.
        """
        pulse = round(random.uniform(1.0, 3.0), 2)  # Heart rate pulse frequency
        shear = round(random.uniform(0.5, 1.5), 2)  # Plaque shear friction
        
        eq_str = f"Real-World Aneurysm Blood Flow: dx/dt={shear}y*sin({pulse}t), dy/dt=-{shear}x*sin({pulse}t), dz/dt=x*y*cos(z)"
        
        def vector_field(x, y, z, t):
            dx_dt = shear * y * torch.sin(pulse * t)
            dy_dt = -shear * x * torch.sin(pulse * t)
            dz_dt = (x * y) * torch.cos(z)
            return dx_dt, dy_dt, dz_dt
            
        return eq_str, vector_field, {"type": "Aneurysm_Blood_Flow", "pulse": pulse, "shear": shear}


# ==========================================
# 3. THE 4D MASSIVE SOLVER ENGINE
# ==========================================
def train_3d_representation(vector_field_fn, epochs=500, hidden=256, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Booting massive 4D Engine on {device}...")
    
    # 4-Dimensional Hypergrid (15x15x15x15 = 50,625 points!)
    x_v = torch.linspace(-1.0, 1.0, 15)
    y_v = torch.linspace(-1.0, 1.0, 15)
    z_v = torch.linspace(-1.0, 1.0, 15)
    t_v = torch.linspace(0.0, 2.0, 15)
    
    # Generate the hypergrid
    grid_x, grid_y, grid_z, grid_t = torch.meshgrid(x_v, y_v, z_v, t_v, indexing='ij')
    
    # Flatten and push to GPU
    x_train = grid_x.reshape(-1, 1).to(device).requires_grad_(True)
    y_train = grid_y.reshape(-1, 1).to(device).requires_grad_(True)
    z_train = grid_z.reshape(-1, 1).to(device).requires_grad_(True)
    t_train = grid_t.reshape(-1, 1).to(device).requires_grad_(True)
    
    model = SIREN_RepNet4D(hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        opt.zero_grad()
        
        # Predict Geometric Manifold Z
        Z = model(x_train, y_train, z_train, t_train)
        
        # Calculate 4 simultaneous partial derivatives across all 50,000 points!
        # This is where the T4 Cloud GPU earns its keep.
        dZ_dx = torch.autograd.grad(Z, x_train, grad_outputs=torch.ones_like(Z), create_graph=True)[0]
        dZ_dy = torch.autograd.grad(Z, y_train, grad_outputs=torch.ones_like(Z), create_graph=True)[0]
        dZ_dz = torch.autograd.grad(Z, z_train, grad_outputs=torch.ones_like(Z), create_graph=True)[0]
        dZ_dt = torch.autograd.grad(Z, t_train, grad_outputs=torch.ones_like(Z), create_graph=True)[0]
        
        # Get the underlying physics vector field
        dx_dt, dy_dt, dz_dt_spatial = vector_field_fn(x_train, y_train, z_train, t_train)
        
        # Total Temporal Derivative (Chain Rule): dZ/dt = Z_x(x') + Z_y(y') + Z_z(z') + Z_t
        total_derivative = (dZ_dx * dx_dt) + (dZ_dy * dy_dt) + (dZ_dz * dz_dt_spatial) + dZ_dt
        
        # We want the total derivative to be 0 (meaning Z is an invariant geometry of the chaos)
        variance_loss = torch.var(total_derivative)
        collapse_penalty = torch.mean((total_derivative + 1.0)**2) * 0.1 
        
        loss = variance_loss + collapse_penalty
        loss.backward()
        opt.step()
        
        if epoch % 100 == 0:
            print(f"    Epoch {epoch}/{epochs} | 4D Geometric Variance: {variance_loss.item():.6f}")

    print("✅ Training complete. Extracting 4D geometry...")
    with torch.no_grad():
        Z_pred = model(x_train, y_train, z_train, t_train).cpu().numpy()
        
    return Z_pred.flatten(), variance_loss.item()

# ==========================================
# 4. THE 3D DATASET GENERATOR
# ==========================================
def generate_3d_dataset(num_equations=3):
    print("==================================================")
    print(" 🌌 PARC 3D ENGINE: NAVIER-STOKES CLOUD GPU TEST 🌌")
    print("==================================================")
    
    for i in range(num_equations):
        # We test both real-world systems
        if i % 2 == 0:
            eq_str, vec_fn, meta = PhysicsRegistry3D.generate_hurricane_lorenz_flow()
        else:
            eq_str, vec_fn, meta = PhysicsRegistry3D.generate_aneurysm_blood_flow()
            
        print(f"\n[Equation {i+1}] {eq_str}")
        
        # Push to the massive 4D engine
        z_arr, phys_loss = train_3d_representation(vec_fn, epochs=500)
        
        print(f"-> Final Geometry Geometric Variance: {phys_loss:.6f}")

        # Save the massive 4D array to a knowledge base
        entry = {
            "equation_string": eq_str,
            "data_z": z_arr.tolist(),
            "meta": meta
        }
        with open("PARC_4D_Knowledge_Base.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    generate_3d_dataset(num_equations=3)
