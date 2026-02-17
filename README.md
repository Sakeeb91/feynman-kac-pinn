# Feynman-Kac PINN Platform

A full-stack web platform for exploring Physics-Informed Neural Networks (PINNs) that leverage the Feynman-Kac formula to solve partial differential equations via random walk Monte Carlo representations.

## Overview

Traditional PINNs minimize PDE residuals at collocation points, which struggles with sharp gradients and boundary layers. This project implements an alternative approach using the **Feynman-Kac formula**: any elliptic/parabolic PDE solution can be written as an expectation over random walks.

```
u(x) = E[g(B_τ) · exp(-∫₀^τ c(B_s)ds)]
```

Where:
- `B` is Brownian motion starting at `x`
- `τ` is the exit time from the domain
- `g` is the boundary condition
- `c` is the potential/reaction term

The neural network learns `u(x)` by matching Monte Carlo estimates from simulated random walks, bypassing PDE residual computation entirely.

## Key Features

- **Dual Problem Support**: Solve both 10D Black-Scholes (finance) and high-dimensional Schrodinger (quantum) equations
- **Interactive Simulations**: Run live simulations with custom parameters or explore pre-computed demonstrations
- **Visualization Suite**: View random walk trajectories, solution surfaces, training convergence, and error analysis
- **Educational Interface**: Step-by-step explanations of the Feynman-Kac connection between stochastic processes and PDEs

## Why This Approach?

| Traditional PINN | Feynman-Kac PINN |
|------------------|------------------|
| Requires automatic differentiation through PDE | Only function evaluation needed |
| Collocation points become sparse in high-D | Random walks scale naturally to high dimensions |
| Struggles with boundary layers | Boundary conditions enforced exactly via exit times |
| Single forward pass per training step | Embarrassingly parallel MC sampling |

## Technology Stack

### Backend
- **FastAPI**: Async Python API for simulation orchestration
- **PyTorch**: Neural network training with MPS (Apple Silicon) and CUDA support
- **NumPy/SciPy**: Numerical computations and random walk simulations

### Frontend
- **React**: Interactive user interface
- **TypeScript**: Type-safe frontend development
- **Plotly/D3.js**: Scientific visualizations

### Infrastructure
- **Docker**: Containerized deployment
- **Modal/Google Colab**: Cloud GPU for heavy computations (free tier)
- **Vercel/Railway**: Frontend and backend hosting

## Project Structure

```
feynman-kac-pinn/
├── backend/
│   ├── app/
│   │   ├── api/            # FastAPI routes
│   │   ├── core/           # Configuration, dependencies
│   │   ├── models/         # Pydantic schemas
│   │   ├── services/       # Business logic
│   │   └── utils/          # Helper functions
│   └── tests/              # Backend tests
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── pages/          # Page components
│   │   ├── hooks/          # Custom React hooks
│   │   └── utils/          # Frontend utilities
│   └── public/             # Static assets
├── ml/
│   ├── models/             # Neural network architectures
│   ├── training/           # Training loops and optimizers
│   ├── inference/          # Inference utilities
│   └── data/               # Data generation (random walks)
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks for exploration
└── .github/workflows/      # CI/CD pipelines
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- PyTorch 2.0+ (with MPS support for Apple Silicon)

### Local Development

```bash
# Clone the repository
git clone https://github.com/Sakeeb91/feynman-kac-pinn.git
cd feynman-kac-pinn

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### Cloud GPU Access (Free)

For heavy training workloads, use one of these free options:

1. **Google Colab**: Open notebooks in `notebooks/` directory with Colab
2. **Modal**: Deploy training jobs to Modal's free tier
3. **Local MPS**: Apple Silicon Macs can use Metal acceleration

## Mathematical Background

### The Feynman-Kac Formula

For a PDE of the form:
```
∂u/∂t + Lu + cu = 0
```
with boundary condition `u(x,T) = g(x)`, where `L` is the generator of a diffusion process, the solution is:

```
u(x,t) = E[g(X_T) · exp(-∫_t^T c(X_s)ds) | X_t = x]
```

### Training Algorithm

1. Sample interior points `{x_i}` from the domain
2. For each `x_i`, simulate `M` random walk trajectories until they exit
3. Compute MC estimate: `û(x_i) = (1/M) Σ g(B_τ^j) · exp(-∫c ds)`
4. Train neural network: minimize `Σ |NN(x_i) - û(x_i)|²`

### Target Problems

#### 10D Black-Scholes
Multi-asset option pricing with correlated underlyings:
```
∂V/∂t + (1/2)Σ σᵢσⱼρᵢⱼSᵢSⱼ ∂²V/∂Sᵢ∂Sⱼ + rΣSᵢ∂V/∂Sᵢ - rV = 0
```

#### High-Dimensional Schrodinger
Ground state of quantum harmonic oscillator:
```
-ℏ²/2m ∇²ψ + V(x)ψ = Eψ
```

## Roadmap

See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for detailed phase breakdown.
- Phase 1 derivation and API notes: [PHASE1_DERIVATION.md](docs/PHASE1_DERIVATION.md)
- Phase 3 problem derivations and registry notes: [PHASE3_PROBLEMS.md](docs/PHASE3_PROBLEMS.md)

- [x] Phase 1: Core random walk engine
- [ ] Phase 2: Neural network architecture
- [ ] Phase 3: FastAPI backend
- [ ] Phase 4: React frontend
- [ ] Phase 5: Visualization suite
- [ ] Phase 6: Cloud deployment

## References

1. Kac, M. (1949). "On distributions of certain Wiener functionals"
2. Raissi, M., et al. (2019). "Physics-informed neural networks"
3. Han, J., et al. (2018). "Solving high-dimensional PDEs using deep learning"
4. Beck, C., et al. (2021). "Deep splitting method for parabolic PDEs"

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome. Please read the implementation plan and open an issue before submitting PRs.
