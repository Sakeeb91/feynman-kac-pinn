# Implementation Plan: Feynman-Kac PINN Platform

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technology Selection](#technology-selection)
4. [Phased Implementation](#phased-implementation)
5. [Risk Assessment](#risk-assessment)
6. [Testing Strategy](#testing-strategy)
7. [First Concrete Task](#first-concrete-task)

---

## Project Overview

### Problem Statement
Standard Physics-Informed Neural Networks (PINNs) minimize PDE residuals at collocation points, which:
- Requires expensive automatic differentiation through PDE operators
- Struggles with sharp gradients and boundary layers
- Suffers from the curse of dimensionality (collocation points become sparse)

### Solution
Leverage the **Feynman-Kac formula** to reformulate PDE solving as a Monte Carlo estimation problem:
- Neural network predicts solution values `u(x)`
- Random walks provide ground truth via MC estimates
- No PDE residual computation needed
- Naturally scales to high dimensions

### Target Audience
- Researchers exploring neural PDE solvers
- Students learning the stochastic-PDE connection
- Practitioners needing high-dimensional PDE solutions

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (React)                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │  Problem    │  │  Parameter   │  │ Simulation  │  │  Visualization  │   │
│  │  Selector   │  │  Controls    │  │  Monitor    │  │    Dashboard    │   │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  └────────┬────────┘   │
└─────────┼────────────────┼─────────────────┼──────────────────┼────────────┘
          │                │                 │                  │
          ▼                ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           REST API (FastAPI)                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌─────────────────┐   │
│  │ /problems   │  │ /simulations │  │   /train    │  │   /results      │   │
│  │   GET       │  │  POST, GET   │  │ POST, WS    │  │   GET           │   │
│  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  └────────┬────────┘   │
└─────────┼────────────────┼─────────────────┼──────────────────┼────────────┘
          │                │                 │                  │
          ▼                ▼                 ▼                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SERVICE LAYER                                       │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │  ProblemRegistry    │  │  SimulationManager  │  │   ResultsCache      │ │
│  │  - Black-Scholes    │  │  - Job queue        │  │   - Pre-computed    │ │
│  │  - Schrodinger      │  │  - Status tracking  │  │   - Live results    │ │
│  └──────────┬──────────┘  └──────────┬──────────┘  └──────────┬──────────┘ │
└─────────────┼────────────────────────┼────────────────────────┼─────────────┘
              │                        │                        │
              ▼                        ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML CORE (PyTorch)                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  RandomWalkEngine │  │  FeynmanKacPINN  │  │  TrainingOrchestrator   │  │
│  │  - Brownian motion│  │  - MLP network   │  │  - Loss computation     │  │
│  │  - Exit times     │  │  - Forward pass  │  │  - Optimizer step       │  │
│  │  - MC estimates   │  │  - Device mgmt   │  │  - Checkpointing        │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPUTE BACKENDS                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │  Local (MPS/CPU) │  │  Google Colab    │  │  Modal (On-demand GPU)   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Request**: Select problem (Black-Scholes/Schrodinger), set parameters
2. **API Layer**: Validate request, create simulation job
3. **Service Layer**: Queue job, manage state, retrieve cached results if available
4. **ML Core**: Generate random walks, compute MC estimates, train PINN
5. **Results**: Stream training progress, return final solution and visualizations

---

## Technology Selection

### Backend Framework: FastAPI

| Criterion | FastAPI | Flask | Django |
|-----------|---------|-------|--------|
| Async support | Native | Via extensions | Limited |
| Type hints | Built-in | Manual | Manual |
| WebSocket | Native | Via extensions | Channels |
| Learning curve | Moderate | Low | High |
| Documentation | Excellent | Good | Excellent |

**Decision**: FastAPI for native async (crucial for long-running simulations) and automatic OpenAPI docs.

### ML Framework: PyTorch

| Criterion | PyTorch | TensorFlow | JAX |
|-----------|---------|------------|-----|
| MPS (Apple Silicon) | Full support | Limited | Experimental |
| Dynamic graphs | Native | Eager mode | JIT required |
| Research flexibility | High | Moderate | High |
| Community | Large | Large | Growing |
| Learning curve | Moderate | Moderate | Steep |

**Decision**: PyTorch for MPS support (your M4 Mac), intuitive API, and strong research community.

### Frontend: React + TypeScript

| Criterion | React | Vue | Svelte |
|-----------|-------|-----|--------|
| Ecosystem | Largest | Large | Growing |
| TypeScript | Excellent | Excellent | Good |
| Visualization libs | Many | Some | Few |
| Job market | Highest | High | Growing |

**Decision**: React for ecosystem (Plotly, D3 integrations) and transferable skills.

### Visualization: Plotly.js

| Criterion | Plotly | D3.js | Chart.js |
|-----------|--------|-------|----------|
| 3D support | Excellent | Complex | Limited |
| Scientific plots | Excellent | Manual | Basic |
| Interactivity | Built-in | Manual | Limited |
| React integration | plotly.js-react | Manual | react-chartjs |

**Decision**: Plotly for built-in 3D surfaces, heatmaps, and line plots needed for PDE visualization.

### Free Cloud Options

| Service | Use Case | GPU | Limits |
|---------|----------|-----|--------|
| Google Colab | Training notebooks | T4 | 12hr sessions |
| Modal | API-triggered training | A10G | $30 free credits |
| Railway | Backend hosting | None | 500hr/month |
| Vercel | Frontend hosting | N/A | Unlimited static |
| GitHub Actions | CI/CD | None | 2000 min/month |

---

## Phased Implementation

### Phase 1: Core Random Walk Engine

**Objective**: Implement the mathematical foundation - Brownian motion simulation and Feynman-Kac estimator.

**Files to Create**:
```
ml/
├── data/
│   ├── __init__.py
│   ├── brownian.py          # Brownian motion simulator
│   ├── domains.py           # Domain definitions (hypercube, sphere)
│   └── boundary.py          # Boundary condition handlers
└── utils/
    ├── __init__.py
    └── mc_estimator.py      # Monte Carlo Feynman-Kac estimator
```

**Key Functions**:

```python
# ml/data/brownian.py
def simulate_brownian_paths(
    x0: torch.Tensor,           # Starting points (batch_size, dim)
    dt: float,                  # Time step
    max_steps: int,             # Maximum steps before timeout
    domain: Domain,             # Domain object with exit condition
    device: str = "mps"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate Brownian motion until domain exit.

    Returns:
        exit_points: (batch_size, dim) - Where paths exited
        exit_times: (batch_size,) - When paths exited
        path_integrals: (batch_size,) - ∫c(B_s)ds along paths
    """
```

```python
# ml/utils/mc_estimator.py
def feynman_kac_estimate(
    x: torch.Tensor,            # Query points
    boundary_fn: Callable,      # g(x) boundary condition
    potential_fn: Callable,     # c(x) potential term
    domain: Domain,
    n_paths: int = 1000,
    dt: float = 0.001
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Feynman-Kac MC estimate.

    Returns:
        estimates: E[g(B_τ) * exp(-∫c ds)]
        variances: Sample variance of estimates
    """
```

**Deliverables**:
- [ ] Brownian motion simulation with batched GPU execution
- [ ] Hypercube and hypersphere domain definitions
- [ ] Exit time detection algorithm
- [ ] Path integral accumulator
- [ ] MC estimator with variance tracking

**Verification**:
1. 1D heat equation: Compare MC estimate to analytical solution
2. Exit time distribution: Match theoretical Brownian exit time for sphere
3. Convergence rate: Verify O(1/sqrt(N)) variance decay

**Challenges**:
- Exit detection at boundaries (interpolation vs. clamping)
- Numerical stability for small dt
- Memory management for long paths in high dimensions

**Definition of Done**:
- All tests pass with <1% relative error on analytical benchmarks
- Can simulate 10,000 paths in 10D in <10 seconds on MPS
- Documentation with mathematical derivation

---

### Phase 2: Neural Network Architecture

**Objective**: Implement the PINN architecture that learns u(x) from MC estimates.

**Files to Create**:
```
ml/
├── models/
│   ├── __init__.py
│   ├── pinn.py              # Main PINN architecture
│   ├── activations.py       # Custom activation functions
│   └── initialization.py    # Weight initialization schemes
└── training/
    ├── __init__.py
    ├── trainer.py           # Training loop
    ├── losses.py            # Loss functions
    └── schedulers.py        # Learning rate schedulers
```

**Key Classes**:

```python
# ml/models/pinn.py
class FeynmanKacPINN(nn.Module):
    """
    MLP that approximates PDE solution u(x).

    Architecture:
        Input(dim) -> [Dense(width) -> Activation]×depth -> Output(1)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 64, 64, 64],
        activation: str = "tanh",
        output_activation: Optional[str] = None
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict u(x) for input points."""
        ...
```

```python
# ml/training/trainer.py
class FeynmanKacTrainer:
    """
    Training orchestrator for Feynman-Kac PINN.
    """
    def __init__(
        self,
        model: FeynmanKacPINN,
        problem: Problem,
        device: str = "mps",
        lr: float = 1e-3
    ):
        ...

    def train_step(self, batch_size: int, n_mc_paths: int) -> Dict:
        """
        Single training step:
        1. Sample interior points
        2. Compute MC estimates
        3. Forward pass through network
        4. Compute MSE loss
        5. Backprop and update
        """
        ...

    def train(self, n_steps: int, ...) -> History:
        """Full training loop with logging."""
        ...
```

**Deliverables**:
- [ ] Flexible MLP architecture with configurable depth/width
- [ ] Training loop with MC estimation integrated
- [ ] Loss tracking and convergence monitoring
- [ ] Checkpoint saving/loading
- [ ] Device management (MPS/CUDA/CPU auto-detection)

**Verification**:
1. Overfit to single point (loss -> 0)
2. 2D Laplace equation on unit square (known analytical solution)
3. Training loss monotonically decreases

**Challenges**:
- Balancing MC variance vs. network learning
- Choosing appropriate network size for problem dimension
- Gradient explosion/vanishing in deep networks

**Definition of Done**:
- Can train 10D network to <5% relative error on test benchmark
- Training converges within 5000 steps on MPS
- Checkpoint can be loaded and inference runs correctly

---

### Phase 3: Problem Definitions

**Objective**: Implement the two target problems with clean interfaces.

**Files to Create**:
```
ml/
└── problems/
    ├── __init__.py
    ├── base.py              # Abstract problem class
    ├── black_scholes.py     # 10D Black-Scholes
    └── schrodinger.py       # High-D Schrodinger
```

**Key Classes**:

```python
# ml/problems/base.py
class Problem(ABC):
    """Abstract base class for PDE problems."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension of the problem."""

    @property
    @abstractmethod
    def domain(self) -> Domain:
        """Domain geometry."""

    @abstractmethod
    def boundary_condition(self, x: torch.Tensor) -> torch.Tensor:
        """g(x) - boundary/terminal condition."""

    @abstractmethod
    def potential(self, x: torch.Tensor) -> torch.Tensor:
        """c(x) - potential/reaction term."""

    @abstractmethod
    def analytical_solution(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Known solution for verification (if available)."""
```

```python
# ml/problems/black_scholes.py
class BlackScholes10D(Problem):
    """
    10-dimensional Black-Scholes equation for basket option pricing.

    PDE: ∂V/∂t + (1/2)Σ σ²S²∂²V/∂S² + rΣS∂V/∂S - rV = 0

    Parameters:
        strike: Option strike price
        risk_free_rate: Risk-free interest rate r
        volatilities: σ for each asset
        correlation: Correlation matrix ρ
        maturity: Time to expiration T
    """
```

```python
# ml/problems/schrodinger.py
class HarmonicOscillatorND(Problem):
    """
    N-dimensional quantum harmonic oscillator.

    PDE: -ℏ²/2m ∇²ψ + (1/2)mω²|x|²ψ = Eψ

    Ground state is Gaussian - good for verification.
    """
```

**Deliverables**:
- [ ] Abstract Problem interface
- [ ] 10D Black-Scholes with configurable parameters
- [ ] N-D Harmonic oscillator with known ground state
- [ ] Parameter validation and sensible defaults

**Verification**:
1. Black-Scholes: Compare 1D case to analytical formula
2. Schrodinger: Ground state matches Gaussian form

**Definition of Done**:
- Both problems pass analytical verification tests
- Can switch between problems with single config change

---

### Phase 4: FastAPI Backend

**Objective**: Create REST API for simulation management and result retrieval.

**Files to Create**:
```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry
│   ├── config.py            # Settings and configuration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── problems.py  # Problem listing endpoints
│   │   │   ├── simulations.py  # Simulation CRUD
│   │   │   └── results.py   # Result retrieval
│   │   └── deps.py          # Dependencies
│   ├── core/
│   │   ├── __init__.py
│   │   └── simulation_manager.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── problem.py       # Problem schemas
│   │   ├── simulation.py    # Simulation schemas
│   │   └── result.py        # Result schemas
│   └── services/
│       ├── __init__.py
│       ├── problem_registry.py
│       └── result_cache.py
└── requirements.txt
```

**Key Endpoints**:

```python
# GET /api/v1/problems
# List available problems with their parameters
[
    {
        "id": "black-scholes-10d",
        "name": "10D Black-Scholes",
        "dimension": 10,
        "parameters": [
            {"name": "strike", "type": "float", "default": 100.0},
            {"name": "risk_free_rate", "type": "float", "default": 0.05},
            ...
        ]
    },
    ...
]

# POST /api/v1/simulations
# Create a new simulation
{
    "problem_id": "black-scholes-10d",
    "parameters": {"strike": 100, ...},
    "training_config": {"n_steps": 5000, "batch_size": 256}
}
# Returns: {"simulation_id": "uuid", "status": "queued"}

# GET /api/v1/simulations/{id}
# Get simulation status and progress
{
    "simulation_id": "uuid",
    "status": "running",
    "progress": 0.45,
    "metrics": {"loss": 0.023, "step": 2250}
}

# GET /api/v1/results/{simulation_id}
# Get final results including solution data
{
    "simulation_id": "uuid",
    "solution": {...},
    "training_history": [...],
    "visualizations": {...}
}
```

**Deliverables**:
- [ ] Problem listing endpoint
- [ ] Simulation creation and status tracking
- [ ] Background task execution for training
- [ ] Result storage and retrieval
- [ ] WebSocket for live training updates (stretch)

**Verification**:
1. API documentation auto-generated at /docs
2. Can create, monitor, and retrieve simulation via curl
3. Concurrent simulations don't interfere

**Definition of Done**:
- All endpoints return correct status codes
- Swagger docs are complete and accurate
- Can run end-to-end simulation via API

---

### Phase 5: React Frontend

**Objective**: Build interactive UI for problem selection, parameter tuning, and visualization.

**Files to Create**:
```
frontend/
├── src/
│   ├── App.tsx
│   ├── index.tsx
│   ├── api/
│   │   └── client.ts        # API client
│   ├── components/
│   │   ├── ProblemSelector.tsx
│   │   ├── ParameterPanel.tsx
│   │   ├── SimulationMonitor.tsx
│   │   └── ResultsViewer.tsx
│   ├── pages/
│   │   ├── HomePage.tsx
│   │   ├── SimulationPage.tsx
│   │   └── ResultsPage.tsx
│   ├── hooks/
│   │   ├── useSimulation.ts
│   │   └── useResults.ts
│   └── types/
│       └── index.ts
├── package.json
├── tsconfig.json
└── vite.config.ts
```

**Key Components**:

```tsx
// ProblemSelector.tsx
interface ProblemSelectorProps {
    problems: Problem[];
    selected: string | null;
    onSelect: (id: string) => void;
}

// ParameterPanel.tsx
interface ParameterPanelProps {
    parameters: Parameter[];
    values: Record<string, number>;
    onChange: (name: string, value: number) => void;
}

// SimulationMonitor.tsx
interface SimulationMonitorProps {
    simulationId: string;
    onComplete: (results: Results) => void;
}

// ResultsViewer.tsx
interface ResultsViewerProps {
    results: Results;
    // Displays solution surface, training curve, error analysis
}
```

**Deliverables**:
- [ ] Problem selection with parameter forms
- [ ] Simulation creation and status polling
- [ ] Training progress visualization (loss curve)
- [ ] Solution visualization (2D slices of high-D solution)
- [ ] Pre-computed demo results

**Verification**:
1. Can complete full user flow (select -> configure -> run -> view)
2. Responsive on mobile and desktop
3. Loading states and error handling work correctly

**Definition of Done**:
- User can run simulation end-to-end through UI
- Visualizations render correctly
- Page load time <3 seconds

---

### Phase 6: Visualization Suite

**Objective**: Create rich scientific visualizations for random walks, solutions, and analysis.

**Files to Create**:
```
frontend/src/components/visualizations/
├── RandomWalkViewer.tsx     # 3D random walk trajectories
├── SolutionSurface.tsx      # 2D/3D solution heatmap
├── TrainingCurve.tsx        # Loss over time
├── ErrorAnalysis.tsx        # Error distribution
└── ConvergenceStudy.tsx     # MC convergence visualization
```

**Visualization Types**:

1. **Random Walk Trajectories**
   - 3D scatter/line plot of walk paths
   - Color by exit time
   - Animate path progression

2. **Solution Surface**
   - 2D heatmap for slice of solution
   - Interactive dimension selection for high-D
   - Contour overlay

3. **Training Dynamics**
   - Loss curve with log scale option
   - Learning rate schedule overlay
   - Gradient norm tracking

4. **Error Analysis**
   - Histogram of pointwise errors
   - Spatial error distribution
   - Comparison to analytical (if available)

**Deliverables**:
- [ ] At least 4 visualization types implemented
- [ ] Interactive controls (zoom, pan, dimension selection)
- [ ] Export to PNG/SVG

**Definition of Done**:
- Visualizations load within 1 second
- Interactive controls respond smoothly
- Works on Chrome, Firefox, Safari

---

### Phase 7: Deployment and Polish

**Objective**: Deploy to free cloud services and add finishing touches.

**Tasks**:
- [ ] Dockerize backend
- [ ] Deploy backend to Railway
- [ ] Deploy frontend to Vercel
- [ ] Set up Modal for GPU training
- [ ] Add pre-computed demo results
- [ ] Performance optimization
- [ ] Documentation completion

**Deliverables**:
- [ ] Live demo at accessible URL
- [ ] One-click local setup with Docker Compose
- [ ] Colab notebooks for experimentation

**Definition of Done**:
- Site accessible at public URL
- Demo works without GPU (pre-computed results)
- README has deployment instructions

---

## Risk Assessment

| Risk | Likelihood | Impact | Early Warning | Mitigation |
|------|-----------|--------|---------------|------------|
| MC variance too high for training | Medium | High | Loss plateaus early | Increase paths, use antithetic variates |
| MPS memory limits hit | Medium | Medium | OOM errors | Reduce batch size, use gradient checkpointing |
| React/FastAPI integration issues | Low | Medium | CORS errors | Set up proxy early, test integration in Phase 4 |
| Training doesn't converge | Medium | High | Loss doesn't decrease | Reduce learning rate, simplify problem first |
| Cloud free tier insufficient | Low | Medium | Timeouts, throttling | Start with small demos, cache results aggressively |
| High-D visualization unclear | Medium | Low | User confusion | Focus on 2D slices, add explanatory text |

### Contingency Cuts
If running behind schedule:
1. Drop live training, use pre-computed only
2. Simplify to single problem (Black-Scholes only)
3. Use basic matplotlib plots instead of Plotly
4. Skip mobile responsiveness

---

## Testing Strategy

### Testing Pyramid

```
         ┌─────────────┐
         │   E2E (5%)  │  Full user flow tests
         ├─────────────┤
         │Integration  │  API + ML integration
         │   (25%)     │
         ├─────────────┤
         │    Unit     │  Individual functions
         │   (70%)     │
         └─────────────┘
```

### Testing Framework
- **Python**: pytest + pytest-asyncio
- **React**: Jest + React Testing Library
- **E2E**: Playwright (if time permits)

### First Three Tests to Write

```python
# tests/test_brownian.py
def test_brownian_exit_time_sphere():
    """
    Brownian motion exit time from unit sphere should match
    theoretical distribution: E[τ] = 1/(2d) for d dimensions.
    """
    domain = Sphere(center=0, radius=1, dim=3)
    x0 = torch.zeros(1000, 3)  # Start at origin
    _, exit_times, _ = simulate_brownian_paths(x0, dt=0.001, domain=domain)

    expected_mean = 1 / 6  # 1/(2*3)
    actual_mean = exit_times.mean().item()
    assert abs(actual_mean - expected_mean) / expected_mean < 0.1  # 10% tolerance


def test_feynman_kac_heat_equation():
    """
    For heat equation with u(boundary) = x,
    interior solution at origin should be mean of boundary values.
    """
    # On unit interval [0,1], boundary is {0, 1}
    # g(0) = 0, g(1) = 1
    # u(0.5) should be 0.5 (mean of boundary)

    x = torch.tensor([[0.5]])
    estimate, _ = feynman_kac_estimate(
        x,
        boundary_fn=lambda x: x,  # g(x) = x
        potential_fn=lambda x: 0,  # No potential
        domain=Interval(0, 1),
        n_paths=10000
    )

    assert abs(estimate.item() - 0.5) < 0.05  # 5% tolerance


def test_pinn_forward_shape():
    """Network output shape matches input batch size."""
    model = FeynmanKacPINN(input_dim=10, hidden_dims=[32, 32])
    x = torch.randn(64, 10)
    y = model(x)
    assert y.shape == (64, 1)
```

### Validation Tests (against analytical solutions)

| Problem | Analytical Solution | Test Criterion |
|---------|---------------------|----------------|
| 1D Heat equation | Mean of boundary values | <5% relative error |
| 2D Laplace on square | Harmonic function | <5% relative error |
| 1D Black-Scholes | Black-Scholes formula | <5% relative error |
| N-D Harmonic oscillator | Gaussian ground state | <5% relative error |

---

## First Concrete Task

### File to Create First
`ml/data/brownian.py`

### Function Signature
```python
import torch
from typing import Tuple, Protocol

class Domain(Protocol):
    """Protocol for domain geometry."""
    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Returns boolean mask of points inside domain."""
        ...

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Project points outside domain to nearest boundary point."""
        ...


def simulate_brownian_paths(
    x0: torch.Tensor,
    dt: float,
    max_steps: int,
    domain: Domain,
    potential_fn: callable = None,
    device: str = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate Brownian motion paths until domain exit.

    Args:
        x0: Starting points, shape (batch_size, dim)
        dt: Time step for Euler-Maruyama discretization
        max_steps: Maximum number of steps before timeout
        domain: Domain object defining the region
        potential_fn: Optional c(x) for path integral computation
        device: Compute device ('mps', 'cuda', 'cpu'). Auto-detected if None.

    Returns:
        exit_points: Points where paths exited, shape (batch_size, dim)
        exit_times: Time of exit for each path, shape (batch_size,)
        path_integrals: ∫₀^τ c(B_s)ds for each path, shape (batch_size,)
    """
    pass  # Implementation here
```

### Starter Code

```python
# ml/data/brownian.py
import torch
from typing import Tuple, Optional, Callable

def get_device() -> str:
    """Auto-detect best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Hypercube:
    """Axis-aligned hypercube domain [low, high]^dim."""

    def __init__(self, low: float, high: float, dim: int):
        self.low = low
        self.high = high
        self.dim = dim

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Check if points are inside the hypercube."""
        return ((x >= self.low) & (x <= self.high)).all(dim=-1)

    def project_to_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp points to hypercube boundary."""
        return torch.clamp(x, self.low, self.high)


def simulate_brownian_paths(
    x0: torch.Tensor,
    dt: float,
    max_steps: int,
    domain,
    potential_fn: Optional[Callable] = None,
    device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simulate Brownian motion paths until domain exit.

    Uses Euler-Maruyama discretization:
        X_{n+1} = X_n + sqrt(dt) * Z_n
    where Z_n ~ N(0, I).
    """
    if device is None:
        device = get_device()

    batch_size, dim = x0.shape
    x0 = x0.to(device)

    # Initialize tracking tensors
    positions = x0.clone()
    exit_points = torch.zeros_like(x0)
    exit_times = torch.zeros(batch_size, device=device)
    path_integrals = torch.zeros(batch_size, device=device)

    # Track which paths have exited
    active = torch.ones(batch_size, dtype=torch.bool, device=device)

    sqrt_dt = dt ** 0.5

    for step in range(max_steps):
        if not active.any():
            break

        # Generate Brownian increments for active paths
        n_active = active.sum().item()
        dW = torch.randn(n_active, dim, device=device) * sqrt_dt

        # Update positions
        positions[active] = positions[active] + dW

        # Accumulate path integral if potential provided
        if potential_fn is not None:
            path_integrals[active] += potential_fn(positions[active]) * dt

        # Check for exits
        inside = domain.contains(positions)
        newly_exited = active & ~inside

        if newly_exited.any():
            # Record exit information
            exit_points[newly_exited] = domain.project_to_boundary(
                positions[newly_exited]
            )
            exit_times[newly_exited] = (step + 1) * dt
            active[newly_exited] = False

    # Handle paths that didn't exit (timeout)
    if active.any():
        exit_points[active] = domain.project_to_boundary(positions[active])
        exit_times[active] = max_steps * dt

    return exit_points, exit_times, path_integrals


# Quick test
if __name__ == "__main__":
    domain = Hypercube(low=0.0, high=1.0, dim=2)
    x0 = torch.full((100, 2), 0.5)  # Start in center

    exit_pts, exit_times, _ = simulate_brownian_paths(
        x0, dt=0.001, max_steps=10000, domain=domain
    )

    print(f"Device: {get_device()}")
    print(f"Mean exit time: {exit_times.mean():.4f}")
    print(f"Exit points shape: {exit_pts.shape}")
```

### Verification Method
```bash
cd ml && python -m data.brownian
# Should print device info and exit statistics without errors
```

### First Commit Message
```
feat(ml): add Brownian motion simulation engine

Implement core random walk functionality for Feynman-Kac PINN:
- Euler-Maruyama discretization for Brownian paths
- Hypercube domain with exit detection
- Path integral accumulation for potential terms
- MPS/CUDA/CPU device auto-detection

This forms the foundation for Monte Carlo PDE solution estimation.
```

---

## Appendix: Learning Resources

### Concepts to Understand Before Coding

| Concept | Why It Matters | Resource |
|---------|---------------|----------|
| Brownian Motion | Core random walk process | [3Blue1Brown](https://www.youtube.com/watch?v=aSNNDQtFUuQ) |
| Feynman-Kac Formula | Connects PDEs to expectations | [Wikipedia](https://en.wikipedia.org/wiki/Feynman%E2%80%93Kac_formula) |
| Black-Scholes Model | Target problem #1 | [Investopedia](https://www.investopedia.com/terms/b/blackscholes.asp) |
| Schrodinger Equation | Target problem #2 | [Khan Academy](https://www.khanacademy.org/science/physics/quantum-physics) |
| Monte Carlo Methods | Estimation approach | [Coursera](https://www.coursera.org/learn/mcmc-bayesian-statistics) |
| PyTorch Basics | Implementation framework | [PyTorch Tutorials](https://pytorch.org/tutorials/) |

### Papers to Skim

1. Han, Jentzen, E (2018) - "Solving high-dimensional PDEs using deep learning"
2. Beck et al. (2021) - "Deep splitting method for parabolic PDEs"
3. Raissi et al. (2019) - "Physics-informed neural networks" (for context)
