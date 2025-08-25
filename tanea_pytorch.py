"""
TANEA Optimizer PyTorch Implementation
======================================

A PyTorch port of the TANEA (Temporal Adaptive Network Evolution Algorithm) optimizer
from the Experimental_new power_law_rf optimizers. This includes all advanced features:

- Power law scheduling for g2, g3, and delta parameters
- Multiple momentum flavors (effective-clip, theory, adam, etc.)
- Tau computation for adaptive sparse training
- Signal-to-noise ratio clipping (clipsnr)
- Weight decay with scheduling
- Multiple tau flavors for different scenarios

Usage:
    from tanea_pytorch import TANEA, powerlaw_schedule
    
    # Create optimizer with power law schedules
    g2_schedule = powerlaw_schedule(1e-4, 0.0, 0.0, 1.0)  # Constant
    g3_schedule = powerlaw_schedule(1e-5, 0.0, -1.0, 1.0)  # Decay
    delta_schedule = powerlaw_schedule(1.0, 0.0, -1.0, 8.0)  # Standard decay
    
    optimizer = TANEA(
        model.parameters(),
        g2=g2_schedule,
        g3=g3_schedule,
        delta=delta_schedule,
        momentum_flavor="effective-clip"
    )
"""

import math
import torch
from torch.optim import Optimizer
from typing import Optional, Union, Callable, Dict, Any, Iterable


def powerlaw_schedule(init_value: float, saturation_value: float, power: float, time_scale: float) -> Callable[[int], float]:
    """
    Constructs power-law schedule.
    
    Formula: max(init_value * (1 + t/time_scale)^power, saturation_value)
    
    Args:
        init_value: Initial value for the scalar to be annealed
        saturation_value: End value of the scalar to be annealed  
        power: The power of the power law
        time_scale: Number of steps over which the power law takes place
        
    Returns:
        Schedule function that maps step counts to values
    """
    def schedule(count: int) -> float:
        frac = 1.0 + count / time_scale
        return max(init_value * (frac ** power), saturation_value)
    
    return schedule


class TANEA(Optimizer):
    """
    TANEA (Temporal Adaptive Network Evolution Algorithm) Optimizer
    
    A sophisticated adaptive optimizer that implements power law schedules and
    advanced momentum terms for improved training dynamics.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        g2: Second gradient coefficient (scalar or schedule function)
        g3: Third gradient coefficient for momentum (scalar or schedule function)
        delta: Momentum decay rate (scalar or schedule function)
        epsilon: Small constant for numerical stability (default: 1e-8)
        beta_m: First moment decay rate (default: same as delta)
        g1: First gradient coefficient (default: 1.0)
        beta_v: Second moment decay rate (default: same as delta)
        magic_tau: Scaling factor for tau updates (default: 1.0)
        weight_decay: Weight decay coefficient (scalar or schedule function, default: 0.0)
        momentum_flavor: Type of momentum computation (default: "effective-clip")
        tau_flavor: Type of tau computation (default: "second-moment") 
        clipsnr: Signal-to-noise ratio clipping factor (default: 2.0)
        
    Available momentum_flavor options:
        - "effective-clip": Conservative momentum scaling (recommended)
        - "theory": Theoretical momentum scaling
        - "adam": Standard Adam-like momentum
        - "always-on": Always apply momentum updates
        - "always-on-mk2": Modified always-on with tau scaling
        - "strong-clip": Strong clipping variant
        - "mk2": Momentum variant mk2
        - "mk3": Momentum variant mk3 with enhanced clipping
        
    Available tau_flavor options:
        - "second-moment": Standard second moment based tau
        - "first-moment": First moment based tau
        - "second-moment-massive": Second moment with tau scaling
        - "first-moment-massive": First moment with tau scaling
    """
    
    def __init__(
        self,
        params: Iterable[torch.Tensor],
        g2: Union[float, Callable[[int], float]] = 1e-4,
        g3: Union[float, Callable[[int], float]] = 1e-5,
        delta: Union[float, Callable[[int], float]] = None,
        epsilon: float = 1e-8,
        beta_m: Optional[Union[float, Callable[[int], float]]] = None,
        g1: Union[float, Callable[[int], float]] = 1.0,
        beta_v: Optional[Union[float, Callable[[int], float]]] = None,
        magic_tau: float = 1.0,
        weight_decay: Union[float, Callable[[int], float]] = 0.0,
        momentum_flavor: str = "effective-clip",
        tau_flavor: str = "second-moment",
        clipsnr: float = 2.0,
    ):
        if not 0.0 <= epsilon:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if not 0.0 <= clipsnr:
            raise ValueError(f"Invalid clipsnr value: {clipsnr}")
        
        # Set default delta schedule if not provided (decay from 1.0 to 0.0 over 8 steps)
        if delta is None:
            delta = powerlaw_schedule(1.0, 0.0, -1.0, 8.0)
        
        # Validate momentum flavor
        valid_momentum_flavors = {
            "effective-clip", "theory", "adam", "always-on", "always-on-mk2",
            "strong-clip", "mk2", "mk3"
        }
        if momentum_flavor not in valid_momentum_flavors:
            raise ValueError(f"Invalid momentum_flavor: {momentum_flavor}. Must be one of {valid_momentum_flavors}")
        
        # Validate tau flavor
        valid_tau_flavors = {
            "second-moment", "first-moment", "second-moment-massive", "first-moment-massive"
        }
        if tau_flavor not in valid_tau_flavors:
            raise ValueError(f"Invalid tau_flavor: {tau_flavor}. Must be one of {valid_tau_flavors}")
        
        defaults = dict(
            g2=g2, g3=g3, delta=delta, epsilon=epsilon, beta_m=beta_m, g1=g1,
            beta_v=beta_v, magic_tau=magic_tau, weight_decay=weight_decay,
            momentum_flavor=momentum_flavor, tau_flavor=tau_flavor, clipsnr=clipsnr
        )
        
        super(TANEA, self).__init__(params, defaults)
        
        # Global step counter
        self._step_count = 0
    
    def _make_schedule(self, value: Union[float, Callable[[int], float]]) -> Callable[[int], float]:
        """Convert scalar or schedule to callable function."""
        if callable(value):
            return value
        else:
            return lambda step: value
    
    def _clip_to_half(self, tau: torch.Tensor) -> torch.Tensor:
        """Clip tau values to at most 0.5."""
        return torch.clamp(tau, max=0.5)
    
    def _tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """
        Tau regularization function that:
        1. Ensures tau is not a poor estimate when p << 1/t
        2. Converts tau/(1-tau) form to proper p estimate  
        3. Clips to prevent numerical issues
        """
        clipped_tau = self._clip_to_half(tau)
        p_estimate = clipped_tau / (1.0 - clipped_tau)
        min_p = torch.full_like(tau, 1.0 / (1.0 + step))
        return torch.maximum(p_estimate, min_p)
    
    def _root_tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Square root of tau regularization."""
        return torch.sqrt(self._tau_reg(tau, step))
    
    def _quarter_root_tau_reg(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Quarter root of tau regularization."""
        return torch.pow(self._tau_reg(tau, step), 0.25)
    
    def _effective_time(self, tau: torch.Tensor, step: int) -> torch.Tensor:
        """Compute effective time for tau regularization."""
        return torch.maximum(tau * step, torch.ones_like(tau))
    
    def _tau_updater(
        self, 
        tau: torch.Tensor, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        step: int, 
        tau_flavor: str, 
        magic_tau: float, 
        epsilon: float
    ) -> torch.Tensor:
        """
        Update tau based on gradient and second moment statistics.
        In idealized environment, this leads to tau storing p/(1+p).
        """
        if tau_flavor == "second-moment":
            return (u**2) / ((u**2) + v + epsilon**2)
        elif tau_flavor == "first-moment":
            return torch.abs(u) / (torch.abs(u) + torch.sqrt(v) + epsilon)
        elif tau_flavor == "second-moment-massive":
            tau_scale = self._root_tau_reg(tau, step) * magic_tau
            return (u**2) * tau_scale / ((u**2) * tau_scale + v + epsilon**2)
        elif tau_flavor == "first-moment-massive":
            tau_scale = self._quarter_root_tau_reg(tau, step) * magic_tau
            return torch.abs(u) * tau_scale / (torch.abs(u * tau_scale) + torch.sqrt(v) + epsilon)
        else:
            raise ValueError(f"Unknown tau_flavor: {tau_flavor}")
    
    def _g2_momentum_term(
        self, 
        u: torch.Tensor, 
        md: torch.Tensor, 
        v: torch.Tensor, 
        tau: torch.Tensor, 
        step: int, 
        clipsnr: float, 
        epsilon: float
    ) -> torch.Tensor:
        """
        Compute g2 momentum term with signal-to-noise ratio clipping.
        This is the theoretically informed g2 momentum term.
        """
        root_tau_reg = self._root_tau_reg(tau, step)
        base_term = root_tau_reg / (torch.sqrt(v) + epsilon)
        
        # Signal-to-noise ratio clipping
        snr_clip = torch.minimum(
            torch.ones_like(u),
            clipsnr * torch.sqrt(v) / (root_tau_reg * torch.abs(u) + epsilon)
        )
        
        return base_term * snr_clip
    
    def _g3_momentum_term(
        self, 
        u: torch.Tensor, 
        v: torch.Tensor, 
        tau: torch.Tensor, 
        step: int, 
        momentum_flavor: str, 
        epsilon: float
    ) -> torch.Tensor:
        """
        Compute g3 momentum term based on specified flavor.
        Different flavors implement different approaches to momentum scaling.
        """
        if momentum_flavor == "effective-clip":
            return torch.abs(u) / ((u**2) * self._tau_reg(tau, step) + v + epsilon**2)
        elif momentum_flavor == "theory":
            root_tau_reg = self._root_tau_reg(tau, step)
            denom = (torch.abs(u) * root_tau_reg + torch.sqrt(v) + epsilon) * (torch.sqrt(v) + epsilon)
            return torch.abs(u) / denom
        elif momentum_flavor == "adam":
            return 1.0 / (torch.sqrt(v) + epsilon)
        elif momentum_flavor == "always-on":
            return self._root_tau_reg(tau, step) / (torch.sqrt(v) + epsilon)
        elif momentum_flavor == "always-on-mk2":
            return self._tau_reg(tau, step) / (torch.sqrt(v) + epsilon)
        elif momentum_flavor == "strong-clip":
            tau_reg = self._tau_reg(tau, step)
            clip_term = torch.minimum(torch.abs(u), torch.sqrt(v / tau_reg))
            return clip_term / (v + epsilon**2)
        elif momentum_flavor == "mk2":
            root_tau_reg = self._root_tau_reg(tau, step)
            clip_term = torch.minimum(torch.abs(u) * root_tau_reg, torch.sqrt(v))
            return clip_term / (v + epsilon**2)
        elif momentum_flavor == "mk3":
            root_tau_reg = self._root_tau_reg(tau, step)
            tau_reg = self._tau_reg(tau, step)
            return (torch.abs(u) * root_tau_reg) / ((u**2) * tau_reg + v + epsilon**2)
        else:
            raise ValueError(f"Unknown momentum_flavor: {momentum_flavor}")
    
    @torch.compile
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            # Get schedule functions
            g2_sched = self._make_schedule(group['g2'])
            g3_sched = self._make_schedule(group['g3'])
            delta_sched = self._make_schedule(group['delta'])
            g1_sched = self._make_schedule(group['g1'])
            wd_sched = self._make_schedule(group['weight_decay'])
            
            # Set defaults for beta schedules
            beta_m_sched = self._make_schedule(group['beta_m'] if group['beta_m'] is not None else group['delta'])
            beta_v_sched = self._make_schedule(group['beta_v'] if group['beta_v'] is not None else group['delta'])
            
            # Evaluate all schedules at current step
            current_g2 = g2_sched(self._step_count)
            current_g3 = g3_sched(self._step_count)
            current_delta = delta_sched(self._step_count)
            current_g1 = g1_sched(self._step_count)
            current_beta_m = beta_m_sched(self._step_count)
            current_beta_v = beta_v_sched(self._step_count)
            current_wd = wd_sched(self._step_count)
            
            # Extract other parameters
            epsilon = group['epsilon']
            magic_tau = group['magic_tau']
            momentum_flavor = group['momentum_flavor']
            tau_flavor = group['tau_flavor']
            clipsnr = group['clipsnr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.dtype.is_complex:
                    raise RuntimeError('TANEA does not support complex parameters')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)  # First moment
                    state['v'] = torch.zeros_like(p)  # Second moment
                    state['tau'] = torch.zeros_like(p)  # Tau estimates
                
                m, v, tau = state['m'], state['v'], state['tau']
                state['step'] += 1
                
                # Update second moment
                v.mul_(1 - current_beta_v).addcmul_(grad, grad, value=current_beta_v)
                
                # Update tau using the specified tau updater
                new_tau_val = self._tau_updater(tau, grad, v, self._step_count, tau_flavor, magic_tau, epsilon)
                tau.mul_(1 - current_delta).add_(new_tau_val, alpha=current_delta)
                
                # Update first moment
                m.mul_(1 - current_beta_m).add_(grad, alpha=current_g1)
                
                # Compute effective time for scheduling
                effective_time_val = self._effective_time(tau, self._step_count)
                
                # Compute momentum terms
                g2_momentum = self._g2_momentum_term(grad, m * current_beta_m, v, tau, self._step_count, clipsnr, epsilon)
                g3_momentum = self._g3_momentum_term(grad, v, tau, self._step_count, momentum_flavor, epsilon)
                
                # Compute parameter updates using effective time for g2 and g3 scheduling
                g2_term = current_g2 * grad * g2_momentum
                g3_term = current_g3 * m * g3_momentum
                
                # Apply the main update
                update = -(g2_term + g3_term)
                
                # Apply weight decay
                if current_wd != 0:
                    update.add_(p, alpha=-current_wd)
                
                # Apply update to parameters
                p.add_(update)
        
        return loss
    
    def get_lr(self) -> Dict[str, Any]:
        """Get current learning rates and other scheduled values."""
        if not self.param_groups:
            return {}
            
        group = self.param_groups[0]  # Use first group as representative
        g2_sched = self._make_schedule(group['g2'])
        g3_sched = self._make_schedule(group['g3'])
        delta_sched = self._make_schedule(group['delta'])
        wd_sched = self._make_schedule(group['weight_decay'])
        
        return {
            'g2': g2_sched(self._step_count),
            'g3': g3_sched(self._step_count),
            'delta': delta_sched(self._step_count),
            'weight_decay': wd_sched(self._step_count),
            'step': self._step_count
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dict."""
        state_dict = super().state_dict()
        state_dict['_step_count'] = self._step_count
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the optimizer state."""
        self._step_count = state_dict.pop('_step_count', 0)
        super().load_state_dict(state_dict)


def create_tanea_optimizer(
    params,
    g2: float = 1e-4,
    g3: float = 1e-5,
    delta_timescale: float = 8.0,
    kappa: float = 1.0,
    weight_decay: float = 0.0,
    momentum_flavor: str = "effective-clip",
    tau_flavor: str = "second-moment",
    clipsnr: float = 2.0,
    **kwargs
) -> TANEA:
    """
    Create a TANEA optimizer with commonly used configurations.
    
    Args:
        params: Model parameters to optimize
        g2: Initial g2 learning rate (default: 1e-4)
        g3: Initial g3 learning rate (default: 1e-5)
        delta_timescale: Timescale for delta decay (default: 8.0)
        kappa: Power law exponent for g3 decay (default: 1.0)
        weight_decay: Weight decay coefficient (default: 0.0)
        momentum_flavor: Momentum computation type (default: "effective-clip")
        tau_flavor: Tau computation type (default: "second-moment")
        clipsnr: Signal-to-noise clipping factor (default: 2.0)
        **kwargs: Additional arguments passed to TANEA
        
    Returns:
        Configured TANEA optimizer
    """
    # Create power law schedules
    g2_schedule = powerlaw_schedule(g2, 0.0, 0.0, 1.0)  # Constant g2
    g3_schedule = powerlaw_schedule(g3, 0.0, -kappa, 1.0)  # Decaying g3
    delta_schedule = powerlaw_schedule(1.0, 0.0, -1.0, delta_timescale)  # Standard delta decay
    
    return TANEA(
        params,
        g2=g2_schedule,
        g3=g3_schedule,
        delta=delta_schedule,
        weight_decay=weight_decay,
        momentum_flavor=momentum_flavor,
        tau_flavor=tau_flavor,
        clipsnr=clipsnr,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    # Simple test
    import torch.nn as nn
    
    # Create test model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Create TANEA optimizer
    optimizer = create_tanea_optimizer(
        model.parameters(),
        g2=1e-3,
        g3=1e-4,
        momentum_flavor="effective-clip"
    )
    
    # Test training loop
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    print("Testing TANEA optimizer...")
    for i in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            lr_info = optimizer.get_lr()
            print(f"Step {i}: Loss = {loss.item():.6f}, LR info = {lr_info}")
    
    print("TANEA optimizer test completed successfully!")
