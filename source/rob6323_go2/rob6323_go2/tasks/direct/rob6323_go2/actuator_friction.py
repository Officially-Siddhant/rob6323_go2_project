import torch

def apply_actuator_friction(
    tau_pd: torch.Tensor,
    qd: torch.Tensor,
    Fs: torch.Tensor,
    mu_v: torch.Tensor,
    vel_scale: float = 0.1,
):
    # Friction torques
    tau_stiction = Fs * torch.tanh(qd / vel_scale)
    tau_viscous  = mu_v * qd
    tau_friction = tau_stiction + tau_viscous

    # Commanded torque
    tau_cmd = tau_pd - tau_friction
    return tau_cmd, tau_friction