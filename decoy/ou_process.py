import numpy as np
from dataclasses import dataclass

@dataclass
class OUProcess():
    theta: float 
    mu: float 
    sigma: float 
    dt: float 
    rng: np.random.Generator

    def __post_init__(self):
        if self.theta <= 0:
            raise ValueError("we need to dev. to mean!")
        if self.dt <= 0:
            raise ValueError
        if self.sigma < 0:
            raise ValueError("please avoid that")

    def step(self, x):
        """ advance OU """
        epsilon = self.rng.normal()
        drift = self.theta * (self.mu - x) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * epsilon
        x_next = x + drift + diffusion
        return x_next

    def simulate(self, x0, n_steps):
        """ simulate OU process for disc time steps. """
        traj = np.empty(n_steps + 1)
        traj[0] = x0
        for t in range(n_steps):
            traj[t + 1] = self.step(traj[t])
        return traj

if __name__ ==  "__main__":
    rng = np.random.default_rng(3)
    proc = OUProcess(
    theta=0.3,
    mu=0.0,
    sigma=1.0,
    dt=0.01,
    rng = rng
    )
    trajectory = proc.simulate(x0=5.0, n_steps=1000)
    print(trajectory[:10])
    print(trajectory.shape)



