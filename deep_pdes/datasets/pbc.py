from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class OneDimConvection:
    def __init__(self, beta: float, source: float = 0):
        self.beta = beta
        self.source = source

    def calc_over_grid(self, n_x: int, n_t: int):
        x = np.linspace(0, 2 * np.pi, n_x, endpoint=False)
        t = np.linspace(0, 1, n_t)
        x = x.reshape(-1, 1)
        t = t.reshape(-1, 1)
        u_vals = convection_diffusion(0, self.beta, self.source, n_x, n_t)
        u_star = u_vals.reshape(-1, 1)  # Exact solution reshaped into (n, 1)
        u = u_star.reshape(len(t), len(x))  # Exact on the (x,t) grid
        return u, x, t


def main(plot_path_str: str = None, beta=5.0, xgrid=256, nt=100):
    one_dim_convection = OneDimConvection(beta=beta)
    u, x, t = one_dim_convection.calc_over_grid(xgrid, nt)

    if plot_path_str is not None:
        plot_path = Path(plot_path_str)
    else:
        plot_path = Path("./plots")
    if not plot_path.exists():
        plot_path.mkdir(parents=True)
    exact_u(u, x, t, path=plot_path / "exact.pdf")


def convection_diffusion(nu, beta, source=0, nx=256, nt=100):
    h = 2 * np.pi / nx
    x = np.arange(0, 2 * np.pi, h)  # not inclusive of the last point
    t = np.linspace(0, 1, nt).reshape(-1, 1)
    _, t_grid = np.meshgrid(x, t)
    u0 = np.sin(x)

    return convection_diffusion_solution(u0, t_grid, nu, beta).flatten()


def convection_diffusion_solution(  # noqa: WPS210
    x_start: np.ndarray,
    t_values: np.ndarray,
    nu: float,
    beta: float,
    source: float = 0,
):
    nx = x_start.shape[0]
    forcing_term = np.zeros_like(x_start) + source  # G is the same size as u0

    ikx_pos = 1j * np.arange(0, nx / 2 + 1, 1)
    ikx_neg = 1j * np.arange(-nx / 2 + 1, 0, 1)  # noqa: WPS221
    ikx = np.concatenate((ikx_pos, ikx_neg))
    ikx2 = ikx * ikx

    uhat0 = np.fft.fft(x_start)
    nu_term = nu * ikx2 * t_values
    beta_term = beta * ikx * t_values
    nu_factor = np.exp(nu_term - beta_term)
    uhat = (
        uhat0 * nu_factor + np.fft.fft(forcing_term) * t_values
    )  # for constant, fft(p) dt = fft(p)*T
    return np.real(np.fft.ifft(uhat))


def exact_u(u, x, t, path):
    fig = plt.figure(figsize=(9, 5))
    sp = 111
    ax = fig.add_subplot(sp)

    h = ax.imshow(
        u.T,
        interpolation="nearest",
        cmap="rainbow",
        extent=[t.min(), t.max(), x.min(), x.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(h, cax=cax)
    labelsize = 15
    cbar.ax.tick_params(labelsize=labelsize)

    fontsize = 30
    ax.set_xlabel("t", fontweight="bold", size=fontsize)
    ax.set_ylabel("x", fontweight="bold", size=fontsize)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={"size": 15},
    )
    ax.tick_params(labelsize=labelsize)

    plt.savefig(path)
    plt.close()


if __name__ == "__main__":
    main()
