import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from uncertainties import UFloat
from uncertainties.core import AffineScalarFunc

Num = float | int
pi = np.pi

colors = ['blue', 'navy', 'dodgerblue', 'slategray', 'darkturquoise', 'darkcyan', 'aquamarine',
          'mediumseagreen', 'green', 'palegreen', 'olivedrab', 'olive', 'yellow', 'beige',
          'darkkhaki', 'khaki', 'gold', 'goldenrod', 'orange', 'tan', 'peru', 'chocolate', 'tomato',
          'red', 'darkred', 'lightcoral', 'rosybrown']


def plot_cost_evolution(
        cost_values: list, 
        result_value: float, 
        target_value: float = 0, 
        shots: int = 200,
        label: str = 'SPSA optimization',
        title_font_size: int = 20,
        legend_position: str = 'upper left',
        legend_font_size: int = 6,
        axis_font_size: int = 14,
        label_font_size: int = 18,
        axis_label: str = 'Iterations',
        hardware_cost_values: list = None,
        hardware_eval_interval: int = 2,
        hardware_label: str = 'Hardware evaluation',
        sim_style: dict = None,
        hardware_style: dict = None,
        iter_range: tuple[int, int] = None,
    ):
    """
    Plot the evolution of the cost function values during optimization, optionally including a second trace
    from hardware evaluations.

    Parameters
    ----------
    cost_values : list
        A list of cost function values over iterations (e.g. from simulator).
    result_value : float
        The final value of the cost function obtained after optimization.
    target_value : float, optional
        The target value of the cost function (default is 0).
    shots : int, optional
        The number of shots used for each optimization step (default is 200).
    title_font_size : int, optional
        Font size for the plot title. Defaults to 20.
    legend_position : str, optional
        Position of the legend on the plot. Defaults to 'upper left'.
    legend_font_size : int, optional
        Font size for the legend. Defaults to 6.
    axis_font_size : int, optional
        Font size for the numbers on the x and y axes. Defaults to 14.
    label_font_size : int, optional
        Font size for the x and y axis labels. Defaults to 18.
    axis_label : str, optional
        Label for the x axis. Defaults to 'Iterations'.
    hardware_cost_values : list, optional
        A second list of cost values (e.g. measured on hardware).
    hardware_eval_interval : int, optional
        The interval at which the hardware cost values were measured (default is 2).
    hardware_label : str, optional
        Label for the hardware cost values in the legend.
    sim_style : dict, optional
        Dictionary of plotting style options for simulator data (e.g. {'color': 'blue', 'marker': '.'}).
    hardware_style : dict, optional
        Dictionary of plotting style options for hardware data (e.g. {'color': 'red', 'marker': 'x'}).
    iter_range : tuple of int, optional
        Tuple (start, end) to select a subset of iterations to plot. Defaults to None (plot all).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.
    """
    # Set default styles if not provided
    sim_style = sim_style or {'color': 'blue', 'marker': '.'}
    hardware_style = hardware_style or {'color': 'red', 'marker': '.'}

    fig, ax = plt.subplots(1, figsize=(12, 8))

    ax.grid(True)
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5)
    
    # Determine iteration range for plotting
    if iter_range is None:
        start_idx, end_idx = 0, len(cost_values)
    else:
        start_idx, end_idx = iter_range
        # Clamp to valid range
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, len(cost_values))

    # Plot simulator cost values (subset)
    sim_x = list(range(start_idx, end_idx))
    sim_y = cost_values[start_idx:end_idx]
    ax.scatter(sim_x, sim_y, label=label, **sim_style)

    # Plot hardware cost values, if provided (subset)
    if hardware_cost_values is not None:
        # Compute hardware iteration indices (all hardware points)
        hardware_iters_all = list(range(0, hardware_eval_interval * len(hardware_cost_values), hardware_eval_interval))

        # Select hardware points within the iteration range
        hw_selected = [(x, y) for x, y in zip(hardware_iters_all, hardware_cost_values) if start_idx <= x < end_idx]
        if hw_selected:
            hardware_iters, hardware_vals = zip(*hw_selected)
        else:
            hardware_iters, hardware_vals = [], []

        # Check for uncertainties
        has_uncertainties = any(isinstance(v, (UFloat, AffineScalarFunc)) for v in hardware_vals) if hardware_vals else False

        if has_uncertainties:
            y_hardware = [v.nominal_value if isinstance(v, (UFloat, AffineScalarFunc)) else v for v in hardware_vals]
            yerr_hardware = [v.std_dev if isinstance(v, (UFloat, AffineScalarFunc)) else 0.0 for v in hardware_vals]
            ax.errorbar(hardware_iters, y_hardware, yerr=yerr_hardware, label=hardware_label,
                fmt=hardware_style.get('marker', 'o'),  # marker style
                color=hardware_style.get('color', 'red'),
                linestyle='None',
                capsize=hardware_style.get('capsize', 4))
        else:
            ax.scatter(hardware_iters, hardware_vals, label=hardware_label, **hardware_style)

    # Reference lines
    ax.axhline(target_value, color='black', linestyle='dashdot', label=r'Ideal av. fidelity')
    ax.axhline(result_value, color='green', linestyle='dashdot', label=r'SPSA(qiskit) result')

    # Set labels, title, legend
    ax.legend(loc=legend_position, fontsize=legend_font_size)
    ax.set_title('Cost Evolution', fontsize=title_font_size)

    x_min, x_max, y_min, y_max = ax.axis('tight')
    ax.axis([x_min, x_max, y_min, y_max])
    ax.set_xlabel(f"{axis_label} ({shots} cc)", fontsize=label_font_size)
    ax.set_ylabel(r'$\mathcal{F}$', fontsize=label_font_size)
    ax.tick_params(axis='both', which='major', labelsize=axis_font_size)

    return fig, ax

def plot_parameter_evolution(
        params: list, 
        index_of_params: list[int] | None = None, 
        answer_values: list[float] | None = None, 
        legend_position: str = 'upper left',
        title_font_size: int = 20,
        legend_font_size: int = 6,
        axis_font_size: int = 14,
        label_font_size: int = 18,
        axis_label: str = 'Iterations'
    ):
    """
    Plot the evolution of parameters during optimization.

    Parameters
    ----------
    params : list
        A list of parameter values over iterations.
    index_of_params : list[int], optional
        List of the parameters that one wants to plot. If None, all parameters will be plotted. Defaults to None.
    answer_values : list[float], optional
        List of values to plot horizontal lines on. Used when one wants to visualize the answer values. Defaults to None.
    legend_position : str, optional
        Position of the legend on the plot. Defaults to 'upper left'.
    legend_font_size : int, optional
        Font size of the legend. Defaults to 6.
    axis_font_size : int, optional
        Font size for the numbers on the x and y axes. Defaults to 14.
    label_font_size : int, optional
        Font size for the x and y axis labels. Defaults to 18.
    axis_label : str, optional
        Label for the x axis. Defaults to iterations.

    Returns
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.

    Description
    -----------
    This function plots the evolution of parameters during optimization. It takes a list of parameter values (`params`) recorded 
    over iterations. Each element of the list represents the evolution of a single parameter. The plot visualizes the evolution 
    of parameter values over iterations, with markers indicating each recorded value. Additionally, it includes horizontal lines 
    at y=0. The font sizes of the axis numbers and axis labels can be customized.
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))

    ax.grid(True)
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.5)

    if index_of_params:
        for i in index_of_params:
            ax.scatter(range(len(params[i])), params[i], label=r'$\theta[{}]$'.format(i), marker='.', color=colors[i])
    else:
        for i, param_i in enumerate(params):
            ax.scatter(range(len(param_i)), param_i, label=r'$\theta[{}]$'.format(i), marker='.', color=colors[i])

    if answer_values:
        for i, value in enumerate(answer_values):
            ax.axhline(value, color=colors[i], linestyle="--")
    else:
        ax.axhline(0, color="black", linestyle="--")
        ax.axhline(pi, color="black", linestyle="--")
        ax.axhline(-pi, color="black", linestyle="--")
        ax.axhline(pi / 2, color="black", linestyle="--")
        ax.axhline(-pi / 2, color="black", linestyle="--")
        ax.axhline(pi / 4, color="black", linestyle="--")
        ax.axhline(-pi / 4, color="black", linestyle="--")

    ax.legend(loc=legend_position, fontsize=legend_font_size, bbox_to_anchor=(0, 1))
    ax.set_title('Parameter Evolution', fontsize=title_font_size)

    # Set font sizes for axes
    ax.tick_params(axis='both', which='major', labelsize=axis_font_size)
    ax.set_xlabel(axis_label, fontsize=label_font_size)
    ax.set_ylabel(r'$\vec{\theta}$ (rad)', fontsize=label_font_size)

    return fig, ax

def cartesian_to_spherical(x: float, y: float, z: float) -> tuple[float]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    -----------
    x : float 
        The x-coordinate.
    y : float 
        The y-coordinate.
    z : float 
        The z-coordinate.

    Returns:
        tuple 
            The azimuthal angle theta and the polar angle phi.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return theta, phi

def draw_horizontal_cross_section_3d(theta: float, phi: float, ax: Axes3D):
    """
    Draw a horizontal cross-section on a 3D plot.

    Parameters
    -----------
    theta : float 
        The azimuthal angle theta.
    phi : float 
        The polar angle phi.
    ax : matplotlib.axes._subplots.Axes3DSubplot 
        The 3D axis object.
    """
    # Convert spherical coordinates to Cartesian coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Calculate the radius of the circle on the sphere
    radius = np.sin(phi)
    
    # Plot the horizontal circle
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    x_circle = radius * np.cos(theta_circle)
    y_circle = radius * np.sin(theta_circle)
    z_circle = z
    ax.plot(x_circle, y_circle, z_circle, color='r')
    
    # Plot the point
    ax.scatter(x, y, z, color='orange', s=50)

def density_matrix_to_bloch_vector(rho: np.ndarray) -> tuple[float]:
    """
    Convert a density matrix to Bloch vector components.

    Parameters
    -----------
    rho : numpy.ndarray 
        The density matrix.

    Returns:
    tuple[float] 
        The components of the Bloch vector.
    """

    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Calculate Bloch vector
    r_x = np.trace(np.dot(rho, sigma_x)).real
    r_y = np.trace(np.dot(rho, sigma_y)).real
    r_z = np.trace(np.dot(rho, sigma_z)).real
    
    return r_x, r_y, r_z

def is_matrix(arr) -> bool:
    """
    Check if the input is a matrix.

    Parameters
    -----------
    arr
        Input object to be checked.

    Returns
    -----------
    bool 
        True if the input is a matrix, False otherwise.
    """
    return isinstance(arr, np.ndarray) and arr.ndim == 2

def bloch_sphere(states: list):
    """
    Plot the Bloch sphere with given quantum states.

    Parameters
    -----------
    states : list
        A list of quantum states. Each state can be a pair of amplitudes or a density matrix.
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)

    # Plot x, y, z axes
    ax.plot([-1, 1], [0, 0], [0, 0], color='grey', linestyle='--')  # x-axis
    ax.plot([0, 0], [-1, 1], [0, 0], color='grey', linestyle='--')  # y-axis
    ax.plot([0, 0], [0, 0], [-1, 1], color='grey', linestyle='--')  # z-axis

    # Plot the |0> and |1> states
    ax.scatter(0, 0, 1, color='b', s=100, label='|0>')
    ax.text(0, 0, 1, '|0>', color='b', fontsize=12)
    ax.scatter(0, 0, -1, color='g', s=100, label='|1>')
    ax.text(0, 0, -1, '|1>', color='g', fontsize=12)

    for state in states:
        # Calculate the coordinates of the vector
        if not is_matrix(state):
            # If the input is a pair of amplitudes
            amplitude_0, amplitude_1 = state
            x_vec = 2 * (amplitude_0 * np.conj(amplitude_1)).real
            y_vec = 2 * (amplitude_0 * np.conj(amplitude_1)).imag
            z_vec = np.abs(amplitude_0)**2 - np.abs(amplitude_1)**2
        elif isinstance(state, np.ndarray):
            # If the input is a density matrix
            x_vec, y_vec, z_vec = density_matrix_to_bloch_vector(state)

        # Convert to spherical coordinates
        theta, phi = cartesian_to_spherical(x_vec, y_vec, z_vec)

        # Plot the horizontal cross-section
        draw_horizontal_cross_section_3d(theta, phi, ax)

        # Plot the vector
        ax.quiver(0, 0, 0, x_vec, y_vec, z_vec, color='k', arrow_length_ratio=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere')
    ax.legend()

    plt.show()
