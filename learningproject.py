import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# Robot Parameters
v_max = 1.0  # Maximum Linear Velocity
omega_max = 2.0  # Maximum angular velocity

# Robot dynamics function
def robot_dynamics(s, m):
    x, y, theta, p1, p2, p3, z = m
    v_star = -0.3 * (p1 * np.cos(theta) + p2 * np.sin(theta))
    omega_star = -0.3 * p3
    
    # Apply velocity constraints
    v_star = np.clip(v_star, -v_max, v_max)
    omega_star = np.clip(omega_star, -omega_max, omega_max)
    
    # State derivatives
    dx_ds = z * v_star * np.cos(theta)
    dy_ds = z * v_star * np.sin(theta)
    dtheta_ds = z * omega_star
    dp1_ds = np.zeros_like(s)
    dp2_ds = np.zeros_like(s)
    dp3_ds = z * v_star * (p2 * np.cos(theta) - p1 * np.sin(theta))
    dz_ds = np.zeros_like(s)  # t_f is constant
    
    return np.vstack([dx_ds, dy_ds, dtheta_ds, dp1_ds, dp2_ds, dp3_ds, dz_ds])

# Boundary conditions function
def boundary_conditions(m0, mf, x0, xf, z_guess):
    return np.array([
        m0[0] - x0[0],  # Initial x
        m0[1] - x0[1],  # Initial y
        m0[2] - x0[2],  # Initial theta
        mf[0] - xf[0],  # Final x
        mf[1] - xf[1],  # Final y
        mf[2] - xf[2],  # Final theta
        mf[6] - z_guess  # Final time guess constraint
    ])

# Solve BVP segment
def solve_bvp_segment(x0, xf, p_guess, z_guess):
    s = np.linspace(0, 1, 200)
    initial_guess = np.zeros((7, len(s)))
    initial_guess[0, :] = np.linspace(x0[0], xf[0], len(s))
    initial_guess[1, :] = np.linspace(x0[1], xf[1], len(s))
    initial_guess[2, :] = np.linspace(x0[2], xf[2], len(s))
    initial_guess[3:6, :] = p_guess[:, None]
    initial_guess[6, :] = z_guess

  
    solution = solve_bvp(
        lambda s, m: robot_dynamics(s, m),
        lambda m0, mf: boundary_conditions(m0, mf, x0, xf, z_guess),
        s,
        initial_guess,
        tol=1e-3  
    )
    return solution

# Define waypoints 
waypoints = [
    np.array([0, 0, 0]),  
    np.array([2, 2, np.pi/4]), 
    np.array([4, 3, np.pi/6]), 
    np.array([6, 5, np.pi/3])  
]

p_guess = np.array([0.1, 0.1, 0.1])
z_guess = 5.0

# Store results
x_vals, y_vals, theta_vals = [], [], []
linear_velocities, angular_velocities = [], []

for i in range(len(waypoints) - 1):
    x0 = waypoints[i]
    xf = waypoints[i + 1]
    
  
    if i > 0:
        p_guess = solution.y[3:6, -1]
        z_guess = solution.y[6, -1]
    
    solution = solve_bvp_segment(x0, xf, p_guess, z_guess)
    
    if not solution.success:
        print(f"BVP solver failed between waypoints {i} and {i+1}")
        continue

    s = solution.x
    states = solution.y
    v_star = -0.3 * (states[3] * np.cos(states[2]) + states[4] * np.sin(states[2]))
    omega_star = -0.3 * states[5]

    # Apply velocity constraints
    v_star = np.clip(v_star, -v_max, v_max)
    omega_star = np.clip(omega_star, -omega_max, omega_max)
    
    x_vals.extend(states[0])
    y_vals.extend(states[1])
    theta_vals.extend(states[2])
    linear_velocities.extend(v_star)
    angular_velocities.extend(omega_star)

# Plot Cartesian path
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Path Followed", color='blue')
plt.scatter(*zip(*[(w[0], w[1]) for w in waypoints]), color='red', marker='o', label="Waypoints")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Path Followed by the Robot")
plt.legend()
plt.grid(True)
plt.show()

# Plot velocities
time = np.linspace(0, len(linear_velocities), len(linear_velocities))
plt.figure(figsize=(8, 6))
plt.plot(time, linear_velocities, label="Linear Velocity [m/s]", color='green')
plt.plot(time, angular_velocities, label="Angular Velocity [rad/s]", color='orange')
plt.axhline(v_max, linestyle='--', color='gray', label="Linear Velocity Limit")
plt.axhline(-v_max, linestyle='--', color='gray')
plt.axhline(omega_max, linestyle='--', color='black', label="Angular Velocity Limit")
plt.axhline(-omega_max, linestyle='--', color='black')
plt.xlabel("Time Steps")
plt.ylabel("Velocity")
plt.title("Linear and Angular Velocities")
plt.legend()
plt.grid(True)
plt.show()
