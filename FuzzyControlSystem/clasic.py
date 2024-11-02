import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
T = 50

phi = np.radians(30)
omega = np.radians(3)

S1 = complex(-1, 1)
S2 = complex(-1, -1)

k1 = (S1 + S2).real
k2 = (-S1 * S2).real


def simulate_classic_system(initial_phi, initial_omega, k1, k2):
    phi = initial_phi
    omega = initial_omega
    t = 0
    data = []

    while t < T - 0.5 * dt:
        mu = k1 * omega + k2 * phi

        phi += omega * dt
        omega += mu * dt
        t += dt

        data.append([t, np.degrees(phi), np.degrees(omega), np.degrees(mu)])

    return data


data_classic = simulate_classic_system(phi, omega, k1, k2)

output_filename = 'simulation_results_classic_algorithm_degrees.txt'
with open(output_filename, 'w') as file:
    file.write('Time(s)\tPhi(deg)\tOmega(deg/s)\tMu(deg/s^2)\n')
    for row in data_classic:
        file.write('\t'.join([f"{value:.4f}" for value in row]) + '\n')

time, angle, angular_velocity, angular_acceleration = zip(*data_classic)

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(time, angle, label='Angle (Phi) [deg]', color='blue')
ax.plot(time, angular_velocity, label='Angular Velocity (Omega) [deg/s]', color='orange')
ax.plot(time, angular_acceleration, label='Angular Acceleration (Mu) [deg/s^2]', color='green')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Values (deg, deg/s, deg/s^2)')
ax.grid()
ax.legend()

plt.tight_layout()
plt.savefig('combined_classic_plot_degrees.png')
plt.show()
