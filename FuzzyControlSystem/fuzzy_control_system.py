import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt = 0.01
T = 50
D_MU_TILDE = 1
mu_min = -np.pi / 36
mu_max = np.pi / 36
phi_min, phi_max = -np.pi, np.pi
omega_min, omega_max = -np.pi / 9, np.pi / 9
mu_tilde = -100

output_filename = 'simulation_results_algorithm_1_degrees.txt'

rules_base = [
    [1, 1, 1, 6], [2, 1, 2, 6], [3, 1, 3, 6], [4, 1, 4, 6], [5, 1, 5, 6], [6, 1, 6, 6],
    [7, 2, 1, 6], [8, 2, 2, 6], [9, 2, 3, 6], [10, 2, 4, 5], [11, 2, 5, 5], [12, 2, 6, 5],
    [13, 3, 1, 6], [14, 3, 2, 6], [15, 3, 3, 6], [16, 3, 4, 5], [17, 3, 5, 5], [18, 3, 6, 5],
    [19, 4, 1, 2], [20, 4, 2, 2], [21, 4, 3, 2], [22, 4, 4, 1], [23, 4, 5, 1], [24, 4, 6, 1],
    [25, 5, 1, 2], [26, 5, 2, 2], [27, 5, 3, 2], [28, 5, 4, 1], [29, 5, 5, 1], [30, 5, 6, 1],
    [31, 6, 1, 1], [32, 6, 2, 1], [33, 6, 3, 1], [34, 6, 4, 1], [35, 6, 5, 1], [36, 6, 6, 1]
]


def scale_phi(phi):
    return (200 / (phi_max - phi_min)) * (phi - phi_min) - 100


def scale_omega(omega):
    return (200 / (omega_max - omega_min)) * (omega - omega_min) - 100


def f_t(x, a, b, c, d):
    if x <= a or x >= d:
        return 0
    elif a <= x <= b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1
    elif c < x < d:
        return (d - x) / (d - c)


fuzzy_functions = {
    1: lambda x: f_t(x, -1000, -200, -100, -50),
    2: lambda x: f_t(x, -100, -50, -50, -10),
    3: lambda x: f_t(x, -50, -10, 0, 0),
    4: lambda x: f_t(x, 0, 0, 10, 50),
    5: lambda x: f_t(x, 10, 50, 50, 100),
    6: lambda x: f_t(x, 50, 100, 200, 1000)
}


def get_rule_parameters(n):
    return rules_base[n - 1][1], rules_base[n - 1][2], rules_base[n - 1][3]


def calculate_chi_star(phi_tilde, omega_tilde, mu_tilde, rules_count):
    chi_star = 0
    n = 1
    while n <= rules_count:
        n1, n2, m = get_rule_parameters(n)
        alpha = fuzzy_functions[n1](phi_tilde)
        beta = fuzzy_functions[n2](omega_tilde)
        gamma = min(alpha, beta)
        delta = fuzzy_functions[m](mu_tilde)
        chi = min(gamma, delta)

        if chi > chi_star:
            chi_star = chi

        n += 1

    return chi_star


def simulate_system(initial_phi, initial_omega, initial_mu_tilde):
    phi = initial_phi
    omega = initial_omega
    t = 0
    data = []

    while t < T - 0.5 * dt:
        phi_tilde = 200 / (phi_max - phi_min) * (phi - phi_min) - 100
        omega_tilde = 200 / (omega_max - omega_min) * (omega - omega_min) - 100

        mu_tilde = initial_mu_tilde
        s1, s2 = 0, 0

        while mu_tilde < 100 - 0.5 * D_MU_TILDE:
            chi_star = calculate_chi_star(phi_tilde, omega_tilde, mu_tilde, len(rules_base))

            if chi_star > 0:
                s1 += mu_tilde * chi_star * D_MU_TILDE
                s2 += chi_star * D_MU_TILDE

            mu_tilde += D_MU_TILDE

        if s2 > 0:
            mu_star_tilde = s1 / s2
            mu = (mu_star_tilde + 100) / 200 * (mu_max - mu_min) + mu_min
        else:
            mu = 0

        phi += omega * dt
        omega += mu * dt
        t += dt

        data.append([t, np.degrees(phi), np.degrees(omega), np.degrees(mu)])

    return data


def main():
    phi = np.radians(30)
    omega = np.radians(3)

    data_algorithm1 = simulate_system(phi, omega, mu_tilde)

    df = pd.DataFrame(data_algorithm1, columns=['Time(s)', 'Phi(deg)', 'Omega(deg/s)', 'Mu(deg/s^2)'])

    output_excel_filename = 'simulation_results_algorithm_1_degrees.xlsx'
    df.to_excel(output_excel_filename, index=False)

    time, angle, angular_velocity, angular_acceleration = zip(*data_algorithm1)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(time, angle, label=f'Algorithm 1 - Angle (Phi)')
    axes[0].set_ylabel('Angle (degrees)')
    axes[0].grid()

    axes[1].plot(time, angular_velocity, label=f'Algorithm 1 - Angular Velocity (Omega)', color='orange')
    axes[1].set_ylabel('Angular Velocity (deg/s)')
    axes[1].grid()

    axes[2].plot(time, angular_acceleration, label=f'Algorithm 1 - Angular Acceleration (Mu)', color='green')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angular Acceleration (deg/s^2)')
    axes[2].grid()

    plt.tight_layout()
    plt.savefig('combined_plot_degrees.png')
    plt.show()


if __name__ == '__main__':
    main()
