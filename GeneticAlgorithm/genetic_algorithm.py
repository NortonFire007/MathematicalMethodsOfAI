import random
import math
from typing import Tuple, List
import matplotlib.pyplot as plt
from collections import Counter

POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.8
X_RANGE = (-3, 3)
Y_RANGE = (0, 4)
PRECISION = 1

a, b = X_RANGE
c, d = Y_RANGE


def fitness_function(x, y):
    return -x ** 2 + 2 * y ** 2


def calculate_chromosome_length_and_quantization_step(
        range_min: float, range_max: float, precision: int
) -> Tuple[int, float]:
    chromosome_length: int = math.ceil(math.log2((range_max - range_min) * 10 ** precision + 1))
    quantization_step: float = (range_max - range_min) / (2 ** chromosome_length - 1)
    return chromosome_length, quantization_step


def initialize_population(population_size: int, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> List[
    Tuple[float, float]]:
    return [(random.uniform(x_range[0], x_range[1]), random.uniform(y_range[0], y_range[1])) for _ in
            range(population_size)]


def calculate_chromosome_value(
        value: float, range_min: float, quantization_step: float
) -> int:
    return round((value - range_min) / quantization_step)


def calculate_population_fitness(
        population: List[Tuple[float, float]], hx: float, hy: float, a: float, c: float
) -> Tuple[List[Tuple[int, int]], List[float], float]:
    fitness_values = []
    total_fitness: float = 0
    chromosome_values: List[Tuple[int, int]] = []

    for x, y in population:
        chx = calculate_chromosome_value(x, a, hx)
        chy = calculate_chromosome_value(y, c, hy)
        chromosome_values.append((chx, chy))
        fitness = fitness_function(x, y)
        fitness_values.append(fitness)
        total_fitness += fitness

    average_fitness: float = total_fitness / len(population)
    return chromosome_values, fitness_values, average_fitness


def adjust_fitness(fitness_values: List[float]) -> List[float]:
    f_min = min(fitness_values)
    adjustment = abs(f_min) * 2
    adjusted_fitness = [f + adjustment for f in fitness_values]
    return adjusted_fitness


def plot_selection_intervals(selection_percentages: List[float], selected_indices: List[int]):
    intervals = []
    start = 0
    for percentage in selection_percentages:
        end = start + percentage
        intervals.append((start, end))
        start = end

    plt.figure(figsize=(36, 14))
    colors = [f"C{i}" for i in range(len(intervals))]

    for i, (start, end) in enumerate(intervals):
        plt.barh(0, end - start, left=start, color=colors[i], edgecolor='black',
                 label=f"Ind{i + 1}: {start:.2f}-{end:.2f}%")

    for i in range(0, 101, 1):
        plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.5)

    for i, (start, end) in enumerate(intervals):
        plt.text((start + end) / 2, 0, f"Ind{i + 1}", rotation=90, va='center', ha='center', fontsize=10, color='black')

    for idx in selected_indices:
        start, end = intervals[idx]
        plt.barh(0, end - start, left=start, color='orange', edgecolor='black')

    plt.xlabel("Проценты")
    plt.yticks([])
    plt.title("Интервалы выбора каждой особи")
    plt.xlim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def roulette_selection(adjusted_fitness: List[float], population: List[Tuple[float, float]], num_selections: int) -> \
        Tuple[List[Tuple[float, float]], List[int], List[float]]:
    """Виконує селекцію методом рулетки з виведенням графіка."""
    total_fitness = sum(adjusted_fitness)
    selection_probabilities = [f / total_fitness * 100 for f in adjusted_fitness]

    random_numbers = [random.uniform(0, 100) for _ in range(num_selections)]
    selected_indices = []

    intervals = []
    start = 0
    for probability in selection_probabilities:
        end = start + probability
        intervals.append((start, end))
        start = end

    selected_individuals = []
    for rnd in random_numbers:
        for i, (start, end) in enumerate(intervals):
            if start <= rnd < end:
                selected_individuals.append(population[i])
                print(f"Сгенерированное число: {rnd:.2f} попало в интервал S{i + 1} [{start:.2f}; {end:.2f}]")
                selected_indices.append(i)
                break

    print('selected_individuals', selected_individuals)
    print('selected_indices', selected_indices)
    print('selection_probabilities', selection_probabilities)

    return selected_individuals, selected_indices, selection_probabilities


def generate_pairs(selected_indices: List[int]) -> List[Tuple[str, str]]:
    new_parent_population = [f"Ind{i + 1}" for i in selected_indices]

    pairs = []
    for i in range(0, len(new_parent_population), 2):
        if i + 1 < len(new_parent_population):
            pairs.append((new_parent_population[i], new_parent_population[i + 1]))
        else:
            pairs.append((new_parent_population[i], None))

    return pairs


# ---------------------------------------------------------------------------------
def single_crossover(part1: int, part2: int, length: int, k: int) -> Tuple[int, int]:
    child1 = (part1 >> (length - k)) << (length - k) | (part2 & ((1 << (length - k)) - 1))
    child2 = (part2 >> (length - k)) << (length - k) | (part1 & ((1 << (length - k)) - 1))
    return child1, child2


def crossover(parent1: Tuple[int, int], parent2: Tuple[int, int], lx: int, ly: int, kx: int, ky: int) -> \
        Tuple[Tuple[int, int], Tuple[int, int]]:
    chx1, chx2 = single_crossover(parent1[0], parent2[0], lx, kx)
    chy1, chy2 = single_crossover(parent1[1], parent2[1], ly, ky)

    return (chx1, chy1), (chx2, chy2)


def generate_mutation_mask(length: int, mutation_rate: float) -> int:
    return sum((1 << i) for i in range(length) if random.random() < mutation_rate)


def mutate(chromosome: Tuple[int, int], mutation_rate: float, lx: int, ly: int) -> Tuple[int, int]:
    mutation_mask_x = generate_mutation_mask(lx, mutation_rate)
    mutation_mask_y = generate_mutation_mask(ly, mutation_rate)

    return chromosome[0] ^ mutation_mask_x, chromosome[1] ^ mutation_mask_y


def apply_crossover_and_mutation(parent_pairs: List[Tuple[str, str]], chromosome_values: List[Tuple[int, int]],
                                 mutation_rate: float, lx: int, ly: int) -> List[Tuple[int, int]]:
    kx = random.randint(1, lx - 1)
    ky = random.randint(1, ly - 1)
    print(f'Locus X: {kx}, Locus Y: {ky}')

    offspring_population = []

    for parent1, parent2 in parent_pairs:
        if parent2 is None:
            continue

        p1_idx = int(parent1[3:]) - 1
        p2_idx = int(parent2[3:]) - 1

        parent1_chromosome = chromosome_values[p1_idx]
        parent2_chromosome = chromosome_values[p2_idx]

        if p1_idx == p2_idx:
            print(f"Идентичные особи: {parent1} и {parent2}, выполнение принудительной мутации.")
            offspring1 = mutate(parent1_chromosome, 1.0, lx, ly)
            offspring2 = mutate(parent2_chromosome, 1.0, lx, ly)
        else:
            offspring1, offspring2 = crossover(parent1_chromosome, parent2_chromosome, lx, ly, kx, ky)

            offspring1 = mutate(offspring1, mutation_rate, lx, ly)
            offspring2 = mutate(offspring2, mutation_rate, lx, ly)

        offspring_population.extend([offspring1, offspring2])

    return offspring_population


def decode_chromosome(chromosome: Tuple[int, int], a: float, c: float, hx: float, hy: float) -> Tuple[float, float]:
    x = a + chromosome[0] * hx
    y = c + chromosome[1] * hy
    return x, y


def get_best_individual(chromosome_values: List[Tuple[int, int]], average_fitness: float,
                        a: float, c: float, hx: float, hy: float) -> Tuple[Tuple[float, float], float]:
    decoded_individuals = [decode_chromosome(chromosome, a, c, hx, hy) for chromosome in chromosome_values]
    fitness_scores = [fitness_function(x, y) for x, y in decoded_individuals]
    print('fitness_scores', fitness_scores)

    best_index = fitness_scores.index(max(fitness_scores))

    return decoded_individuals[best_index], fitness_scores[best_index]


def main():
    generations = []
    average_fitness_values = []
    best_fitness_values = []
    best_individuals = []
    global_best_individual = None
    global_best_fitness = float('-inf')

    random.seed(17)

    population = initialize_population(POPULATION_SIZE, X_RANGE, Y_RANGE)

    print("Початкова популяція:")
    for i, (x, y) in enumerate(population):
        print(f"Ind{i + 1}: x={x:.2f}, y={y:.2f}")

    Lx, hx = calculate_chromosome_length_and_quantization_step(a, b, PRECISION)
    print(f"Довжина хромосоми для x (Lx): {Lx} біт")
    print(f"Фактичний крок квантування для x: {hx:.2f}")

    Ly, hy = calculate_chromosome_length_and_quantization_step(c, d, PRECISION)
    print(f"Довжина хромосоми для y (Ly): {Ly} біт")
    print(f"Фактичний крок квантування для y: {hy:.2f}")

    for generation in range(NUM_GENERATIONS):
        print(f"Поколение {generation + 1}")

        chromosome_values, fitness_values, average_fitness = calculate_population_fitness(population, hx, hy, a, c)

        average_fitness_values.append(average_fitness)

        print("\nХромосоми популяції:")
        for i, (chx, chy) in enumerate(chromosome_values):
            print(f"Ind{i + 1}: chx={chx:0{Lx}b}, chy={chy:0{Ly}b} (десяткове: chx={chx}, chy={chy})")

        print(f"\nСередня пристосованість популяції: {average_fitness:.2f}")
        adjusted_fitness = adjust_fitness(fitness_values)

        selected_individuals, selected_indices, selection_probabilities = roulette_selection(adjusted_fitness,
                                                                                             population,
                                                                                             POPULATION_SIZE)

        print("\nВідібрані особини (методом рулетки):")
        for i, (x, y) in enumerate(selected_individuals):
            print(f"Ind{selected_indices[i] + 1}: x={x:.2f}, y={y:.2f}")

        parent_pairs = generate_pairs(selected_indices)

        print("\nРозподіл нової батьківської популяції по парах:")
        for i, (ind1, ind2) in enumerate(parent_pairs, start=1):
            if ind2 is not None:
                print(f"Пара {i}: {ind1} - {ind2}")
            else:
                print(f"Пара {i}: {ind1} (непарна особа)")

        offspring_population = apply_crossover_and_mutation(parent_pairs, chromosome_values, MUTATION_RATE, Lx, Ly)

        print("\nПотомки после кроссовера и мутации:")
        for i, (chx, chy) in enumerate(offspring_population):
            print(f"Offspring{i + 1}: chx={chx:0{Lx}b}, chy={chy:0{Ly}b}")

        # -----------------------------------------------------------------------------

        new_population = [(decode_chromosome((chx, chy), a, c, hx, hy)) for chx, chy in offspring_population]
        print('new_population', new_population)

        best_individual, best_fitness = get_best_individual(chromosome_values, average_fitness, a, c, hx, hy)
        print(
            f"Покоління {generation + 1}: Найкращий індивідуум - x={best_individual[0]:.2f}, y={best_individual[1]:.2f}, Fitness={best_fitness:.2f}")

        generations.append(generation + 1)
        best_fitness_values.append(best_fitness)
        best_individuals.append(best_individual)

        if best_fitness > global_best_fitness:
            global_best_fitness = best_fitness
            global_best_individual = best_individual

        population = new_population

    print(
        f"\nОтримана найкраща точка максимуму: x={global_best_individual[0]:.2f}, y={global_best_individual[1]:.2f}, Fitness={global_best_fitness:.2f}")

    # -------------------Графік успішності поколінь--------------------------
    plt.figure(figsize=(32, 20))
    plt.plot(generations, best_fitness_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Покоління')
    plt.ylabel('Найкраща пристосованість')
    plt.title('Графік успішності поколінь')
    plt.grid(True)
    plt.show()

    # -----------------Графік середньої пристосованості----------------------
    plt.figure(figsize=(30, 14))
    plt.plot(generations, average_fitness_values, marker='x', linestyle='--', color='r',
             label="Середня пристосованість")
    plt.xlabel('Покоління')
    plt.ylabel('Пристосованість')
    plt.title('Графік середньої пристосованості')
    plt.grid(True)
    plt.legend()
    plt.show()

    # --------------------------Графік щільності потрапляння-----------------

    x_values = [round(ind[0], 2) for ind in best_individuals]
    y_values = [round(ind[1], 2) for ind in best_individuals]

    points = list(zip(x_values, y_values))

    point_counts = Counter(points)

    x_unique, y_unique = zip(*point_counts.keys())

    sizes = [point_counts[point] * 100 for point in point_counts]
    colors = [point_counts[point] for point in point_counts]
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_unique, y_unique, s=sizes, c=colors, alpha=0.6, edgecolors='w', cmap='viridis')
    plt.colorbar(scatter, label='Density (Count)')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Графік щільності потрапляння')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
