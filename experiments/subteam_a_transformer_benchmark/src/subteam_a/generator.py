"""Recreate Subteam C’s synthetic transformer generator and preserve its random draw order."""
import random

HEADERS = ["x", "y", "install_date", "model_id", "altitude", "measured_humid", "measured_salt",
           "measured_overgrowth", "life", "random_placebo", "measured_gas_a", "measured_gas_b",
           "measured_gas_c", "total_gas", "leakage"]
# Only these columns are noisy, because of how they are measured. The rest are exact by design.
NOISY = {"measured_humid", "measured_salt", "measured_overgrowth", "random_placebo",
         "measured_gas_a", "measured_gas_b", "measured_gas_c"}

def draw_model(day_count, rng):
    """Sample an equipment model for the supplied installation day and random generator."""
    if day_count < 200:
        return 0
    if day_count < 400:
        return rng.randint(0, 1)
    if day_count < 700:
        return rng.randint(0, 2)
    if day_count < 1000:
        return rng.randint(1, 3)
    if day_count < 1200:
        return rng.randint(2, 3)
    return 3


def draw_gases(model, day_count, leakage, rng):
    """
    Return clean gas A, B and C from model, installation day, leakage and manufacturing
    variability.
    """
    hidden_var_a = rng.uniform(0.975, 1.025)
    hidden_var_b = rng.uniform(0.975, 1.025)
    hidden_var_c = rng.uniform(0.975, 1.025)
    start_a, start_b, start_c = {
        0: (10000, 10000, 10000),
        1: (12000, 9000, 9000),
        2: (9000, 12000, 9000),
        3: (9000, 9000, 12000),
    }[model]
    gas_a = max(hidden_var_a * start_a - day_count * leakage, 0)
    gas_b = max(hidden_var_b * start_b - 0.0005 * day_count * day_count * leakage, 0)
    gas_c = max(hidden_var_c * start_c - day_count * leakage - 0.00025 * day_count * day_count * leakage, 0)
    return gas_a, gas_b, gas_c


def remaining_life(model, day_count, humid, salt, overgrowth, total_gas):
    """Return the simulated life value from the model-specific exposure and total gas penalties."""
    gas_penalty = 25000 * (total_gas / 300000)
    if model == 0:
        return 100000 - day_count * humid - day_count * salt - day_count * overgrowth - gas_penalty
    if model == 1:
        return 100000 - day_count * humid - day_count * overgrowth - gas_penalty
    if model == 2:
        return 100000 - day_count * salt - day_count * overgrowth - gas_penalty
    return 100000 - day_count * overgrowth - gas_penalty


def generate(seed):
    """Return clean transformer rows, measurement-noise draws and mask draws for one seed."""
    rng = random.Random(seed)
    truth = []
    day_count = 0
    day_num_installs = rng.randint(5, 15)
    for i in range(100):
        for j in range(100):
            if day_count % 7 == 0:
                day_count += 2
            model = draw_model(day_count, rng)
            altitude = i + j + 100
            humid = rng.random() * (altitude / 300)
            salt = rng.random() * ((j + 200) / 300)
            overgrowth = max(2 * humid - salt, 0)
            leakage = rng.uniform(0.8, 1.2)
            gas_a, gas_b, gas_c = draw_gases(model, day_count, leakage, rng)
            total_gas = gas_a + gas_b + gas_c
            life = remaining_life(model, day_count, humid, salt, overgrowth, total_gas)
            random_var = rng.random()
            truth.append([i, j, day_count, model, altitude, humid, salt, overgrowth, life, random_var,
                          gas_a, gas_b, gas_c, total_gas, leakage])
            day_num_installs -= 1
            if day_num_installs == 0:
                day_count += 1
                day_num_installs = rng.randint(5, 15)

    # One draw per cell, shared by all scenarios.
    noise_draws = [[rng.uniform(-1, 1) for _ in HEADERS] for _ in truth]
    mask_draws = [[rng.random() for _ in HEADERS] for _ in truth]
    return truth, noise_draws, mask_draws
