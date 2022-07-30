import math
import random
import numpy
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# REGISTER FUNCTION FOR GENERATE AN RANDOM VALUE FOR EACH VARIABLE
toolbox.register("attr_uniform", random.uniform, -100, 100)

# REGISTER FUNCTION FOR CREATE GUY RANDOMLY
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_uniform, 2)

# REGISTER FUNCTION FOR CREATE INITIAL POPULATION
toolbox.register("population", tools.initRepeat, list, toolbox.individual, 10)


def target_function(x):
    for i in range(len(x)):
        if x[i] > 100 or x[i] < -100:
            return -1
    return math.sqrt(x[0] ** 2 + x[1] ** 2),


# REGISTER THE TARGET FUNCTION
toolbox.register("evaluate", target_function)

# MAKE THE CROSSING
toolbox.register("mate", tools.cxOnePoint)

# GAUSSIAN MUTATION
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=5, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


# MAIN FUNCTION FOR CONFIGURE ALGORITHM GENETIC
def main():
    random.seed(42)
    crossover_probability, mutation_probability, generation_number = 0.5, 0.3, 20
    population = toolbox.population()

    # OBTAIN THE BEST GUY FOUND
    hall_of_fame = tools.HallOfFame(1)

    # DEFINE AN OBJECT FOR GENERATE STATISTICS OF POPULATION DURING PROCESS
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    # TYPICAL DEVIATION
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    logbook = tools.Logbook()

    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability, ngen=generation_number, stats=stats, halloffame=hall_of_fame, verbose=True)

    return hall_of_fame, logbook


if __name__ == "__main__":
    best, log = main()
    print("Best fitness: %f" %best[0].fitness.values)
    print("Best individual %s" %best[0])















