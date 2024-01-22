import pickle
import neat


genomes_paths = [
    "result\\in_progressV2_400cont\\best_genome200.pkl",
    "result\\in_progressV2_400cont\\best_genome160.pkl",
    "result\\in_progressV2_400cont\\best_genome120.pkl",
    "result\\in_progressV2_400cont\\best_genome80.pkl",
]

raw_genomes = []

# Set configuration file
config_path = "./config"
config = neat.config.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    config_path,
)

for g_path in genomes_paths:
    with open(g_path, "rb") as f:
        # fixed_genome = pickle.load(f)
        # fixed_policy = neat.nn.FeedForwardNetwork.create(fixed_genome, config)
        raw_genomes.append(pickle.load(f))



def calculate_distances(genomes, print_d : bool):
    distances = []

    for g in genomes:
        # print(g.fitness, g.key, g.distance(genomes[0], config.genome_config))
        d = []
        for g2 in genomes:
            d.append(g.distance(g2, config.genome_config))
        distances.append(d)

    if print_d:

        top_str = "Key_(fitness)" + "_|__"
        # top_str += " "*12 + "|  "
        for g in genomes:
            top_str += str(g.key) + "_"*(8-len(str(g.key)))
        print(top_str)

        for i in range(len(distances)):

            id_str = str(genomes[i].key)
            id_str = id_str + " "*(11 - len(id_str) - len(f"{genomes[i].fitness:0.0f}"))
            id_str = id_str + "(" + f"{genomes[i].fitness:0.0f}" + ")"
            print(id_str, end=" |  ")

            for d in distances[i]:
                print(f"{d:0.2f}", end="    ")
            print()

    return distances

def choose_genomes(genomes):
    choosen_genomes = []

    # filter genomes with low distance
    threshold = 0.5 # Could perhaps need a different value?????????????????????????????????????????????????????????????????????
    seen = []
    for i in range(len(genomes)):

        if genomes[i] in seen:
            continue
        seen.append(genomes[i])

        similar = []
        for n in range(i+1, len(genomes)):
            # Search for similar genomes
            if genomes[i].distance(genomes[n], config.genome_config) < threshold:
                similar.append(genomes[n])
        
        if len(similar) == 0:
            choosen_genomes.append(genomes[i])
        else:
            # add genomes to seen and add genome with highest fitness to choosen_genomes
            seen.extend(similar)
            similar.append(genomes[i])
            choosen_genomes.append(
                max(similar, key=lambda g: g.fitness)
            )
    
    return choosen_genomes




calculate_distances(raw_genomes, True)


print("\nAfter:")
calculate_distances(choose_genomes(raw_genomes), True)