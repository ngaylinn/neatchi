import numpy as np
import taichi as ti

from population import Population

ti.init()

CHANNELS = 3
IMG_SIZE = 256
NUM_CPPNS_SQRT = 3
NUM_CPPNS = NUM_CPPNS_SQRT ** 2
WORLD_SHAPE = (IMG_SIZE * NUM_CPPNS_SQRT,) * 2

image_field = ti.Vector.field(
    CHANNELS, shape=(NUM_CPPNS, IMG_SIZE, IMG_SIZE), dtype=ti.f32)


# def evolve():
#     population = neat.initialize_population(NUM_CPPNS)
#     for _ in range(NUM_GENERATIONS):
#         simulate(population)
#         population.scores.from_numpy(score(population))
#         population = neat.propagate(population)
#     return population


def main():
    gui = ti.GUI('CPPN Demo', WORLD_SHAPE, show_gui=True)

    population = Population(count=NUM_CPPNS)
    for _ in range(5):
        population.mutate()
    for c in range(NUM_CPPNS):
        print(f'CPPN #{c}:')
        population.print(c)
        print()
    population.render(image_field)
    images = image_field.to_numpy()

    # Normalize each image separately.
    for c in range(NUM_CPPNS):
        min_val = images[c].min()
        max_val = images[c].max()
        val_range = max_val - min_val
        images[c] = (images[c] - min_val) / val_range

    composite = np.vstack(
        (np.hstack(images[0*NUM_CPPNS_SQRT:1*NUM_CPPNS_SQRT]),
         np.hstack(images[1*NUM_CPPNS_SQRT:2*NUM_CPPNS_SQRT]),
         np.hstack(images[2*NUM_CPPNS_SQRT:3*NUM_CPPNS_SQRT])))

    while gui.running:
        gui.set_image(composite)
        gui.show()


if __name__ == '__main__':
    main()
