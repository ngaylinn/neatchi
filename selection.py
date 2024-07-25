import taichi as ti

from .reproduction import rand_range, NONE

# Constants for testing compatibility between two individuals in a Population.
DISJOINT_COEFF = 1.0
WEIGHT_COEFF = 0.4
EPSILON = 1e-8

TOURNAMENT_SIZE = 2
COMP_THRESHOLD = 3.0
CROSSOVER_RATE = 0.6


@ti.func
def get_compatibility(pop, sp, p, m):
    """Computes compatibility between a pair of CPPNs in a population."""
    num_disjoint = 0
    weight_delta = 0.0
    num_common_weights = 0

    num_p_links = pop.num_links(sp, p)
    num_m_links = pop.num_links(sp, m)
    largest_num_links = ti.max(num_p_links, num_m_links)

    # Compare the links between parent and mate to see how similar they are.
    # Note that we ignore nodes entirely, and this is probably okay. Extra
    # nodes are only relevant if their are extra links, also. If one has fewer
    # nodes (say, because of a random deletion), the associated links will also
    # have been deleted.
    # TODO: Optimize?
    for pl in range(num_p_links):
        p_link = pop.get_link(sp, p, pl)
        for ml in range(num_m_links):
            m_link = pop.get_link(sp, m, ml)
            # If both parent and mate have the same link, then compare their
            # weights and factor that into compatibility.
            if (p_link.innov == m_link.innov):
                weight_delta += abs(p_link.weight - m_link.weight)
                num_common_weights += 1
            # Otherwise, one parent has a link the other doesn't, so that link
            # counts as disjoint. The original NEAT algorithm discriminates
            # between "disjoint" and "excess" genes, but we don't.
            else:
                num_disjoint += 1

    return float(
        DISJOINT_COEFF * num_disjoint / (EPSILON + largest_num_links) +
        WEIGHT_COEFF * weight_delta / (EPSILON + num_common_weights))


@ti.data_oriented
class Matchmaker:
    """A general class for performing tournament selection on populations."""

    def __init__(self, population_shape, history_length=1):
        self.population_shape = population_shape
        self.num_sub_pops, self.num_individuals = population_shape
        self.history_length = history_length

        # Fields to track compatibility and diversity of the population.
        self.comp_matrix = ti.field(float,
            shape=(self.num_sub_pops, self.num_individuals, self.num_individuals))
        self.comp_matrix.fill(ti.math.nan) # For the main diagonals.
        self.diversity = ti.field(float, shape=(self.num_sub_pops))

        # Keep a history of fitness scores and mate selections so that we can
        # avoid copying them to the host on every iteration.
        self.fitness = ti.field(float,
            shape=(history_length, self.num_sub_pops, self.num_individuals))
        self.matches = ti.Vector.field(n=2, dtype=int,
            shape=(history_length, self.num_sub_pops, self.num_individuals))

    def reset(self):
        # Clear history data.
        self.fitness.fill(0.0)
        self.matches.fill(NONE)

    @ti.func
    def unrestrited_tournament(self, g, sp):
        """Run a tournament to pick a parent from a given sub-population."""
        p = NONE
        p_fitness = -ti.math.inf

        # Look at TOURNAMENT_SIZE random individuals and return the most fit.
        for _ in range(TOURNAMENT_SIZE):
            c = rand_range(0, self.num_individuals)
            c_fitness = self.fitness[g, sp, c]
            if c_fitness > p_fitness:
                p = c
                p_fitness = c_fitness
        return p, p_fitness

    @ti.func
    def count_compatible(self, sp, p):
        """Count how many individuals are compatible with the given parent."""
        num_compatible_mates = 0
        for c in range(self.num_individuals):
            if self.comp_matrix[sp, p, c] < COMP_THRESHOLD:
                num_compatible_mates += 1
        return num_compatible_mates

    @ti.func
    def restricted_tournament(self, g, sp, p):
        """Run a parent to pick a mate compatible with the given parent."""
        m = NONE
        m_fitness = -ti.math.inf

        # Count how many compatible mates there are. We need to know this in
        # order to randomly pick one of them. Note that we traverse the
        # compatibility matrix again in the following loop rather than caching
        # results and taking up even more GPU memory.
        num_compatible_mates = self.count_compatible(sp, p)

        # If there are compatible mates, hold a tournament to pick one.
        if num_compatible_mates > 0:
            for _ in range(TOURNAMENT_SIZE):
                # Pick one of the compatible mates at random. Note, this is not
                # an array index, it's the number of valid mates to skip before
                # we get to our selected candidate.
                nth_candidate = rand_range(0, num_compatible_mates)
                for c in range(self.num_individuals):
                    if self.comp_matrix[sp, p, c] < COMP_THRESHOLD:
                        # If we've found the nth compatible mate, compare it to
                        # the others in the tournament and keep the best one.
                        if nth_candidate == 0:
                            c_fitness = self.fitness[g, sp, c]
                            if c_fitness > m_fitness:
                                m = c
                                m_fitness = c_fitness

                            # Once we test the nth candidate, we can move on to
                            # the next round of the tournament.
                            break

                        # If we haven't reached the nth one yet, count down and
                        # keep going.
                        nth_candidate -= 1
        return m, m_fitness

    @ti.kernel
    def update_matches(self, g: int):
        """Perform selection for all individuals in all sub-populations."""
        for sp, i in ti.ndrange(*self.population_shape):
            # Hold a tournament to randomly pick a parent for this individual.
            p, p_fitness = self.unrestrited_tournament(g, sp)

            # Randomly decide whether to attempt crossover at all or if this parent
            # will simply clone itself.
            m = NONE
            if ti.random() < CROSSOVER_RATE:
                # Hold a tournament among all the mates compatible with parent
                # to pick one.
                m, m_fitness = self.restricted_tournament(g, sp, p)

                # Like the original NEAT algorithm, always put the most fit
                # parent first, creating a slight fitness bias in crossover.
                if m_fitness > p_fitness:
                    p, m = m, p

            # Finalize selections for this individual in the next generation.
            self.matches[g, sp, i] = (p, m)

    @ti.kernel
    def analyze_compatibility(self, pop: ti.template()):
        # This function populates a compatibility matrix for each sub-population in
        # pop, with one row and one column for each individual. Since these are
        # symmetric matrices, we allocate one thread per unique value we need to
        # fill (half the matrix minus the main diagonal, which is unfilled).
        triangle_size = int(((self.num_individuals - 1) / 2) * self.num_individuals)
        for sp, tri in ti.ndrange(self.num_sub_pops, triangle_size):
            # Identify the row and column this thread should fill (ie, which
            # potential parents to compute compatibility for).
            p = tri // self.num_individuals
            m = tri % self.num_individuals

            # If this would compute a cell in the lower triangle of the matrix, use a
            # a position on the opposite side of the upper triangle instead.
            if p >= m:
                p = self.num_individuals - p - 2
                m = self.num_individuals - m - 1

            # For convenience, fill both side of the matrix. We have to allocate
            # this much space anyway, so the only thing that really matters for
            # performance is that we avoid recomputing symmetric values. Note that
            # the main diagonal of the matrix is not populated.
            compatibility = pop.get_compatibility(sp, p, m)
            self.comp_matrix[sp, p, m] = compatibility
            self.comp_matrix[sp, m, p] = compatibility

            # Sum compatibility for all pairs of individuals in this sub-population
            # (Taichi should automatically optimize this reduction).
            self.diversity[sp] += compatibility

        # Invert the total compatibility to get diversity.
        for sp in range(self.num_sub_pops):
            self.diversity[sp] = 1.0 / self.diversity[sp]
