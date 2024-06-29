import taichi as ti

from .reproduction import rand_range, NONE


# Constants for testing compatibility between two individuals in a Population.
DISJOINT_COEFF = 1.0
WEIGHT_COEFF = 0.4
COMP_THRESHOLD = 3.0
EPSILON = 1e-8

# The number of individuals to consider in selection. The minimum value is 1.0
# which indicates fully random selection. The higher this goes, the more strict
# the selection will be.
TOURNAMENT_SIZE = 2
CROSSOVER_RATE = 0.6

@ti.func
def is_compatible(pop, sp, p, m):
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


@ti.kernel
def analyze_compatibility(pop: ti.template()):
    # This function populates a compatibility matrix for each sub-population in
    # pop, with one row and one column for each individual. Since these are
    # symmetric matrices, we allocate one thread per unique value we need to
    # fill (half the matrix minus the main diagonal, which is unfilled).
    triangle_size = int(((pop.num_individuals - 1) / 2) * pop.num_individuals)
    for sp, tri in ti.ndrange(pop.num_sub_pops, triangle_size):
        # Identify the row and column this thread should fill.
        col = tri % pop.num_individuals
        row = tri // pop.num_individuals

        # If this would compute a cell in the lower triangle of the matrix, use a
        # a position on the opposite side of the upper triangle instead.
        if row >= col:
            row = pop.num_individuals - row - 2
            col = pop.num_individuals - col - 1

        # For convenience, fill both side of the matrix. We have to allocate
        # this much space anyway, so the only thing that really matters for
        # performance is that we avoid recomputing symmetric values. Note that
        # the main diagonal of the matrix is not populated.
        compatibility = is_compatible(pop, sp, row, col)
        pop.comp_matrix[sp, row, col] = compatibility
        pop.comp_matrix[sp, col, row] = compatibility

        # Sum compatibility for all pairs of individuals in this sub-population
        # (Taichi should automatically optimize this reduction).
        pop.diversity[sp] += compatibility

    # Invert the total compatibility to get diversity.
    for sp in range(pop.num_sub_pops):
        pop.diversity[sp] = 1.0 / pop.diversity[sp]


@ti.kernel
def tournament_select(pop: ti.template()):
    # For all individuals in each sub_population, pick its parent(s).
    for sp, i in ti.ndrange(*pop.population_shape):
        # Parent and mate indices for this matchup.
        p = NONE
        m = NONE

        # Uncomment to support fractional tournament sizes by picking some
        # integer value between floor() and ceil() of TOURNAMENT_SIZE with
        # probabily determined by the fractional part of TOURNAMENT_SIZE.
        tournament_size = TOURNAMENT_SIZE # (
        #    (ti.random() < TOURNAMENT_SIZE % 1.0) +
        #    int(TOURNAMENT_SIZE))

        # Consider TOURNAMENT_SIZE candidates and choose the most fit one to be
        # the parent in this match.
        p_fitness = -ti.math.inf
        for _ in range(tournament_size):
            c = rand_range(0, pop.num_individuals)
            # Compare log fitness to encourage diversity.
            c_fitness = pop.fitness[sp, c]
            if c_fitness > p_fitness:
                p = c
                p_fitness = c_fitness

        # Randomly decide whether to attempt crossover at all or if this parent
        # will simply clone itself.
        if ti.random() < CROSSOVER_RATE:
            # Count how many compatible mates there are. We need to know this
            # in order to randomly pick one of them. We traverse this list
            # twice rather than saving the results to save memory.
            num_compatible_mates = 0
            for c in range(pop.num_individuals):
                if pop.comp_matrix[sp, p, c] < COMP_THRESHOLD:
                    num_compatible_mates += 1

            # If there are compatible mates, hold a tournament to pick one.
            if num_compatible_mates > 0:
                m_fitness = -ti.math.inf
                for _ in range(tournament_size):
                    # Pick one of the compatible mates at random. Note, this is
                    # not an array index, it's the number of valid mates to
                    # skip before we get to our selected candidate.
                    nth_candidate = rand_range(0, num_compatible_mates)
                    for c in range(pop.num_individuals):
                        if pop.comp_matrix[sp, p, c] < COMP_THRESHOLD:
                            # If we've found the nth compatible mate, compare
                            # it to the ones we've seen so far.
                            if nth_candidate == 0:
                                # Compare log fitness to encourage diversity.
                                c_fitness = pop.fitness[sp, c]
                                if c_fitness > m_fitness:
                                    m = c
                                    m_fitness = c_fitness

                                # Once we test the nth candidate, we can move
                                # on to the next round of the tournament.
                                break

                            # If we haven't reached the nth one yet, count down
                            # and keep going.
                            nth_candidate -= 1

                # Like the original NEAT algorithm, always put the most fit
                # parent first, creating a slight fitness bias in crossover.
                if m_fitness > p_fitness:
                    p, m = m, p

        # Finalize parent selections for this individual in the next generation.
        pop.matches[sp, i] = (p, m)
