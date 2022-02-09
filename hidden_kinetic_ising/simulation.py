def simulate_full(ising, visible_idx, T, burn_in=0):
    """
    Simulate the full Kinetic Ising model to produce data

    :param T: number of steps to simulate
    :param burn_in:
    :return:
    """
    full_s = []
    s = []
    for t in range(0, T + burn_in):
        ising.ParallelUpdate()
        full_s.append(ising.s)
        if t >= burn_in:
            s.append(ising.s[visible_idx])
    # print('Spins', s)
    return full_s, s
