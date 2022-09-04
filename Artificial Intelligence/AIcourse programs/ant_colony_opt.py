"""
On the implementation of a simple ant colony algorithm stack overflow
https://stackoverflow.com/questions/65309403/on-the-implementation-of-a-simple-ant-colony-algorithm
"""

import random
import matplotlib.pyplot as plt


def compute_probability(tau1, tau2):
    """
    the probability of selecting a path between
    E1 and E2 can be expressed as
    Pi = Ti/(T1+T2)
    Where Ti: the pheromone strength
    :param tau1: pheromone strength of edge 1
    :param tau2: pheromone strength of edge 2
    :return tuple with the calculated probabilities
    """
    return tau1/(tau1 + tau2), tau2/(tau1 + tau2)


# Why they didn't used 
# random.choices([1, 2], weights=[prob1, prob2])[0]
def weighted_random_choice(choices):
    # max: sum of the probabilities 
    max = sum(choices.values())
    # pick: uniform random value between 
    # 0 and max (the sum of the probabilities)
    pick = random.uniform(0, max)
    current = 0
    for key, value in choices.items():
        # current partial accumulate of probabilities 
        current += value
        if current > pick:
            return key


def select_path(prob1, prob2):
    # dict. choices associates probabilities 
    # with paths
    choices = {1: prob1, 2: prob2}
    return weighted_random_choice(choices)


def update_accumulation(link_id):
    """
    the pheromone value is updated based on the path length
    and the pheromone evaporation rate
    :param link_id: path
    """
    global tau1
    global tau2
    if link_id == 1:
        tau1 = tau1 + Q / l1
    else:
        tau2 = tau2 + Q / l2


def update_evaporation():
    """
    The evaporation rate of pheromone is calculated as:
    Ti = (1-R)*Ti
    Where R is used to regulate the pheromone evaporation.
    R belongs to the interval (0, 1]
    """
    global tau1
    global tau2
    tau1 = (1 - rho)  * tau1
    tau2 = (1 - rho) * tau2


def report_results(success):
    plt.ylim(0.0, 1.0)
    plt.xlim(0, 150)
    plt.plot(success)
    plt.show()


if __name__ == '__main__':
    N = 10  # number of ants
    l1 = 1.1  # length path 1
    l2 = 1.5  # length path 2
    rho = 0.05  # parameter that regulates the pheromone evaporation
    Q = 1  # amount of pheromone added (depends on path length)
    tau1 = 0.5  # pheromone value of edge 1
    tau2 = 0.5  # pheromone value of edge 2

    samples = 10
    epochs = 150

    success = [0 for x in range(epochs)]

    for sample in range(samples):
        for epoch in range(epochs):
            temp = 0
            for ant in range(N):
                prob1, prob2 = compute_probability(tau1, tau2)
                selected_path = select_path(prob1, prob2)
                if selected_path == 1:
                    temp += 1
                update_accumulation(selected_path)
                update_evaporation()
            ratio = ((temp + 0.0) / N)
            success[epoch] += ratio
        # reset pheromone values here to evaluate new sample
        tau1 = 0.5
        tau2 = 0.5

    success = [x / samples for x in success]

    for x in success:
        print(x)

    report_results(success)
    pass