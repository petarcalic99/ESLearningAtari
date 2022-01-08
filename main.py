import numpy as np
from argparse import ArgumentParser
from src.es import OpenES, sepCEM
from src.policy import Policy
from src.logger import Logger
import gym
import json
import time


def run_solver(solver, run_duration):
    """
        Run the given solver, print and save logs
    """
    history = []
    start_time = time.time()
    logger = Logger("logs/"+solver.name+"-"+str(start_time))
    timeout = start_time + run_duration * 3600
    best_score = 0
    iteration = 1
    global policy

    while time.time() < timeout:
        solutions = solver.ask()
        fitness_list = np.zeros(solver.popsize)
        for i in range(solver.popsize):
            fitness_list[i] = fit_func(solutions[i])
        solver.tell(fitness_list)
        # first element is the best solution, second element is the best fitness
        result = solver.result()
        history.append(result[1])
        duration = time.time() - start_time

        logger.log(str(iteration)+"\t"+str(duration)+"\t"+str(np.mean(fitness_list))+"\t" +
                   str(np.min(fitness_list))+"\t"+str(np.max(fitness_list))+"\t"+str(result[1])+"\n")

        if iteration % 20 == 0 or result[1] > best_score:
            print("fitness at iteration", iteration, result[1])
            print("time from start "+str(duration))
            print("mean fit : "+str(np.mean(fitness_list)))
            print("min fit : "+str(np.min(fitness_list)))
            print("max fit : "+str(np.max(fitness_list)))
            # save solution
            logger.save_parameters(np.asarray(result[0]), iteration)
            # save corresponding virtual batch
            logger.save_vb(policy.vb)
            best_score = result[1]

        iteration += 1
    print("local optimum discovered by solver:\n", result[0])
    print("fitness score at this local optimum:", result[1])
    return history


def eval_actor(parameters, nb_eval=10):
    """
        Evaluate the given parameters (solution) and return the mean reward
    """
    global policy
    fit = []
    policy.set_parameters(parameters)  # set the parameters to the policy
    for i in range(nb_eval):
        rew, steps = policy.rollout(False)  # evaluate this policy parameters
        fit.append(rew)
    return np.mean(fit)  # return the mean fitness


if __name__ == "__main__":  # lots of warning at start due to imports
    parser = ArgumentParser()
    parser.add_argument('--env', default='Qbert', type=str)    # gym env
    parser.add_argument('--duration', default=1, type=int)
    parser.add_argument('--pop_size', default=16, type=int)
    # optimizer to run, choices : OpenES, CEM
    parser.add_argument('--algo', default="OpenES", type=str)
    parser.add_argument('--elitism', default=True, type=bool)  # use elitism ?
    args = parser.parse_args()
    print("################################")
    print("Launched : "+str(args.algo)+", env "+str(args.env))

    # create atari env :
    env_name = env_name = '%sNoFrameskip-v4' % args.env  # use NoFrameskip game
    env = gym.make(env_name)
    # defines fintess function (actor evaluation function)
    fit_func = eval_actor
    # create evaluation policy for parameters, based on the Nature network of CES (needs CES configuration file)
    with open("configurations/sample_configuration.json", 'r') as f:
        configuration = json.loads(f.read())
    policy = Policy(
        env, network=configuration['network'], nonlin_name=configuration['nonlin_name'])
    vb = policy.get_vb()  # init virtual batch
    nb_params = len(policy.get_parameters())

    if args.algo == "OpenES":
        # defines OpenAI's ES algorithm solver. Note that we needed to anneal the sigma parameter # pop 10
        optimizer = OpenES(nb_params,              # number of model parameters
                           sigma_init=0.6,                # initial standard deviation
                           sigma_decay=1,                 # annealing coefficient for standard deviation
                           learning_rate=0.1,             # learning rate for standard deviation
                           learning_rate_decay=1,       # annealing coefficient for learning rate
                           popsize=args.pop_size,         # population size
                           antithetic=False,              # whether to use antithetic sampling
                           weight_decay=0,                # weight decay coefficient
                           rank_fitness=False,            # use rank rather than fitness numbers
                           forget_best=not args.elitism)  # cancel elitism ?
    elif args.algo == "CEM":
        optimizer = sepCEM(nb_params,
                           mu_init=None,
                           sigma_init=1e-3,
                           pop_size=args.pop_size,
                           damp=1e-3,
                           damp_limit=1e-5,
                           parents=None,
                           elitism=args.elitism,
                           antithetic=False)
    else:
        print("Optimizer is not available. Available : OpenES, CEM.")

    history = run_solver(optimizer, args.duration)  # launching the algorithm
