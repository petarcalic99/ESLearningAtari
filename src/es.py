import numpy as np


def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y


def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

# adopted from:
# https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/optimizers.py


class Optimizer(object):
    def __init__(self, pi, epsilon=1e-08):
        self.pi = pi
        self.dim = pi.num_params
        self.epsilon = epsilon
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        theta = self.pi.mu
        ratio = np.linalg.norm(step) / (np.linalg.norm(theta) + self.epsilon)
        self.pi.mu = theta + step
        return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class BasicSGD(Optimizer):
    def __init__(self, pi, stepsize):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize

    def _compute_step(self, globalg):
        step = -self.stepsize * globalg
        return step


class SGD(Optimizer):
    def __init__(self, pi, stepsize, momentum=0.9):
        Optimizer.__init__(self, pi)
        self.v = np.zeros(self.dim, dtype=np.float32)
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, globalg):
        self.v = self.momentum * self.v + (1. - self.momentum) * globalg
        step = -self.stepsize * self.v
        return step


class Adam(Optimizer):
    def __init__(self, pi, stepsize, beta1=0.99, beta2=0.999):
        Optimizer.__init__(self, pi)
        self.stepsize = stepsize
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v = np.zeros(self.dim, dtype=np.float32)

    def _compute_step(self, globalg):
        a = self.stepsize * \
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


class CMAES:
    '''CMA-ES wrapper.'''

    def __init__(self, num_params,      # number of model parameters
                 sigma_init=0.10,       # initial standard deviation
                 popsize=255,           # population size
                 weight_decay=0.01):    # weight decay coefficient

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.popsize = popsize
        self.weight_decay = weight_decay
        self.solutions = None

        import cma
        self.es = cma.CMAEvolutionStrategy(self.num_params * [0],
                                           self.sigma_init,
                                           {'popsize': self.popsize,
                                            })

    def rms_stdev(self):
        sigma = self.es.result[6]
        return np.mean(np.sqrt(sigma*sigma))

    def ask(self):
        '''returns a list of parameters'''
        self.solutions = np.array(self.es.ask())
        return self.solutions

    def tell(self, reward_table_result):
        reward_table = -np.array(reward_table_result)
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay
        # convert minimizer to maximizer.
        self.es.tell(self.solutions, (reward_table).tolist())

    def current_param(self):
        return self.es.result[5]  # mean solution, presumably better with noise

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.es.result[0]  # best evaluated solution

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        r = self.es.result
        return (r[0], -r[1], -r[1], r[6])


class SimpleGA:
    '''Simple Genetic Algorithm.'''

    def __init__(self, num_params,      # number of model parameters
                 sigma_init=0.1,        # initial standard deviation
                 sigma_decay=0.999,     # anneal standard deviation
                 sigma_limit=0.01,      # stop annealing if less than this
                 popsize=256,           # population size
                 elite_ratio=0.1,       # percentage of the elites
                 forget_best=False,     # forget the historical best elites
                 weight_decay=0.01,     # weight decay coefficient
                 ):

        self.num_params = num_params
        self.sigma_init = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.popsize = popsize

        self.elite_ratio = elite_ratio
        self.elite_popsize = int(self.popsize * self.elite_ratio)

        self.sigma = self.sigma_init
        self.elite_params = np.zeros((self.elite_popsize, self.num_params))
        self.elite_rewards = np.zeros(self.elite_popsize)
        self.best_param = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_iteration = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay

    def rms_stdev(self):
        return self.sigma  # same sigma for all parameters.

    def ask(self):
        '''returns a list of parameters'''
        self.epsilon = np.random.randn(
            self.popsize, self.num_params) * self.sigma
        solutions = []

        def mate(a, b):
            c = np.copy(a)
            idx = np.where(np.random.rand((c.size)) > 0.5)
            c[idx] = b[idx]
            return c

        elite_range = range(self.elite_popsize)
        for i in range(self.popsize):
            idx_a = np.random.choice(elite_range)
            idx_b = np.random.choice(elite_range)
            child_params = mate(
                self.elite_params[idx_a], self.elite_params[idx_b])
            solutions.append(child_params + self.epsilon[i])

        solutions = np.array(solutions)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert(len(reward_table_result) ==
               self.popsize), "Inconsistent reward_table size reported."

        reward_table = np.array(reward_table_result)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay

        if self.forget_best or self.first_iteration:
            reward = reward_table
            solution = self.solutions
        else:
            reward = np.concatenate([reward_table, self.elite_rewards])
            solution = np.concatenate([self.solutions, self.elite_params])

        idx = np.argsort(reward)[::-1][0:self.elite_popsize]

        self.elite_rewards = reward[idx]
        self.elite_params = solution[idx]

        self.curr_best_reward = self.elite_rewards[0]

        if self.first_iteration or (self.curr_best_reward > self.best_reward):
            self.first_iteration = False
            self.best_reward = self.elite_rewards[0]
            self.best_param = np.copy(self.elite_params[0])

        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

    def current_param(self):
        return self.elite_params[0]

    def set_mu(self, mu):
        pass

    def best_param(self):
        return self.best_param

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_param, self.best_reward, self.curr_best_reward, self.sigma)


class sepCEM:  # ADDED CEM

    def __init__(self, num_params,  # number of model parameters
                 mu_init=None,
                 sigma_init=1e-3,
                 pop_size=256,
                 damp=1e-3,
                 damp_limit=1e-5,
                 parents=None,
                 elitism=False,
                 antithetic=False):
        # misc
        self.num_params = num_params
        self.name = "CES"

        # distribution parameters
        if mu_init is None:
            self.mu = np.zeros(self.num_params)
        else:
            self.mu = np.array(mu_init)
        self.sigma = sigma_init
        self.damp = damp
        self.damp_limit = damp_limit
        self.tau = 0.95
        self.cov = self.sigma * np.ones(self.num_params)

        # elite stuff
        self.elitism = elitism
        self.elite = np.sqrt(self.sigma) * np.random.rand(self.num_params)
        self.elite_score = None

        # sampling stuff
        self.popsize = pop_size
        self.antithetic = antithetic

        ##########         MODIFS         ##########
        # 2 GB of random noise as in OpenAI paper.
        self.noise_table = np.random.RandomState(
            123).randn(int(5e8)).astype('float32')
        self.use_noise_table = True

        ############################################

        if self.antithetic or self.use_noise_table:
            self.half_popsize = int(self.popsize / 2)
        if parents is None or parents <= 0:
            self.parents = self.popsize // 2
        else:
            self.parents = parents
        self.weights = np.array([np.log((self.parents + 1) / i)
                                 for i in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

    def r_noise_id(self):
        return np.random.random_integers(0, len(self.noise_table)-self.num_params)

    def ask(self):
        """
        Returns a list of candidates parameters
        """

        ##########      MODIF      ##########
        unpair = self.popsize % 2
        if self.use_noise_table:
            self.epsilon_half = np.zeros(
                (self.half_popsize + unpair, self.num_params))
            for i in range(self.half_popsize + unpair):
                r_id = self.r_noise_id()
                self.epsilon_half[i] = self.noise_table[r_id:(
                    r_id + self.num_params)]
            self.epsilon = np.concatenate(
                [self.epsilon_half, - self.epsilon_half])
            if unpair == 1:
                self.epsilon = np.delete(self.epsilon, 0, 0)
        #####################################

        else:
            self.epsilon = np.random.randn(self.popsize, self.num_params)

        inds = self.mu + self.epsilon * np.sqrt(self.cov)
        if self.elitism:
            inds[-1] = self.elite

        solutions = np.array(inds)
        self.solutions = solutions

        return solutions

    def tell(self, reward_table_result):  # (self, solutions, scores):
        """
        Updates the distribution
        """
        scores = np.array(reward_table_result)
        scores *= -1
        idx_sorted = np.argsort(scores)

        old_mu = self.mu
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit
        self.mu = self.weights @ self.solutions[idx_sorted[:self.parents]]

        z = (self.solutions[idx_sorted[:self.parents]] - old_mu)
        self.cov = 1 / self.parents * \
            self.weights @ (z * z) + self.damp * np.ones(self.num_params)

        self.elite = self.solutions[idx_sorted[0]]
        self.elite_score = scores[idx_sorted[0]]
        # print(self.cov)

    def get_distrib_params(self):
        """
        Returns the parameters of the distrubtion:
        the mean and sigma
        """
        return np.copy(self.mu), np.copy(self.cov)

    def best_param(self):
        return self.elite

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.elite, -1*self.elite_score, self.elite_score, self.sigma)


class OpenES:
    ''' Basic Version of OpenAI Evolution Strategies.'''

    def __init__(self, num_params,             # number of model parameters
                 sigma_init=0.01,               # initial standard deviation
                 sigma_decay=1,  # 0.999,            # anneal standard deviation
                 sigma_limit=0.01,             # stop annealing if less than this
                 learning_rate=1,  # 0.01,           # learning rate for standard deviation
                 learning_rate_decay=1,  # 0.9999, # annealing the learning rate
                 learning_rate_limit=0.001,  # stop annealing learning rate
                 popsize=256,                  # population size
                 antithetic=False,             # whether to use antithetic sampling
                 weight_decay=0.01,            # weight decay coefficient
                 rank_fitness=True,            # use rank rather than fitness numbers
                 forget_best=True):            # forget historical best

        self.num_params = num_params
        self.sigma_decay = sigma_decay
        self.sigma = sigma_init
        self.sigma_init = sigma_init
        self.sigma_limit = sigma_limit
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_limit = learning_rate_limit
        self.popsize = popsize
        self.antithetic = antithetic
        self.name = "OES"

        ##########         MODIFS         ##########
        # 2 GB of random noise as in OpenAI paper.
        self.noise_table = np.random.RandomState(
            123).randn(int(5e8)).astype('float32')
        self.use_noise_table = True

        ############################################

        if self.antithetic or self.use_noise_table:
            self.half_popsize = int(self.popsize / 2)

        self.reward = np.zeros(self.popsize)
        self.mu = np.zeros(self.num_params)
        self.curr_best_mu = np.zeros(self.num_params)
        self.best_mu = np.zeros(self.num_params)
        self.best_reward = 0
        self.first_interation = True
        self.forget_best = forget_best
        self.weight_decay = weight_decay
        self.rank_fitness = rank_fitness

        if self.rank_fitness:
            self.forget_best = True  # always forget the best one if we rank
        # choose optimizer
        self.optimizer = Adam(self, learning_rate)

    def r_noise_id(self):
        return np.random.random_integers(0, len(self.noise_table)-self.num_params)

    def rms_stdev(self):
        sigma = self.sigma
        return np.mean(np.sqrt(sigma*sigma))

    def ask(self):
        '''returns a list of parameters'''

        ##########      MODIF      ##########
        unpair = self.popsize % 2
        if self.use_noise_table:
            self.epsilon_half = np.zeros(
                (self.half_popsize + unpair, self.num_params))
            for i in range(self.half_popsize + unpair):
                r_id = self.r_noise_id()
                self.epsilon_half[i] = self.noise_table[r_id:(
                    r_id + self.num_params)]
            self.epsilon = np.concatenate(
                [self.epsilon_half, - self.epsilon_half])
            if unpair == 1:
                self.epsilon = np.delete(self.epsilon, 0, 0)
        #####################################

        # antithetic sampling
        elif self.antithetic:
            self.epsilon_half = np.random.randn(
                self.half_popsize, self.num_params)
            self.epsilon = np.concatenate(
                [self.epsilon_half, - self.epsilon_half])
        else:
            self.epsilon = np.random.randn(self.popsize, self.num_params)

        self.solutions = self.mu.reshape(
            1, self.num_params) + self.epsilon * self.sigma

        return self.solutions

    def tell(self, reward_table_result):
        # input must be a numpy float array
        assert(len(reward_table_result) ==
               self.popsize), "Inconsistent reward_table size reported."

        reward = np.array(reward_table_result)

        if self.rank_fitness:
            reward = compute_centered_ranks(reward)

        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward += l2_decay

        idx = np.argsort(reward)[::-1]

        best_reward = reward[idx[0]]
        best_mu = self.solutions[idx[0]]

        self.curr_best_reward = best_reward
        self.curr_best_mu = best_mu

        if self.first_interation:
            self.first_interation = False
            self.best_reward = self.curr_best_reward
            self.best_mu = best_mu
        else:
            if self.forget_best or (self.curr_best_reward > self.best_reward):
                self.best_mu = best_mu
                self.best_reward = self.curr_best_reward

        # main bit:
        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = 1./(self.popsize*self.sigma) * \
            np.dot(self.epsilon.T, normalized_reward)

        #self.mu += self.learning_rate * change_mu

        self.optimizer.stepsize = self.learning_rate
        update_ratio = self.optimizer.update(-change_mu)

        # adjust sigma according to the adaptive sigma calculation
        if (self.sigma > self.sigma_limit):
            self.sigma *= self.sigma_decay

        if (self.learning_rate > self.learning_rate_limit):
            self.learning_rate *= self.learning_rate_decay

    def current_param(self):
        return self.curr_best_mu

    def set_mu(self, mu):
        self.mu = np.array(mu)

    def best_param(self):
        return self.best_mu

    def result(self):  # return best params so far, along with historically best reward, curr reward, sigma
        return (self.best_mu, self.best_reward, self.curr_best_reward, self.sigma)
