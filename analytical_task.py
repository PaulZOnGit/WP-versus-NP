import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group



"""
Exemplary code that demonstrates the learning of WP, NP, WP0 and HP for the
analytically tractable tasks of the article

    "Weight versus Node Perturbation Learning in Temporally Extended Tasks:
     Weight Perturbation Often Performs Similarly or Better" PRX (2023)

* Choose the learning rule by setting trainings_mode
* For repeating input tasks set Neff_trial = Neff_task
* For tasks comprised of multiple subtasks set Neff_task > Neff_trial
* Set E_opt > 0 to introduce irreproducible target components 
  (requires Neff_task < T)
* Set gamma_decay < 1 to add multiplicative weight decay
* Set rotate_inputs = True to perform a random input rotation. 
  This only affects WP0

All parameters needed to redo the plots are given in the article.
"""





""" ##########
    Parameters
    ########## """


# Task parameters
n_trials = 10_000    # Number of trials

M_out = 10  # Number of outputs
N_in = 100  # Number of inputs
T = 100     # Number of time bins, trial duration
Neff_task = 50  # Effective task dimension
Neff_trial = 10 # Effective trial dimension (set Neff_trial=Neff_task for
                # tasks with repeated inputs)

full_input_dim = (Neff_task==T) # Then there can be no unrealizable components
multiple_subtasks = (Neff_task>Neff_trial)  # To differentiate

alpha_squared = N_in/Neff_trial
    # Input strength per latent input, 
    # division by Neff_trial keeps total input strength const
E_opt = 2 if not full_input_dim else 0
    # Limiting error that remains even with optimal
    # weights, due to unrealizable target components
    # E_opt must be 0 if Neff_task==T

w_init_val = 0  # All weights are initialized to zero
w_opt_val = 0.1 # Target weights that yield (realizable part
                # of) the target output

rotate_inputs = False    # If True, inputs are randomly rotated, so that 
                # input components are represented by multiple neurons 
                # (and all input neurons carry nonzero inputs, so that 
                # WP0 behaves just like WP)


# Learning rule parameters
trainings_mode = "WP" # Can be "WP", "NP", "WP0" or "HP"

# Setting optimal learning rate
if trainings_mode=="WP":
    eta_opt_wp = 1/((M_out*min(Neff_task,N_in,T)+2) * alpha_squared)
    eta = eta_opt_wp    # Operate at optimal learning rate
elif trainings_mode=="NP":
    eta_opt_np = 1/((M_out*min(Neff_trial,N_in,T)+2) * alpha_squared)
    eta = eta_opt_np    # NP and WP0 have same high eta_opt
elif trainings_mode=="WP0":
    if rotate_inputs:   # WP0 = WP as all inputs are nonzero
        eta_opt_wp0 = 1/((M_out*min(Neff_task,N_in,T)+2) * alpha_squared)
    else:               # For sparse (diagonal) inputs, WP0 learns as fast as NP
        eta_opt_wp0 = 1/((M_out*min(Neff_trial,N_in,T)+2) * alpha_squared)
    eta = eta_opt_wp0   # Operate at optimal learning rate
elif trainings_mode=="HP":
    eta_opt_hp = 1/((M_out*min(Neff_trial,N_in,T)+2) * alpha_squared**2 * T)
    eta = eta_opt_hp    # Updates scale with alpha^4 instead of alpha^2 (which
                        # is related to HP having biased updates for
                        # inhomogeneous input strengths), thus the additional
                        # division by alpha_squared. They are additionally 
                        # scaled down by T (to offset the additional eligibility
                        # trace as compared to WP)

sigma_eff = 1e-10 #0.04    # Effective perturbation strength; 
                    # Experiments used 0.04 as finite or 1e-10 as negligible
if trainings_mode in ["NP"]:
    sigma = sigma_eff   # Amplitude of node perturbations
elif trainings_mode in ["WP", "WP0", "HP"]:
    sigma = sigma_eff / np.sqrt(alpha_squared * Neff_trial)
                        # Amplitude of weight perturbations

gamma_decay = 1 # Weight decay factor; 1 means no decay, 1-1e-3 was used in
                # the weight diffusion experiment


""" ########################
    Generate input functions
    ######################## """


def orthonormalize_gram_schmidt(inputs, min_norm=1e-15):
    """ Takes (NxT) dimensional inputs and orthonormalizes them.
        Inputs (T-dim.) whose norm of their linearly independent
        part is smaller than min_norm are ignored. This is motivated
        by the fact that a readout can/should not magnify arbitrarily
        small differences in inputs """
    basis_inputs = np.copy(inputs)
    for i in range(inputs.shape[0]):
        # Orthogonalize
        for j in range(i):
            basis_inputs[i,:] -= np.dot(basis_inputs[i,:], basis_inputs[j,:])\
                                    * basis_inputs[j,:]
        # Normalize and ignore dependent inputs
        norm = np.linalg.norm(basis_inputs[i,:])
        basis_inputs[i,:] *= 1/norm if norm>min_norm else 0

    return basis_inputs


def generate_basis_functions(T):
    """ Generate T smooth orthonormal basis functions
        from sine waves. Summed squares along time-dimension
        normalized to 1, orthogonalized via Gram-Schmidt. """
    t_grid = np.linspace(0, (1-1/T)*2*np.pi, T)
    basis_funcs = np.zeros(shape=(T, T))
    for i in range(T):
            order = (i+1)//2
            type_ = (i+1)%2

            func = np.sin if type_==0 else np.cos
            basis_funcs[i,:] = func(order * t_grid)
            basis_funcs[i,:] /= np.linalg.norm(basis_funcs[i,:])

    # To be sure, orthonormalize again
    basis_funcs = orthonormalize_gram_schmidt(basis_funcs, min_norm=1e-15)

    return basis_funcs


def create_inputs_from_basis_functions(input_strengths, basis_funcs):
    """ basis_funcs: T x T-dimensional array of orthonormal basis function 
        (last axis = time dimension). 
        input_strengths: N-dimensional array switching inputs on and off
        (determining their strengths).  """
    N = len(input_strengths)
    T = basis_funcs.shape[1]
    inputs = np.zeros(shape=(N, T))
    coeffs = np.sqrt(T * input_strengths)
    if N<T:
        inputs = np.dot(np.diag(coeffs), basis_funcs[:N, :])
    elif N==T:
        inputs = np.dot(np.diag(coeffs), basis_funcs)
    elif N>T:
        inputs[:T, :] = np.dot(np.diag(coeffs[:T]), basis_funcs)

    return inputs


""" ######################
    Define task and target
    ###################### """


# Generate T basis functions of length T
basis_funcs = generate_basis_functions(T)

# Target weights
w_tar = np.zeros(shape=(M_out, N_in))   
w_tar[:, :Neff_task] = w_opt_val    # Each target output (its realizable part)
                                    # is obtained by reading out the first 
                                    # Neff_task inputs with weights w_opt_val

# Apply random input rotation
if rotate_inputs:
    O_random = special_ortho_group.rvs(N_in)
    w_tar = np.dot(w_tar, O_random.T)

# Irreporducible target component
if Neff_task < T:    # Else every T-dimensional output is realizable
    alpha_d_squared = 2*E_opt/M_out # Strength of missing component
    # The (Neff_task+1)th input component is zero; add it as unrealizable target
    irrep_tar_comps = np.sqrt(alpha_d_squared * T) * basis_funcs[Neff_task+1, :]
else:
    irrep_tar_comps = np.zeros(shape=(T))


""" ########################
    Set up linear perceptron
    ######################## """


class linear_perceptron:
    def __init__(self,
        N,  # Number of input traces
        M,  # Number of output traces
        T,  # Number of timesteps, length of signals
        eta,    # Learning rate
        sigma,  # Perturbation strength
        gamma_decay  # Weight decay const
    ):
        self.N = N
        self.M = M
        self.T = T
        self.eta = eta
        self.sigma = sigma
        self.gamma_decay = gamma_decay

        # Inputs, to be set later
        self.input_traces = np.zeros(shape=(N, T))

        # Perturbations
        self.weight_pert = np.zeros(shape=(M, N))
        self.node_pert = np.zeros(shape=(M, T))
        self.output_perturbations = np.zeros(shape=(M, T))  # Used in HP learning

        # Variables
        self.w_out = np.zeros(shape=(M, N))     # Perceptron weights
        self.elig = np.zeros(shape=(M, N))      # Eligibility trace
        self.last_update = np.zeros(shape=(M, N))   # To store and analyse


    def evolve(self):
        self.output_unperturbed = np.dot(self.w_out, self.input_traces)
        self.output_perturbations \
            = np.dot(self.weight_pert, self.input_traces) + self.node_pert
        self.output = self.output_unperturbed + self.output_perturbations
        self.elig = np.dot(self.output_perturbations, self.input_traces.T)

    """ Calculating and applying perturbations """
    def apply_random_weight_perturbation(self):
        """ Random gaussian noise with std sigma and shape (M, N) """
        self.weight_pert = self.sigma*np.random.normal(0, 1,
                                            size=(self.M, self.N))

    def apply_random_node_perturbation(self):
        """ Random gaussian noise with std sigma and shape (M, T) """
        self.node_pert = self.sigma*np.random.normal(0, 1, 
                                            size=(self.M, self.T))


    """ Updates """
    def update_weights_WP(self, dR):
        self.last_update = self.eta*dR*self.weight_pert / self.sigma**2
        self.w_out += self.last_update
        self.w_out *= self.gamma_decay   # Multiplicative weight decay

    def update_weights_NP(self, dR):
        self.last_update = self.eta*dR*self.elig / self.sigma**2
        self.w_out += self.last_update
        self.w_out *= self.gamma_decay   # Multiplicative weight decay

    def update_weights_WP0(self, dR):
        """ Like WP, but doesn't update weights that receive zero input.
            Could be extended to ignore f'=0 of potential output nonlinearity 
            as well """
        self.last_update = self.eta*dR*self.weight_pert / self.sigma**2
        nonzero_inputs = np.where(np.any(self.input_traces!=0, axis=1), 1, 0)
        self.last_update *= nonzero_inputs[np.newaxis, :]
        self.w_out += self.last_update
        self.w_out *= self.gamma_decay   # Multiplicative weight decay


    """ Setting the state """
    def reset_perturbations(self):
        self.weight_pert *= 0
        self.node_pert *= 0
        self.output_perturbations *= 0

    def reset_elig(self):
        self.elig *= 0
        self.elig_wnp *= 0


# Create network instance
net = linear_perceptron(
    N = N_in,  # Number of input traces
    M = M_out,  # Number of output traces
    T = T,  # Number of timesteps, length of signals
    eta = eta,    # Learning rate
    sigma = sigma,   # Perturbation strength
    gamma_decay = gamma_decay
)




""" #########
    Main Loop
    ######### """


def get_reward(output, target, T):
    # Quadratic reward function (reward = negative error)
    return - 0.5 * np.sum( (target-output)**2 ) / T

# Data arrays
R0_data = np.zeros(shape=(n_trials))
R_pert_data = np.zeros(shape=(n_trials))
w_relev_stats_data = np.zeros(shape=(n_trials, 2))
w_irrel_stats_data = np.zeros(shape=(n_trials, 2))
for i_trial in range(n_trials):
    """ Generate inputs and targets """
    # Randomly chose Neff_trial of the first Neff_task inputs to be active
    active_input_ids = np.random.choice(np.arange(Neff_task), replace=False,
                                        size=int(Neff_trial))
    input_strengths = np.zeros(shape=(N_in))
    input_strengths[active_input_ids] = alpha_squared
    input_traces = create_inputs_from_basis_functions(input_strengths, basis_funcs)
    if rotate_inputs:
        input_traces = np.dot(O_random, input_traces)
    net.input_traces = input_traces # Apply inputs
    
    # Generate targets
    realizable_targets = np.dot(w_tar, input_traces)
    target_traces = realizable_targets + irrep_tar_comps

    """ Unperturbed trial """
    net.reset_perturbations()
    net.evolve()
    R0 = get_reward(net.output, target_traces, T)

    """ Perturbed trial """
    # Apply perturbations
    if trainings_mode in ["WP", "WP0", "HP"]:
        net.apply_random_weight_perturbation()
    elif trainings_mode=="NP":
        net.apply_random_node_perturbation()
    # Evaluate 
    net.evolve()
    R_pert = get_reward(net.output, target_traces, T)
    dR = R_pert-R0

    """ Update """
    if trainings_mode=="WP":
        net.update_weights_WP(dR)
    elif trainings_mode in ["NP", "HP"]:
        net.update_weights_NP(dR)
    elif trainings_mode=="WP0":
        net.update_weights_WP0(dR)

    # Store data
    R0_data[i_trial] = R0
    R_pert_data[i_trial] = R_pert
    # Ensemble statistics of relevant and irrelevant weights
    if rotate_inputs:
        w_relev = np.dot(net.w_out, O_random)[:,:Neff_task]
    else:
        w_relev = net.w_out[:,:Neff_task]
    w_relev_stats_data[i_trial] = [np.mean(w_relev),
                                    np.std(w_relev)]
    if not full_input_dim:  # Irrelevant weights only defined in this case
        if rotate_inputs:
            w_irrel = np.dot(net.w_out, O_random)[:,Neff_task:]
        else:
            w_irrel = net.w_out[:,Neff_task:]
        w_irrel_stats_data[i_trial] = [np.mean(w_irrel),
                                        np.std(w_irrel)]


""" ###################################
    Predicted expected reward evolution
    ################################### """


def predict_reward_multiple_subtasks(Neff_task, Neff_trial, M_out, T,
                                            E_opt, R_init, t):
    # Analytic prediction of expected reward when learning at
    # eta = eta_opt, starting from R_init. Assumes negligibly
    # small perturbations (sigma appr. 0)
    R_opt = -E_opt

    a_wp = 1 - (Neff_trial/Neff_task)*1/(M_out*Neff_task+2)
    a_np = 1 - (Neff_trial/Neff_task)*1/(M_out*Neff_trial+2)

    R_pred_WP = R_opt + (R_init-R_opt)*a_wp**t
    R_pred_NP = 2*R_opt + (R_init-2*R_opt)*a_np**t

    return R_pred_WP, R_pred_NP


def predict_reward_single_subtask(eta, sigma, N_eff, T, M_out,
                                     alpha_squared, E_opt, R_init, t):
    # Analytic prediction of expected reward when learning a task consisting
    # of a single subtask, that is N_eff = Neff_task = Neff_trial
    R_opt = -E_opt

    a_par = 1 - 2*eta*alpha_squared + eta**2 * alpha_squared**2 * (M_out*N_eff+2)
    b_WP = - 0.125*sigma**2 * eta**2 * alpha_squared**3 \
            * (M_out**3 * N_eff**3 + 6*M_out**2 * N_eff**2 + 8*M_out*N_eff)
    b_NP = - 0.125*sigma**2 * eta**2 * alpha_squared**2 \
                * (M_out**3 * N_eff*T + 6*M_out**2 * N_eff + 8*M_out*N_eff/T) \
            + eta**2 * alpha_squared**2 * M_out * N_eff * R_opt

    R_pred_WP = R_opt + (R_init-R_opt - b_WP/(1-a_par))*a_par**t + b_WP/(1-a_par)
    R_pred_NP = R_opt + (R_init-R_opt - b_NP/(1-a_par))*a_par**t + b_NP/(1-a_par)

    return R_pred_WP, R_pred_NP



""" ########
    Plotting
    ######## """


# Theoretic predictions
t_grid = np.arange(n_trials)
R_init = R0_data[0]
eta_opt_wp = 1/((M_out*min(Neff_task,N_in,T)+2) * alpha_squared)
eta_opt_np = 1/((M_out*min(Neff_trial,N_in,T)+2) * alpha_squared)
if multiple_subtasks:   # Multiple subtasks
    R_pred_wp_data, R_pred_np_data \
        = predict_reward_multiple_subtasks(Neff_task, Neff_trial, M_out, T,
                                            E_opt, R_init, t_grid)
else:   # Repeated input, single subtask
    R_pred_wp_data, _ \
        = predict_reward_single_subtask(eta_opt_wp, sigma, Neff_task, T, M_out, 
                                        alpha_squared, E_opt, R_init, t_grid)
    _, R_pred_np_data \
        = predict_reward_single_subtask(eta_opt_np, sigma, Neff_task, T, M_out, 
                                        alpha_squared, E_opt, R_init, t_grid)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 10

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

colors = {
    "WP" : "blue",
    "NP" : "orange",
    "WP0": "black",
    "HP" : "gray"
}

title_snippet = trainings_mode + " learning. "
if multiple_subtasks and sigma_eff<1e-6:
    title_snippet += "Multiple subtasks. "
elif multiple_subtasks and sigma_eff>=1e-6:
    title_snippet += r'Multiple subtasks, predictions neglect finite $\sigma$! '
else:
    title_snippet += "Single task. "
#
if rotate_inputs:
    title_snippet += "Mixed inputs. "
else:
    title_snippet += "Diagonal inputs. "
#
if E_opt==0:
    title_snippet += "Fully realizable targets. "
    if Neff_task==T:
        title_snippet += "No irrelevant weights. "
else:
    title_snippet += "Targets partially unrealizable. "
#
if gamma_decay<1:
    title_snippet += r'Weight decay $\gamma=$'+"{:.4f} not considered in predictions. ".format(gamma_decay)

fig, axes = plt.subplots(figsize=(10, 3.5), ncols=2)
fig.subplots_adjust(left=0.07, bottom=0.14, right=0.99, top=0.8, wspace=0.185, hspace=0)
fig.suptitle(title_snippet + "\n" \
            +r'$M$='+"{:d}".format(M_out)+r', $T$='+"{:d}".format(T) \
            +r', $N$='+"{:d}".format(N_in) \
            +r', $N_\mathrm{eff}^\mathrm{task}$='+"{:d}".format(Neff_task) \
            +r', $N_\mathrm{eff}^\mathrm{trial}$='+"{:d}".format(Neff_trial) \
            +r', $\sigma_\mathrm{eff}$='+"{:.2e}".format(sigma_eff) \
            +r', $E_\mathrm{opt}$='+"{:.1f}\n".format(E_opt))

axes[0].set_title("Error evolution")
axes[0].plot(-R0_data, c=colors[trainings_mode], label=trainings_mode, alpha=0.5)
axes[0].plot(-R_pred_wp_data, c=colors["WP"], linestyle="dashed", label="WP predicted")
axes[0].plot(-R_pred_np_data, c=colors["NP"], linestyle="dashed", label="NP predicted")
axes[0].legend()
axes[0].set_xlabel(r'Trials')
axes[0].set_ylabel(r'Error')

axes[1].set_title("Evolution of relevant and irrelevant weights")
axes[1].plot(w_relev_stats_data[:,0], c="black")
axes[1].fill_between(t_grid,
                    w_relev_stats_data[:,0]-w_relev_stats_data[:,1],
                    w_relev_stats_data[:,0]+w_relev_stats_data[:,1],
                    color="black", alpha=0.2)
axes[1].plot(w_irrel_stats_data[:,0], c="red")
axes[1].fill_between(t_grid,
                    w_irrel_stats_data[:,0]-w_irrel_stats_data[:,1],
                    w_irrel_stats_data[:,0]+w_irrel_stats_data[:,1],
                    color="red", alpha=0.2)
axes[1].set_xlabel(r'Trials')
axes[1].set_ylabel(r'Weight')

plt.show()