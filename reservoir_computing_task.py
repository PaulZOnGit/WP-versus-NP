import numpy as np
import matplotlib.pyplot as plt


"""
Exemplary code that demonstrates the learning of WP, NP, WP0 and HP for the
reservoir computing task of the article

    "Weight versus Node Perturbation Learning in Temporally Extended Tasks:
     Weight Perturbation Often Performs Similarly or Better" PRX (2023)

"""


""" ##########
    Parameters
    ########## """


# Learning rule parameters
trainings_mode = "HP"   # Can be "WP", "NP", "WP0" or "HP"
                        # Compare WP and NP at small sigma_eff (for convergence)
                        # and at large sigma_eff (i.e. 5e-3, for final error)
                        # WP0 and WP work identically (no zero inputs)
                        # HP does not perform well (explained in the main text)
sigma_eff = 5e-3    # Effective perturbation strength; 
                    # Experiments used 5e-3 as finite or 1e-10 as negligible

# Task parameters
n_trials = 30_000    # Number of trials
N_ext = 5   # Number of external inputs (sine functions)
N_res = 500 # Number of reservoir neurons
M_out = 2   # Number of outputs (butterfly coordinates)
T = 500     # Number of time bins, trial duration

# Reservoir properties
tau_res = 10    # Reservoir time constant (in time steps)



""" ###############################
    Construct N_ext external inputs
    ############################### """


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

t_grid = np.linspace(0, (1-1/T)*2*np.pi, T)
inputs = np.zeros(shape=(N_ext, T))
for i in range(N_ext):
        order = (i+1)//2
        type_ = (i+1)%2

        func = np.sin if type_==0 else np.cos
        inputs[i,:] = func(order * t_grid)
        inputs[i,:] /= np.linalg.norm(inputs[:,i])

# Orthonormalize, to be sure (and multiply by T so each input has strength 1)
inputs = np.sqrt(T) * orthonormalize_gram_schmidt(inputs, min_norm=1e-15)


""" ###############################
    Construct N_ext external inputs
    ############################### """


def rosenbaum_butterfly(t, T):
    """ Return 2-D output in the shape of a butterfly, 
        as used in Kyle and Rosenbaum. Accept scalar or array-like input """
    tau = 2*np.pi*t/(T-1)
    c = .1

    r = c * (9 - np.sin(tau) + 2*np.sin(3*tau) + 2*np.sin(5*tau) - np.sin(7*tau)
             + 3*np.cos(2*tau) - 2*np.cos(4*tau))

    return np.array([r*np.cos(tau), r*np.sin(tau)])

t_grid = np.arange(T)
targets = rosenbaum_butterfly(t_grid, T)


""" #########################
    Record reservoir activity
    ######################### """


""" Draw random connectivity """
w_in = (1/np.sqrt(N_ext))*np.random.normal(size=(N_res, N_ext))
# Recurrent connectivity: Normally distributed, largest eigenvalue = 1
w_rec = np.random.normal(size=(N_res, N_res))
w_rec_eigvals, _ = np.linalg.eig(w_rec)
spec_radius = np.max(np.real(w_rec_eigvals))
w_rec /= spec_radius

""" Evolve reservoir """
# Initialize activity
n_pre = 100 # Initialize reservoir n_pre timesteps before recording
x = np.zeros(shape=(N_res))
res_activities = np.zeros(shape=(N_res, T))

gamma = np.exp(-1/tau_res)
for i_t in range(1-n_pre,T):
    x = gamma*x + (1-gamma)*(np.dot(w_in, inputs[:,i_t])
                            +np.dot(w_rec, np.tanh(x)))
    res_activities[:,i_t] = np.tanh(x)


""" ##########################
    Analyse reservoir activity
    ########################## """


S = np.dot(res_activities, res_activities.T) / T
S_eig_vals, S_eig_vecs = np.linalg.eig(S)
sorting_indices = S_eig_vals.argsort()[::-1]
S_eig_vals = np.real(S_eig_vals[sorting_indices])
S_eig_vecs = np.real(S_eig_vecs[:,sorting_indices])

""" Estimate optimal learning rate """
# Effective dimensionality (participation ratio)
PR = (np.sum(S_eig_vals)**2) / np.sum(S_eig_vals**2)    # Measure of eff. dim.
alpha_sq_eff = np.sum(S_eig_vals) / PR  # Input strength if input was equally
                                        # distributed onto only PR latent inputs

# Calculate optimal learning rates by approximating the reservoir as having
# PR equally strong latent inputs of strength alpha_sq_eff. The resulting
# learning rates are close to those obtained by scanning eta
eta_opt = 1 / ((M_out*PR + 2) * alpha_sq_eff)
if trainings_mode=="HP":
    eta_opt = 1 / ((M_out*PR + 2) * alpha_sq_eff**2 * T)
eta = eta_opt   # Learning rate; Change manually to check around estimate


""" Adjust perturbation strength """
if trainings_mode=="NP":
    sigma = sigma_eff
elif trainings_mode in ["WP", "WP0", "HP"]:
    sigma = sigma_eff / np.sum(S_eig_vals)  # To keep total strength of output
                                            # perturbations equal for WP and NP

""" Calculate optimal solution using largest 5 input directions """
S_top5 = np.zeros(shape=(N_res, N_res))
for j in range(5):  # Only consider largest 5 input components
    S_top5 += S_eig_vals[j] * np.outer(S_eig_vecs[:,j], S_eig_vecs[:,j])

S_inv = np.linalg.pinv(S_top5, rcond=1e-10)
w_out_opt = targets @ res_activities.T @ S_inv
w_out_opt /= T
top5_output = w_out_opt @ res_activities

""" Calculate largest 5 latent inputs """
C = np.dot(res_activities.T, res_activities)    # TxT dimensional correlations
C_eig_vals, C_eig_vecs = np.linalg.eig(C)
sorting_indices = C_eig_vals.argsort()[::-1]
C_eig_vecs = np.real(C_eig_vecs[:,sorting_indices])



""" ########################
    Set up learnable readout
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
readout = linear_perceptron(
    N = N_res,  # Number of input traces
    M = M_out,  # Number of output traces
    T = T,      # Number of timesteps, length of signals
    eta = eta,      # Learning rate
    sigma = sigma,  # Perturbation strength
    gamma_decay = 1 # Weight decay const
)

readout.input_traces = res_activities # Apply inputs


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
    """ Unperturbed trial """
    readout.reset_perturbations()
    readout.evolve()
    R0 = get_reward(readout.output, targets, T)

    """ Perturbed trial """
    # Apply perturbations
    if trainings_mode in ["WP", "WP0", "HP"]:
        readout.apply_random_weight_perturbation()
    elif trainings_mode=="NP":
        readout.apply_random_node_perturbation()
    # Evaluate 
    readout.evolve()
    R_pert = get_reward(readout.output, targets, T)
    dR = R_pert-R0

    """ Update """
    if trainings_mode=="WP":
        readout.update_weights_WP(dR)
    elif trainings_mode in ["NP", "HP"]:
        readout.update_weights_NP(dR)
    elif trainings_mode=="WP0":
        readout.update_weights_WP0(dR)

    # Store data
    R0_data[i_trial] = R0
    R_pert_data[i_trial] = R_pert

final_output = np.dot(readout.w_out, res_activities)
R_top5 = get_reward(top5_output, targets, T)

colors = {
    "WP" : "blue",
    "NP" : "orange",
    "WP0": "green",
    "HP" : "red"
}

fig, axes = plt.subplots(figsize=(10, 6), nrows=2, ncols=3)
fig.subplots_adjust(left=0.05, bottom=0.08, right=0.98, top=0.9, wspace=0.27, hspace=0.3)
fig.suptitle("{} learning a ".format(trainings_mode) \
            +r'reservoir computing task. $\sigma_\mathrm{eff} = $' \
            +"{:.1e}".format(sigma_eff))
axes[0,0].set_title("External inputs")
axes[0,0].plot(inputs.T)
axes[0,0].set_xlabel("Time")

axes[1,0].set_title("Reservoir neuronal and latent activity")
largest_res_comps = C_eig_vecs[:,:5] * np.sqrt(T)
axes[1,0].plot(res_activities[:10,:].T, color="gray", alpha=0.5)
axes[1,0].plot(largest_res_comps, color="black")
axes[1,0].set_xlabel("Time")

axes[0,1].set_title("Final output")
axes[0,1].plot(targets[0], targets[1], c="black", label="target")
axes[0,1].plot(final_output[0], final_output[1], 
                c=colors[trainings_mode], label=trainings_mode)
axes[0,1].plot(top5_output[0], top5_output[1], c="gray", label="Top 5 PCA fit")
axes[0,1].set_xlabel("x")
axes[0,1].set_ylabel("y")
axes[0,1].legend()

axes[1,1].set_title("Input strengths (largest 20)")
axes[1,1].bar(np.arange(start=1, stop=21), S_eig_vals[:20])
axes[1,1].axvline(x=PR, color="red")
axes[1,1].set_xlabel("Input component")

axes[0,2].set_title(r'Error (E_f='+"{:.2e})".format(R0_data[-1]))
axes[0,2].plot(-R0_data)
axes[1,2].set_title("Error (log-scale)")
axes[1,2].plot(-R0_data)
axes[1,2].set_yscale("log")
plt.show()