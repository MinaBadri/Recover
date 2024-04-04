import torch
import numpy as np
import torch.nn as nn
import torchbnn as bnn
from torch.nn import Parameter
import math
import torch.nn.functional as F
########################################################################################################################
# Modules
########################################################################################################################


class LinearModule(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearModule, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class ReLUModule(nn.ReLU):
    def __init__(self):
        super(ReLUModule, self).__init__()

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]
        
class ScaledSigmoid(nn.Module):
    def __init__(self, scale_factor=100):
        super(ScaledSigmoid, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        if isinstance(x, list):
            # Apply sigmoid to each tensor in the list
            return [self.scale_factor * F.sigmoid(tensor) for tensor in x]
        else:
            # Apply sigmoid to a single tensor
            return self.scale_factor * F.sigmoid(x)

class FilmModule(torch.nn.Module):
    def __init__(self, num_cell_lines, out_dim):
        super(FilmModule, self).__init__()
        film_init = 1 / 100 * torch.randn(num_cell_lines, 2 * out_dim)
        film_init = film_init + torch.Tensor([([1] * out_dim) + ([0] * out_dim)])

        self.film = Parameter(film_init)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [
            self.film[cell_line][:, : x.shape[1]] * x
            + self.film[cell_line][:, x.shape[1]:],
            cell_line]


class FilmWithFeatureModule(torch.nn.Module):

    def __init__(self, num_cell_line_features, out_dim):
        super(FilmWithFeatureModule, self).__init__()

        self.out_dim = out_dim

        self.condit_lin_1 = nn.Linear(num_cell_line_features, num_cell_line_features)
        self.condit_relu = nn.ReLU()
        self.condit_lin_2 = nn.Linear(num_cell_line_features, 2 * out_dim)

        # Change initialization of the bias so that the expectation of the output is 1 for the first columns
        self.condit_lin_2.bias.data[: out_dim] += 1

    def forward(self, input):
        x, cell_line_features = input[0], input[1]

        # Compute conditioning
        condit = self.condit_lin_2(self.condit_relu(self.condit_lin_1(cell_line_features)))

        return [
            condit[:, :self.out_dim] * x
            + condit[:, self.out_dim:],
            cell_line_features
        ]



class LinearFilmWithFeatureModule(nn.Module):
    def __init__(self, num_cell_line_features, out_dim):
        super(LinearFilmWithFeatureModule, self).__init__()

        self.out_dim = out_dim

        self.condit_lin_1 = nn.Linear(num_cell_line_features, 2 * out_dim)

        # Change initialization of the bias so that the expectation of the output is 1 for the first columns
        self.condit_lin_1.bias.data[: out_dim] += 1

    def forward(self, input):
        x, cell_line_features = input[0], input[1]

        # Compute conditioning
        condit = self.condit_lin_1(cell_line_features)

        return [
            condit[:, :self.out_dim] * x
            + condit[:, self.out_dim:],
            cell_line_features
        ]
class SimpleBayesianLinearModule(bnn.BayesLinear):
    def __init__(self, in_features, out_features, prior_mu=0, prior_sigma=0.01):
        super(SimpleBayesianLinearModule, self).__init__(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=in_features, out_features=out_features)

    def forward(self, input):
        x, cell_line = input[0], input[1]
        return [super().forward(x), cell_line]


class ScaleMixtureGaussian(object): #scale mixture Gaussian
    def __init__(self, pi , sigma1, sigma2): #, sigmas):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
        # self.gaussians = [torch.distributions.Normal (0, sigma) for sigma in sigmas]
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        
        # Normalize probabilities
        total_prob = self.pi * prob1 + (1 - self.pi) * prob2
        normalized_prob1 = self.pi * prob1 / total_prob
        normalized_prob2 = (1 - self.pi) * prob2 / total_prob

        # Calculate log probability and sum
        log_prob = torch.log(normalized_prob1 + normalized_prob2)
        

        if self.pi == 1:
            return torch.log(normalized_prob1).sum()
        # return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()
        return log_prob.sum()
    

# Hyperparameters for the mixture model
PI =  0.25 #if pi = 0 and sigma 2 = -4.6, it would be the same as after merge implementation
SIGMA_1 = torch.FloatTensor([math.exp(-0)]) #torch.FloatTensor([0.005]) #
SIGMA_2 = torch.FloatTensor([math.exp(-6)]) #

class BayesianLinearModule(nn.Linear):
    def __init__(self, in_features, out_features): #, config):
        super().__init__(in_features, out_features) #BayesianLinearModule, self
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        #self.weight = Gaussian(self.weight_mu, self.weight_rho)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))

        # self.number_gaussian = config["number_gaussian"]

        # self.pi = nn.Parameter(torch.Tensor(self.number_gaussian).uniform_())
        # self.pi.data = self.pi.data / self.pi.data.sum()
        # self.sigmas = [torch.FloatTensor([math.exp(-i)]) for i in range(self.number_gaussian)]
     
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def sample_bias(self):
        # Sample bias using reparameterization trick
        epsilon = torch.distributions.Normal(0,1).sample(self.bias_rho.size())
        sigma = torch.log1p(torch.exp(self.bias_rho))
        return self.bias_mu + sigma * epsilon

    def sample_weight(self):
        # Sample weight using reparameterization trick
        epsilon = torch.distributions.Normal(0,1).sample(self.weight_rho.size())
        sigma = torch.log1p(torch.exp(self.weight_rho))
        return self.weight_mu + sigma * epsilon
    
    def log_prob_bias(self, input):
        # Calculate the log probability of the bias
        sigma = torch.log1p(torch.exp(self.bias_rho))
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.bias_mu) ** 2) / (2 * sigma ** 2)).sum()
        # return self.bias_prior.log_prob(input[0])

    def log_prob_weight(self, input):
        # Calculate the log probability of the Weight
        sigma = torch.log1p(torch.exp(self.weight_rho))
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(sigma)
                - ((input - self.weight_mu) ** 2) / (2 * sigma ** 2)).sum()
        # return self.weight_prior.log_prob(input[0])

    
    def forward(self, input, sample=True):
        x, cell_line = input[0], input[1] #BAYESIAN ADD ON
        
        # Sample weights and biases
        # weight_mu = self.weight_mu
        # weight_rho = self.weight_rho
        # bias_mu = self.bias_mu
        # bias_rho = self.bias_rho
        
        weight = self.sample_weight()
        bias = self.sample_bias()

        if self.training and sample:
            # Calculate the log prior and log variational posterior during training
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.log_prob_weight(weight) + self.log_prob_bias(bias)
                    
        else:
            # Set prior and variational posterior to zero during evaluation
            self.log_prior, self.log_variational_posterior = torch.FloatTensor([0]), torch.FloatTensor([0])

        return [F.linear(x, weight, bias), cell_line]

    def kl_loss(self):
        # Calculate the Kullback-Leibler divergence loss
        return self.log_variational_posterior - self.log_prior

##################
# class ScaleMixtureGaussian(object):
#     def __init__(self, pis, sigmas):
#         super().__init__()
#         self.pis = pis
#         self.sigmas = sigmas
#         self.gaussians = [torch.distributions.Normal(0, sigma) for sigma in sigmas]

#     def log_prob(self, input):
#         probs = [g.log_prob(input) for g in self.gaussians]
#         weighted_probs = [pi * prob.exp() for pi, prob in zip(self.pis, probs)]
#         print("torch.log(sum(weighted_probs)).sum(): ", torch.log(sum(weighted_probs)).sum())
#         return torch.log(sum(weighted_probs)).sum()

# # Hyperparameters for the mixture model
# N = 4  # Number of Gaussians
# SIGMAS = [torch.FloatTensor([math.exp(-i)]) for i in range(1, N + 1)]  

# class BayesianLinearModule(nn.Linear):
#     def __init__(self, in_features, out_features):
#         super().__init__(in_features, out_features)
#         self.in_features = in_features
#         self.out_features = out_features

#         # Weight parameters
#         self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
#         self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

#         # Bias parameters
#         self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
#         self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

#         print("self.weight_mu.shape: ", self.weight_mu.shape)
#         print("self.weight_rho.shape: ",self.weight_rho.shape)

#         # Calculate pi values based on the weight parameters
#         # weights = [(self.weight_mu[i].item() + torch.log1p(torch.exp(self.weight_rho[i])).item()) for i in range(N)]
#         weights = [(self.weight_mu[i, j].item() + torch.log1p(torch.exp(self.weight_rho[i, j])).item())
#                     for i in range(self.weight_mu.size(0)) for j in range(self.weight_mu.size(1))]

#         total_weights = sum(weights)
#         self.pis = [weight / total_weights for weight in weights]

#         # Prior distributions
#         self.weight_prior = ScaleMixtureGaussian(self.pis, SIGMAS)
#         self.bias_prior = ScaleMixtureGaussian(self.pis, SIGMAS)
#         self.log_prior = 0
#         self.log_variational_posterior = 0

#     def sample_bias(self):
#         # Sample bias using reparameterization trick
#         epsilon = torch.distributions.Normal(0,1).sample(self.bias_rho.size())
#         sigma = torch.log1p(torch.exp(self.bias_rho))
#         return self.bias_mu + sigma * epsilon

#     def sample_weight(self):
#         # Sample weight using reparameterization trick
#         epsilon = torch.distributions.Normal(0,1).sample(self.weight_rho.size())
#         sigma = torch.log1p(torch.exp(self.weight_rho))
#         return self.weight_mu + sigma * epsilon
    
#     def log_prob_bias(self, input):
#         # Calculate the log probability of the bias
#         sigma = torch.log1p(torch.exp(self.bias_rho))
#         return (-math.log(math.sqrt(2 * math.pi))
#                 - torch.log(sigma)
#                 - ((input - self.bias_mu) ** 2) / (2 * sigma ** 2)).sum()
#         # return self.bias_prior.log_prob(input[0])

#     def log_prob_weight(self, input):
#         # Calculate the log probability of the Weight
#         sigma = torch.log1p(torch.exp(self.weight_rho))
#         return (-math.log(math.sqrt(2 * math.pi))
#                 - torch.log(sigma)
#                 - ((input - self.weight_mu) ** 2) / (2 * sigma ** 2)).sum()
#         # return self.weight_prior.log_prob(input[0])

    
#     def forward(self, input, sample=True):
#         x, cell_line = input[0], input[1] #BAYESIAN ADD ON
        
#         # Sample weights and biases
#         # weight_mu = self.weight_mu
#         # weight_rho = self.weight_rho
#         # bias_mu = self.bias_mu
#         # bias_rho = self.bias_rho
        
#         weight = self.sample_weight()
#         bias = self.sample_bias()


#         # if self.training and sample:
#         #     # Calculate the log prior and log variational posterior during training
#         #     self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
#         #     self.log_variational_posterior = self.log_prob_weight(weight) + self.log_prob_bias(bias)
          
#         # else:
#         #     # Set prior and variational posterior to zero during evaluation
#         #     self.log_prior, self.log_variational_posterior = torch.FloatTensor([0]), torch.FloatTensor([0])
        
#         # print("Maximum value in weight tensor:", torch.max(weight).item())
#         # print("Minimum value in weight tensor:", torch.min(weight).item())

#         self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
#         self.log_variational_posterior = self.log_prob_weight(weight) + self.log_prob_bias(bias)

#         return [F.linear(x, weight, bias), cell_line]
        
    
#     def kl_loss(self):
#         # Calculate the Kullback-Leibler divergence loss
#         return self.log_variational_posterior - self.log_prior
# # def kl_loss(self):
# #     # Calculate the Kullback-Leibler divergence loss
# #     kl_loss = 0.5 * torch.sum(self.weight_prior.log_prob(self.weight_mu) - self.log_variational_posterior)
# #     kl_loss += 0.5 * torch.sum(self.bias_prior.log_prob(self.bias_mu) - self.log_variational_posterior)
# #     return kl_loss




#Sparse Variational Dropout(SVDO) on Bayesian Nueral Network
class BayesianLinearDropoutModule(nn.Linear):
    def __init__(self, in_features, out_features, threshold = 3, prior_mu=0, prior_sigma=0.01):
        super().__init__(in_features, out_features) 
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.W_mu = Parameter(torch.Tensor(out_features, in_features).normal_(mean=prior_mu, std=prior_sigma))
        self.W_rho = Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))  # Initialize rho
        self.bias_mu = Parameter(torch.Tensor(out_features).normal_(mean=prior_mu, std=prior_sigma))
        self.bias_rho = Parameter(torch.Tensor(out_features).uniform_(-5, -4))  # Initialize rho
        
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     self.W_rho.data.fill_(-5)  # Reset rho to a constant value
    #     self.bias_rho.data.fill_(-5)  # Reset rho for bias to a constant value
        
    def sample_weight(self):
        epsilon = torch.randn_like(self.W_mu)  # Sample random noise
        sigma = torch.log1p(torch.exp(self.W_rho))  # Calculate sigma from rho
        return self.W_mu + sigma * epsilon  # Return sampled weight
    
    def sample_bias(self):
        epsilon = torch.randn_like(self.bias_mu)  # Sample random noise
        sigma = torch.log1p(torch.exp(self.bias_rho))  # Calculate sigma from rho
        return self.bias_mu + sigma * epsilon  # Return sampled bias
    
    def kl_loss(self):
        # Calculate KL divergence regularization term
        kl = 0.5 * (self.W_rho.exp().pow(2) + self.W_mu.pow(2) - 2 * self.W_rho - 1).sum() + \
             0.5 * (self.bias_rho.exp().pow(2) + self.bias_mu.pow(2) - 2 * self.bias_rho - 1).sum()
        return kl
        
    def forward(self, input):
        x, cell_line = input[0], input[1]
        
        # Sample weights and biases
        weight = self.sample_weight()
        bias = self.sample_bias()
        
        # Calculate log_alpha
        log_alpha = self.W_rho * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(weight))
        log_alpha = torch.clamp(log_alpha, -10, 10) 
        
        if self.training:
            lrt_mean = F.linear(x, weight) + bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(2 * self.W_rho)) + 1e-8)
            eps = torch.randn_like(lrt_std)
            return [lrt_mean + lrt_std * eps, cell_line]
    
        # Apply sparse variational dropout during inference
        return [F.linear(x, weight * (log_alpha < self.threshold).float()) + bias, cell_line]


class LaplacePrior(object):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.laplace = torch.distributions.Laplace(0, scale)
    
    def log_prob(self, input):
        return self.laplace.log_prob(input).sum()

# Hyperparameter for Laplace prior
SCALE = torch.FloatTensor([math.exp(-4)]) 

class LaplaceBayesianLinearModule(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        # Prior distributions
        self.weight_prior = LaplacePrior(SCALE)
        self.bias_prior = LaplacePrior(SCALE)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def sample_bias(self):
        epsilon = torch.distributions.Normal(0, 1).sample(self.bias_rho.size())
        sigma = torch.log1p(torch.exp(self.bias_rho))
        return self.bias_mu + sigma * epsilon

    def sample_weight(self):
        epsilon = torch.distributions.Normal(0, 1).sample(self.weight_rho.size())
        sigma = torch.log1p(torch.exp(self.weight_rho))
        return self.weight_mu + sigma * epsilon
    
    def log_prob_bias(self, input):
        sigma = torch.log1p(torch.exp(self.bias_rho))
        return (-math.log(2 * self.bias_prior.scale)
            - torch.abs(input - self.bias_mu) / sigma
            - torch.log(sigma)).sum()

    def log_prob_weight(self, input):
        sigma = torch.log1p(torch.exp(self.weight_rho))
        return (-math.log(2 * self.weight_prior.scale)
            - torch.abs(input - self.weight_mu) / sigma
            - torch.log(sigma)).sum()
    
    def forward(self, input, sample=True):
        x, cell_line = input[0], input[1]

        weight = self.sample_weight()
        bias = self.sample_bias()

        if self.training and sample:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.log_prob_weight(weight) + self.log_prob_bias(bias)
          
        else:
            self.log_prior, self.log_variational_posterior = torch.FloatTensor([0]), torch.FloatTensor([0])

        return [F.linear(x, weight, bias), cell_line]
    
    def kl_loss(self):
        return self.log_variational_posterior - self.log_prior

########################################################################################################################
# Advanced Bayesian MLP
########################################################################################################################

class AdvancedBayesianBilinearMLPPredictor(nn.Module): #BAYESIAN ADD ON

    def __init__(self, data, config, predictor_layers):
        super().__init__()

        # Initialize model parameters
        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.layer_dims = predictor_layers
        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]
        
        # self.layers = BayesianLinearModule(in_features, out_features)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]

        
        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)
        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            # layers_before_merge = self.add_bayesian_layer(
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )
        
        # Build last layers (after addition of the two embeddings)
        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_bayesian_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        # Define sequential modules for layers
        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        # self.before_merge_mlp = layers_before_merge
        # self.after_merge_mlp = layers_after_merge

        # Number of bilinear transformations == the dimension of the layer at which the merge is performed
        # Initialize weights close to identity
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))
        self.allow_neg_eigval = config["allow_neg_eigval"]
        if self.allow_neg_eigval:
            self.bilinear_diag = Parameter(1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)

    def forward(self, data, drug_drug_batch, sample=True):

        # Extract drug embeddings and cell line information
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply MLP before the merge operation
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]
        # # We do
        # input_1 = [h_drug_1, cell_lines]
        # for layer in self.before_merge_mlp:
        #     # if layer.__class__.__name__ == 'BayesianLinearModule':
        #     #     input_1 = layer(input_1, sample=False)
        #     # else:
        #     #     input_1 = layer(input_1)
        #     input_1 = layer(input_1)

        # input_2 = [h_drug_2, cell_lines]
        # for layer in self.before_merge_mlp:
        #     # if layer.__class__.__name__ == 'BayesianLinearModule':
        #     #     input_2 = layer(input_2, sample=False)
        #     # else:
        #     #     input_2 = layer(input_2)
        #     input_2 = layer(input_2)

        # h_1 = input_1[0]
        # h_2 = input_2[0]

        # compute <W.h_1, W.h_2> = h_1.T . W.T.W . h_2

        # Apply bilinear transformation
        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T
        
        if self.allow_neg_eigval:
            # Multiply by diagonal matrix to allow for negative eigenvalues
            h_2 *= self.bilinear_diag
        # "Transpose" h_1
        h_1 = h_1.permute(0, 2, 1)
        # Multiplication
        h_1_scal_h_2 = (h_1 * h_2).sum(1)
        # Add offset
        h_1_scal_h_2 += self.bilinear_offsets

        # Apply MLP after the merge operation
        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        # We do
        # input_3 = [h_1_scal_h_2, cell_lines]
        # for layer in self.after_merge_mlp:
        #     input_3 = layer(input_3)#, sample=True)
         
        # comb = input_3[0]

        return comb
    
    def get_batch(self, data, drug_drug_batch):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]
        return h_drug_1, h_drug_2, cell_lines
    
    def add_layer(self, layers, i, dim_i, dim_i_plus_1):

        # Add a layer to the list of layers
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())
        return layers

    def add_bayesian_layer(self, layers, i, dim_i, dim_i_plus_1):

        # Add a Bayesian layer to the list of layers
        # layers.extend(self.bayesian_linear_layer(i, mu, sigma, dim_i, dim_i_plus_1))
        # bayesian_linear_layer = [BayesianLinearModule(dim_i, dim_i_plus_1)]
        
        layers.extend(self.bayesian_linear_layer(i, dim_i, dim_i_plus_1))
        
        
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())
        # else:
        #     # layers.append(nn.Sigmoid() * 100)
        #     layers.append(ScaledSigmoid(scale_factor=100))
        return layers

    def bayesian_linear_layer(self, i, dim_i, dim_i_plus_1):
        # Return a Bayesian linear layer
        # return [BayesianLinearDropoutModule(dim_i, dim_i_plus_1)]
        if self.bayesian_single_prior:
            return [SimpleBayesianLinearModule(dim_i, dim_i_plus_1)]

        elif self.variational_dropout:
            return [BayesianLinearDropoutModule(dim_i, dim_i_plus_1)]
        
        elif self.Laplace_prior:
            return [LaplaceBayesianLinearModule(dim_i, dim_i_plus_1)]

        else:
            return [BayesianLinearModule(dim_i, dim_i_plus_1)]
            
        

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        # Return a linear layer
        return [LinearModule(dim_i, dim_i_plus_1)]

    def kl_loss(self): 
        # Calculate the total KL divergence loss
        if self.bayesian_single_prior:
            pass    
        else:
            kl = 0
            for layer in self.before_merge_mlp:
                if hasattr(layer, "kl_loss"):
                    kl += layer.kl_loss()
            for layer in self.after_merge_mlp:
                if hasattr(layer, "kl_loss"):
                    kl += layer.kl_loss()
            
            return kl
    
#Recover Original MLPs 
########################################################################################################################
# Bilinear MLP
########################################################################################################################

class BilinearMLPPredictor(nn.Module):

    def __init__(self, data, config, predictor_layers):

        super(BilinearMLPPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.layer_dims = predictor_layers

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        # Build last layers (after addition of the two embeddings)
        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):

            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        # Number of bilinear transformations == the dimension of the layer at which the merge is performed
        # Initialize weights close to identity
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

        self.allow_neg_eigval = config["allow_neg_eigval"]
        if self.allow_neg_eigval:
            self.bilinear_diag = Parameter(1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        # compute <W.h_1, W.h_2> = h_1.T . W.T.W . h_2
        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        if self.allow_neg_eigval:
            # Multiply by diagonal matrix to allow for negative eigenvalues
            h_2 *= self.bilinear_diag

        # "Transpose" h_1
        h_1 = h_1.permute(0, 2, 1)

        # Multiplication
        h_1_scal_h_2 = (h_1 * h_2).sum(1)

        # Add offset
        h_1_scal_h_2 += self.bilinear_offsets

        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


########################################################################################################################
# Bilinear MLP with Film conditioning
########################################################################################################################


class BilinearFilmMLPPredictor(AdvancedBayesianBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(BilinearFilmMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmModule(self.num_cell_lines, self.layer_dims[i + 1])]
  

    # def linear_layer(self, i, dim_i, dim_i_plus_1):
    #     return [BayesianLinearModule(dim_i, dim_i_plus_1), FilmModule(self.num_cell_lines, self.layer_dims[i + 1])]
       

class BilinearFilmWithFeatMLPPredictor(AdvancedBayesianBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        self. cl_features_dim = data.cell_line_features.shape[1]
        super(BilinearFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]
        

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmWithFeatureModule(self.cl_features_dim, self.layer_dims[i + 1])]
        
    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, batch_cl_features


class BilinearLinFilmWithFeatMLPPredictor(BilinearFilmWithFeatMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(BilinearLinFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]


    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), LinearFilmWithFeatureModule(self. cl_features_dim, self.layer_dims[i + 1])]
########################################################################################################################
# No permutation invariance MLP with Bayesian
########################################################################################################################

class BayesianMLPPredictor(nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(BayesianMLPPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.layer_dims = predictor_layers

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        # We will concatenate the two single drug embeddings so the input of the after_merge_mlp is twice its usual dim
        self.layer_dims[- 1 - self.merge_n_layers_before_the_end] *= 2

        # Build last layers (after addition of the two embeddings)
        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):

            layers_after_merge = self.add_bayesian_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        comb = self.after_merge_mlp([torch.cat((h_1, h_2), dim=1), cell_lines])[0]

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

        return layers

    def add_bayesian_layer(self, layers, i, dim_i, dim_i_plus_1):

        # Add a Bayesian layer to the list of layers
        layers.extend(self.bayesian_linear_layer(i, dim_i, dim_i_plus_1))

        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())
        # else:
        #     # layers.append(nn.Sigmoid() * 100)
        #     layers.append(ScaledSigmoid(scale_factor=100))
        return layers

    def bayesian_linear_layer(self, i, dim_i, dim_i_plus_1):
        # Return a Bayesian linear layer
        if self.bayesian_single_prior:
            return [SimpleBayesianLinearModule(dim_i, dim_i_plus_1)]

        elif self.variational_dropout:
            return [BayesianLinearDropoutModule(dim_i, dim_i_plus_1)]
        
        elif self.Laplace_prior:
            return [LaplaceBayesianLinearModule(dim_i, dim_i_plus_1)]

        else:
            return [BayesianLinearModule(dim_i, dim_i_plus_1)]

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        # Return a linear layer
        return [LinearModule(dim_i, dim_i_plus_1)]

    def kl_loss(self): 
        if self.bayesian_single_prior:
            pass
        # Calculate the total KL divergence loss
        else:
            kl = 0
            for layer in self.before_merge_mlp:
                if hasattr(layer, "kl_loss"):
                    kl += layer.kl_loss()
            for layer in self.after_merge_mlp:
                if hasattr(layer, "kl_loss"):
                    kl += layer.kl_loss()
            return kl

########################################################################################################################
# No permutation invariance MLP 
########################################################################################################################

class MLPPredictor(nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(MLPPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.layer_dims = predictor_layers

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        # We will concatenate the two single drug embeddings so the input of the after_merge_mlp is twice its usual dim
        self.layer_dims[- 1 - self.merge_n_layers_before_the_end] *= 2

        # Build last layers (after addition of the two embeddings)
        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):

            layers_after_merge = self.add_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        comb = self.after_merge_mlp([torch.cat((h_1, h_2), dim=1), cell_lines])[0]

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]
########################################################################################################################
# No permutation invariance MLP with Film conditioning
########################################################################################################################


class FilmMLPPredictor(BayesianMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(FilmMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]
        

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmModule(self.num_cell_lines, self.layer_dims[i + 1])]
        

class FilmWithFeatMLPPredictor(BayesianMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        self. cl_features_dim = data.cell_line_features.shape[1]
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]
        super(FilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)
        
    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmWithFeatureModule(self. cl_features_dim, self.layer_dims[i + 1])]
        
    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, batch_cl_features


class LinFilmWithFeatMLPPredictor(FilmWithFeatMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(LinFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), LinearFilmWithFeatureModule(self. cl_features_dim, self.layer_dims[i + 1])]
        

########################################################################################################################
# Deep Synergy
########################################################################################################################

class DeepSynergyPredictor(nn.Module):

    def __init__(self, data, config, predictor_layers):
        super(DeepSynergyPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.cl_features_dim = data.cell_line_features.shape[1]
        predictor_layers[0] += self.cl_features_dim + data.x_drugs.shape[1]

        self.layer_dims = predictor_layers

        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        assert self.merge_n_layers_before_the_end == -1

        layers = []

        # Input dropout
        layers.append(DropoutModule(p=0.2))

        # Build early layers (before addition of the two embeddings)
        for i in range(len(self.layer_dims) - 1):
            layers = self.add_layer(
                layers,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        self.mlp = nn.Sequential(*layers)

        self.normalization_mean = torch.cat([data.x_drugs.mean(dim=0),
                                             data.x_drugs.mean(dim=0),
                                             data.cell_line_features.mean(dim=0)])

        self.normalization_std = torch.cat([data.x_drugs.std(dim=0),
                                             data.x_drugs.std(dim=0),
                                             data.cell_line_features.std(dim=0)])

        self.normalization_std[self.normalization_std == 0] = 1e-2  # Avoid zero std

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, batch_cl_features = self.get_batch(data, drug_drug_batch)

        x_input = torch.cat((h_drug_1, h_drug_2, batch_cl_features), dim=1)
        x_input_permut = torch.cat((h_drug_2, h_drug_1, batch_cl_features), dim=1)

        # Normalization
        x_input = torch.tanh((x_input - self.normalization_mean) / self.normalization_std)
        x_input_permut = torch.tanh((x_input_permut - self.normalization_mean) / self.normalization_std)

        # Apply before merge MLP
        comb = 1/2 * (self.mlp([x_input, batch_cl_features])[0] + self.mlp([x_input_permut, batch_cl_features])[0])

        return comb

    def get_batch(self, data, drug_drug_batch):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, batch_cl_features

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(DropoutModule(p=0.5))
            layers.append(ReLUModule())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]


########################################################################################################################
# Shuffled models
########################################################################################################################


class ShuffledBilinearMLPPredictor(AdvancedBayesianBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):

        # Shuffle the identities of the drugs
        data.x_drugs = data.x_drugs[torch.randperm(data.x_drugs.shape[0])]

        # Shuffle the identities of the cell lines
        value_perm = torch.randperm(len(data.cell_line_to_idx_dict))
        data.cell_line_to_idx_dict = {k: value_perm[v].item() for k, v in data.cell_line_to_idx_dict.items()}

        super(ShuffledBilinearMLPPredictor, self).__init__(data, config, predictor_layers)


class ShuffledBilinearFilmMLPPredictor(ShuffledBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(ShuffledBilinearFilmMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmModule(self.num_cell_lines, self.layer_dims[i + 1])]
        

class ShuffledBilinearFilmWithFeatMLPPredictor(ShuffledBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        self. cl_features_dim = data.cell_line_features.shape[1]
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]
        super(ShuffledBilinearFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmWithFeatureModule(self. cl_features_dim, self.layer_dims[i + 1])]
        
    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, batch_cl_features


class ShuffledBilinearLinFilmWithFeatMLPPredictor(ShuffledBilinearFilmWithFeatMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(ShuffledBilinearLinFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), LinearFilmWithFeatureModule(self. cl_features_dim, self.layer_dims[i + 1])]


########################################################################################################################
# Partially shuffled models
########################################################################################################################


class PartiallyShuffledBilinearMLPPredictor(AdvancedBayesianBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):

        prop_of_shuffled_drugs = config["prop_of_shuffled_drugs"]
        assert 1 >= prop_of_shuffled_drugs >= 0

        if prop_of_shuffled_drugs > 0:
            indices_to_be_shuffled = np.random.choice(data.x_drugs.shape[0],
                                                      size=int(data.x_drugs.shape[0] * prop_of_shuffled_drugs),
                                                      replace=False)

            permuted_indices_to_be_shuffled = np.random.permutation(indices_to_be_shuffled)

            # Shuffle the identities of some of the drugs
            data.x_drugs[indices_to_be_shuffled] = data.x_drugs[permuted_indices_to_be_shuffled]

        super(PartiallyShuffledBilinearMLPPredictor, self).__init__(data, config, predictor_layers)


class PartiallyShuffledBilinearFilmMLPPredictor(PartiallyShuffledBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(PartiallyShuffledBilinearFilmMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]
        

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmModule(self.num_cell_lines, self.layer_dims[i + 1])]
     
class PartiallyShuffledBilinearFilmWithFeatMLPPredictor(PartiallyShuffledBilinearMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        self. cl_features_dim = data.cell_line_features.shape[1]
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]
        super(PartiallyShuffledBilinearFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)
        

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), FilmWithFeatureModule(self. cl_features_dim, self.layer_dims[i + 1])]
       
    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch
        batch_cl_features = data.cell_line_features[cell_lines]

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, batch_cl_features


class PartiallyShuffledBilinearLinFilmWithFeatMLPPredictor(PartiallyShuffledBilinearFilmWithFeatMLPPredictor):
    def __init__(self, data, config, predictor_layers):
        super(PartiallyShuffledBilinearLinFilmWithFeatMLPPredictor, self).__init__(data, config, predictor_layers)
        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]
        

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1), LinearFilmWithFeatureModule(self. cl_features_dim, self.layer_dims[i + 1])]
        
############################
#Advanced Bayesian with one prior
############################
class simpleBayesianBilinearMLPPredictor(torch.nn.Module):
    def __init__(self, data, config, predictor_layers):

        super(simpleBayesianBilinearMLPPredictor, self).__init__()

        self.num_cell_lines = len(data.cell_line_to_idx_dict.keys())
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        self.layer_dims = predictor_layers
        
            
        self.merge_n_layers_before_the_end = config["merge_n_layers_before_the_end"]
        self.merge_dim = self.layer_dims[-self.merge_n_layers_before_the_end - 1]

        self.bayesian_single_prior = config["bayesian_single_prior"]
        self.variational_dropout = config["variational_dropout"]
        self.Laplace_prior = config["Laplace_prior"]

        assert 0 < self.merge_n_layers_before_the_end < len(predictor_layers)

        layers_before_merge = []
        layers_after_merge = []

        # Build early layers (before addition of the two embeddings)
        
        for i in range(len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end):
            layers_before_merge = self.add_layer(
                layers_before_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

        # Build last layers (after addition of the two embeddings)

        for i in range(
            len(self.layer_dims) - 1 - self.merge_n_layers_before_the_end,
            len(self.layer_dims) - 1,
        ):
            layers_after_merge = self.add_bayes_layer(
                layers_after_merge,
                i,
                self.layer_dims[i],
                self.layer_dims[i + 1]
            )

                
        self.before_merge_mlp = nn.Sequential(*layers_before_merge)
        self.after_merge_mlp = nn.Sequential(*layers_after_merge)

        # Number of bilinear transformations == the dimension of the layer at which the merge is performed
        # Initialize weights close to identity
        self.bilinear_weights = Parameter(
            1 / 100 * torch.randn((self.merge_dim, self.merge_dim, self.merge_dim))
            + torch.cat([torch.eye(self.merge_dim)[None, :, :]] * self.merge_dim, dim=0)
        )
        self.bilinear_offsets = Parameter(1 / 100 * torch.randn((self.merge_dim)))

        self.allow_neg_eigval = config["allow_neg_eigval"]
        if self.allow_neg_eigval:
            self.bilinear_diag = Parameter(1 / 100 * torch.randn((self.merge_dim, self.merge_dim)) + 1)

    def forward(self, data, drug_drug_batch):
        h_drug_1, h_drug_2, cell_lines = self.get_batch(data, drug_drug_batch)

        # Apply before merge MLP
        h_1 = self.before_merge_mlp([h_drug_1, cell_lines])[0]
        h_2 = self.before_merge_mlp([h_drug_2, cell_lines])[0]

        # compute <W.h_1, W.h_2> = h_1.T . W.T.W . h_2
        h_1 = self.bilinear_weights.matmul(h_1.T).T
        h_2 = self.bilinear_weights.matmul(h_2.T).T

        if self.allow_neg_eigval:
            # Multiply by diagonal matrix to allow for negative eigenvalues
            h_2 *= self.bilinear_diag

        # "Transpose" h_1
        h_1 = h_1.permute(0, 2, 1)

        # Multiplication
        h_1_scal_h_2 = (h_1 * h_2).sum(1)

        # Add offset
        h_1_scal_h_2 += self.bilinear_offsets

        comb = self.after_merge_mlp([h_1_scal_h_2, cell_lines])[0]

        return comb

    def get_batch(self, data, drug_drug_batch):

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1 = data.x_drugs[drug_1s]
        h_drug_2 = data.x_drugs[drug_2s]

        return h_drug_1, h_drug_2, cell_lines

    def add_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.linear_layer(i, dim_i, dim_i_plus_1))
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())

        return layers

    def add_bayes_layer(self, layers, i, dim_i, dim_i_plus_1):
        layers.extend(self.bayes_linear_layer(i, dim_i, dim_i_plus_1))
        
        if i != len(self.layer_dims) - 2:
            layers.append(ReLUModule())
            
        # if self.sigmoid:
        #     if i == len(self.layer_dims) - 2:
        #         layers.append(ScaledSigmoid())

        return layers

    def linear_layer(self, i, dim_i, dim_i_plus_1):
        return [LinearModule(dim_i, dim_i_plus_1)]
        
    def bayes_linear_layer(self, i, dim_i, dim_i_plus_1):

        if self.bayesian_single_prior:
            return [SimpleBayesianLinearModule(dim_i, dim_i_plus_1)]
        elif self.variational_dropout:
            return [BayesianLinearDropoutModule(dim_i, dim_i_plus_1)]
        