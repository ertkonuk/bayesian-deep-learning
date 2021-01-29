import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, priors=None, lrt=False, device='cuda'):
        super(BayesianConv2d, self).__init__()

        # . . conv2d module parameters
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = (kernel_size, kernel_size)
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation
        self.groups       = 1
        self.bias         = bias
        # . . local reparameterization trick: doubles the training time
        self.lrt          = lrt
        # . . cuda or cpu
        self.device       = device

        # . . the default priors
        # . . 
        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'initial_posterior_mu' : ( 0.0, 0.1), # . . (mean, std)
                'initial_posterior_rho': (-5.0, 0.1), # . . (mean, std)
            }

        
        # . . prior probabilities
        self.prior_mu      = priors['prior_mu']
        self.prior_sigma   = priors['prior_sigma']
        self.posterior_mu  = priors['initial_posterior_mu']
        self.posterior_rho = priors['initial_posterior_rho']

        # . . mu and rho weights
        self.W_mu  = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))
        self.W_rho = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size), device=self.device))

        # . . if the conv layer has bias
        if bias:
            self.bias_mu  = nn.Parameter(torch.empty(out_channels, device=self.device))
            self.bias_rho = nn.Parameter(torch.empty(out_channels, device=self.device))
        else:
            self.register_parameter('bias_mu' , None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_( *self.posterior_mu )
        self.W_rho.data.normal_(*self.posterior_rho)

        if self.bias:
            self.bias_mu.data.normal_( *self.posterior_mu )
            self.bias_rho.data.normal_(*self.posterior_rho)


    # . . forward propagation
    def forward(self, x, sample=True):
        if self.lrt:
            return self.forward_lrt(x, sample)
        else:
            return self.forward_nolrt(x, sample)

            
    # . . forward propagation: bayes by backprop with local reparameterization trick
    def forward_lrt(self, x, sample=True):
        # . . a small number to ebsure that variance is not zero
        small_number = 1e-15

        # . . ensure that sigma is always positive
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))

        if self.bias:
            # . . bias of sigma
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            # . . bias of variance
            bias_var = self.bias_sigma.pow(2)
        else:
            self.bias_sigma = bias_var = None

        # . . the conv nets for mean and the variance
        # . . the mean of the posterior
        mu  = F.conv2d(x, self.W_mu, self.bias_mu, self.stride, self.padding, self.dilation, self.groups)
        # . . the variance of the posterior
        variance = F.conv2d(x.pow(2), self.W_sigma.pow(2), bias_var, self.stride, self.padding, self.dilation, self.groups) + small_number
        # . . the standard deviation
        std = torch.sqrt(variance)

        # . . if train the model or sample from the posterior
        if self.training or sample:
            # . . sample random noise from the normal distribution
            eps = torch.empty(mu.size()).normal_(0, 1).to(self.device)
            # . . shift by mean and scale by the standard deviation
            return mu + std * eps
        else:
            # . . just return the mean
            return mu

    # . . forward propagation: bayes by backprop
    def forward_nolrt(self, x, sample=True):
        
        # . . if train the model or sample from the posterior
        if self.training or sample:
            # . . ensure that sigma is always positive
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                # . . bias of sigma
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                # . . bias of
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias   = self.bias_mu if self.bias else None

        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        kldiv = 0.5 * (2 * torch.log(sigma_p / sigma_q) - 1 + (sigma_q / sigma_p).pow(2) + ((mu_p - mu_q) / sigma_p).pow(2)).sum()
        return kldiv

    # . . the KL divergence loss
    def kl_div_loss(self):
        kl_div = self.kl_div(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)

        if self.bias:
            kl_div += self.kl_div(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)

        return kl_div
            


class BayesianLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, priors=None, lrt=False, device='cuda'):
        super(BayesianLinear, self).__init__()

        # . . parameters
        self.in_features  = in_features
        self.out_features = out_features
        self.bias         = bias
        # . . local reparameterization trick: doubles the training time
        self.lrt          = lrt
        # . . cuda or cpu
        self.device       = device

        # . . the default priors
        # . . 
        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'initial_posterior_mu' : ( 0.0, 0.1), # . . (mean, std)
                'initial_posterior_rho': (-5.0, 0.1), # . . (mean, std)
            }
        
        # . . prior probabilities
        self.prior_mu      = priors['prior_mu']
        self.prior_sigma   = priors['prior_sigma']
        self.posterior_mu  = priors['initial_posterior_mu']
        self.posterior_rho = priors['initial_posterior_rho']

        # . . mu and rho weights
        self.W_mu  = nn.Parameter(torch.empty((out_features, in_features), device=self.device))
        self.W_rho = nn.Parameter(torch.empty((out_features, in_features), device=self.device))

        # . . if the conv layer has bias
        if bias:
            self.bias_mu  = nn.Parameter(torch.empty(out_features, device=self.device))
            self.bias_rho = nn.Parameter(torch.empty(out_features, device=self.device))
        else:
            self.register_parameter('bias_mu' , None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.W_mu.data.normal_( *self.posterior_mu )
        self.W_rho.data.normal_(*self.posterior_rho)

        if self.bias:
            self.bias_mu.data.normal_( *self.posterior_mu )
            self.bias_rho.data.normal_(*self.posterior_rho)

    # . . forward propagation
    def forward(self, x, sample=True):
        if self.lrt:
            return self.forward_lrt(x, sample)
        else:
            return self.forward_nolrt(x, sample)

    # . . forward propagation: bayes by backprop with local reparameterization trick
    def forward_lrt(self, x, sample=True):
        # . . a small number to ebsure that variance is not zero
        snum = 1e-15

        # . . ensure that sigma is always positive
        self.W_sigma = torch.log1p(torch.exp(self.W_rho))

        if self.bias:
            # . . bias of sigma
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            # . . bias of variance
            bias_var = self.bias_sigma.pow(2)
        else:
            self.bias_sigma = bias_var = None

        # . . the conv nets for mean and the variance
        # . . the mean of the posterior
        mu  = F.linear(x, self.W_mu, self.bias_mu)
        # . . the variance of the posterior
        variance = F.linear(x.pow(2), self.W_sigma.pow(2), bias_var) + snum
        # . . the standard deviation
        std = torch.sqrt(variance)

        # . . if train the model or sample from the posterior
        if self.training or sample:
            # . . sample random noise from the normal distribution
            eps = torch.empty(mu.size()).normal_(0, 1).to(self.device)
            # . . shift by mean and scale by the standard deviation
            return mu + std * eps
        else:
            # . . just return the mean
            return mu

    # . . forward propagation: bayes by backprop
    def forward_nolrt(self, x, sample=True):

        # . . if train the model or sample from the posterior
        if self.training or sample:
            # . . ensure that sigma is always positive
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).to(self.device)
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                # . . bias of sigma
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                # . . bias of
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias   = self.bias_mu if self.bias else None

        return F.linear(x, weight, bias)

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        kldiv = 0.5 * (2 * torch.log(sigma_p / sigma_q) - 1 + (sigma_q / sigma_p).pow(2) + ((mu_p - mu_q) / sigma_p).pow(2)).sum()
        return kldiv

    # . . the KL divergence loss
    def kl_div_loss(self):
        kl_div = self.kl_div(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)

        if self.bias:
            kl_div += self.kl_div(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)

        return kl_div
                        