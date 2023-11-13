import torch
import gpytorch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        input_dim = train_x.shape[1]
        num_tasks = train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim), num_tasks=num_tasks, rank=1
        )
 
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class MultiOutputGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks
        )
 
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class Trainer(object):
    def __init__(self, gpr, likelihood, optimizer, mll):
        self.gpr = gpr
        self.likelihood = likelihood
        self.optimizer = optimizer
        self.mll = mll

    def update_hyperparameter(self, epochs):
        self.gpr.train()
        self.likelihood.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.gpr(self.gpr.train_inputs[0])

            loss = - self.mll(output, self.gpr.train_targets)
            loss.backward()
            self.optimizer.step()

            if (epoch+1) % ((epochs)//10) == 0:
                print('Epoch %d/%d - Loss: %.3f ' % (
                    epoch + 1, epochs, loss.item(),
                    ))
            elif (epoch+1) == 1:
                print('Epoch %d/%d - Loss: %.3f ' % (
                    epoch + 1, epochs, loss.item(),
                    ))               
        
        return self.gpr, self.likelihood