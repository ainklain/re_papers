import numpy as np
import torch
from torch import nn
from .base_policy import BasePolicy
from cs285.infrastructure.torch_utils import MLP

class MLPPolicy(BasePolicy):

    def __init__(self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        learning_rate=1e-4,
        discrete=False, # unused for now
        nn_baseline=False, # unused for now
        **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate

        # build torch graph
        self.build_graph()
        self.loss_func = torch.nn.MSELoss()

    ##################################

    def build_graph(self):
        mean = MLP(self.ob_dim, output_size=self.ac_dim, n_layers=self.n_layers, size=self.size)
        logstd = torch.zeros(self.ac_dim, dtype=torch.float32, requires_grad=True)
        self.parameters = {'mean':mean, 'logstd':nn.Parameter(logstd)}
        
        self.optimizer = torch.optim.Adam([{
            'params': self.parameters['mean'].parameters(), 
            'params': self.parameters['logstd']}
            ], 
            self.learning_rate)
        
    ##################################

    def save(self, filepath):
        save_path = os.path.join(filepath, "saved_model.pt")
        torch.save({
            'ep': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print('model saved successfully. ({})'.format(path))


    def load(self, filepath):
        load_path = os.path.join(filepath, "saved_model.pt")
        if not os.path.exists(load_path):
            return False

        checkpoint = torch.load(load_path)
        self.parameters.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(tu.device)
        model.eval()
        print('model loaded successfully. ({})'.format(filepath))
        return checkpoint['ep']

    ##################################

    # query this policy with observation(s) to get selected action(s)
    def get_action(self, obs):
        if len(obs.shape)>1:
            observation = obs
        else:
            observation = obs[None]

        with torch.set_grad_enabled(self.training):
            mean = self.parameters['mean'](torch.tensor(observation, dtype=torch.float32))
            logstd = self.parameters['logstd']
            ac = mean + torch.exp(logstd) * torch.randn_like(mean)
        
        return ac

    # update/train this policy
    def update(self, observations, actions):
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):

    """
        This class is a special case of MLPPolicy,
        which is trained using supervised learning.
        The relevant functions to define are included below.
    """

    def update(self, observations, actions):

        assert self.training, 'Policy must be created with training=True in order to perform training updates...'
        
        true_actions = actions
        self.optimizer.zero_grad()        
        pred_actions = self.get_action(observations)
        loss = self.loss_func(pred_actions, true_actions)
        loss.backward()
        self.optimizer.step(loss)


