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
        self.parameters = torch.nn.ParameterDict({'mean':mean, 'logstd':logstd})
        
        self.optimizer = torch.optim.Adam(self.parameters.parameters, self.learning_rate)
        
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

        mean, logstd = self.parameters
        with torch.set_grad_enabled(self.training):
            ac = mean(observation) + torch.exp(logstd) * torch.randn_like(mean)
        
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

    def define_placeholders(self):
        # placeholder for observations
        self.observations_pl = tf.placeholder(shape=[None, self.ob_dim], name="ob", dtype=tf.float32)

        # placeholder for actions
        self.actions_pl = tf.placeholder(shape=[None, self.ac_dim], name="ac", dtype=tf.float32)

        if self.training:
            self.acs_labels_na = tf.placeholder(shape=[None, self.ac_dim], name="labels", dtype=tf.float32)


    def update(self, observations, actions):

        assert self.training, 'Policy must be created with training=True in order to perform training updates...'
        
        true_actions = actions
        self.optimizer.zero_grad()        
        pred_actions = self.get_action(observations)
        loss = self.loss_func(pred_actions, true_actions)
        loss.backward()
        self.optimizer.step(loss)


