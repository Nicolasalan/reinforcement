import numpy as np
import random

from model import Actor, Critic # importar modelo de rede neural
from Noise import OUNoise # importar classe de ruído
from replaybuffer import ReplayBuffer # importar classe de buffer de replay

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # tamanho do buffer de replay
BATCH_SIZE = 128        # tamanho do batch
GAMMA = 0.99            # fator de desconto
TAU = 1e-3              # fator de atualização do target network
LR_ACTOR = 1e-3         # taxa de aprendizado do ator 
LR_CRITIC = 1e-3        # taxa de aprendizado do crítico
WEIGHT_DECAY = 0        # decai o peso do gradiente

LEARN_EVERY = 20        # intervalo de tempo de aprendizado
LEARN_NUM = 10          # número de vezes que o agente aprende

N_LEARN_UPDATES = 20    # número de atualizações de aprendizagem
N_TIME_STEPS = 10       # cada n passo de tempo atualize

EPSILON = 1.0           # explorar->explorar processo de ruído adicionado à etapa de ação
EPSILON_DECAY = 1e-6    # taxa de decaimento para processo de ruído

# utilizar GPU se disponível
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interage e aprende com o ambiente."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Inicializar um objeto Agente.
        
        Parâmetros
        ======
            state_size (int): dimensão de cada estado
            action_size (int): dimensão de cada ação
            random_seed (int): semente aleatória
        """
        self.state_size = state_size # dimensão do estado
        self.action_size = action_size  # dimensão da ação
        self.seed = random.seed(random_seed) # semente aleatória
        self.epsilon = EPSILON # exploracao

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device) # rede neural local do ator
        self.actor_target = Actor(state_size, action_size, random_seed).to(device) # rede neural alvo do ator
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR) # otimizador do ator

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device) # rede neural local do crítico
        self.critic_target = Critic(state_size, action_size, random_seed).to(device) # rede neural alvo do crítico
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) # otimizador do crítico

        # Processo de ruído
        self.noise = OUNoise(action_size, random_seed) # processo de ruído

        # Memória de repetição
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed) # buffer de replay
    
    def step(self, state, action, reward, next_state, done, timestep):
        """Salve a experiência na memória de repetição e use amostras aleatórias do buffer para aprender."""
        # Salvar experiência / recompensa
        self.memory.add(state, action, reward, next_state, done)

        # Aprenda, se amostras suficientes estiverem disponíveis na memória
        # se memoria for maior que o tamanho do batch e o tempo de aprendizado for divisível pelo tempo de aprendizado
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
            # para cada vez que o agente aprende
            for _ in range(LEARN_NUM):
                # para cada atualização de aprendizagem
                experiences = self.memory.sample() 
                '''
                sample():
                retorna uma lista de tamanho particular de itens escolhidos na sequência, ou seja, lista, tupla, string ou conjunto
                '''
                # atualizar a rede neural
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Retorna ações para determinado estado de acordo com a política atual."""
        # converter o estado para tensor
        state = torch.from_numpy(state).float().to(device)
        # definir o modelo para avaliação
        self.actor_local.eval()
        # desativar o cálculo de gradiente
        with torch.no_grad():
            # obter ação do ator local
            action = self.actor_local(state).cpu().data.numpy()
        # definir o modelo para treinamento
        self.actor_local.train()
        # adicionar ruído
        if add_noise:
            # adicionar ruído + explorar
            action += self.epsilon * self.noise.sample()
        # retorna ação entre -1 e 1
        ''' 
        clip():
        retorna um novo array com os valores limitados entre um valor mínimo e máximo
        '''
        return np.clip(action, -1, 1)

    def reset(self):
        '''Reseta o processo de ruído'''
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Atualize a política e os parâmetros de valor usando determinado lote de tuplas de experiência.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        Onde:
            ator_alvo(estado) -> ação
            critic_target(state, action) -> Q-value
        Parâmetros
        ======
            experiências (Tuple[torch.Tensor]): tupla de (s, a, r, s', feito) tuplas
            gama (float): fator de desconto
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- atualizar o crítico ---------------------------- #
        # Obtenha ações de próximo estado previstas e valores Q de modelos de destino
        actions_next = self.actor_target(next_states)
        # obter valores Q de estados e ações de modelos de destino
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Calcula os alvos Q para os estados atuais (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Calcula a perda crítica
        Q_expected = self.critic_local(states, actions)
        # calcular a perda crítica
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimizar a perda
        self.critic_optimizer.zero_grad()
        # retropropagação
        critic_loss.backward()
        # atualizar os pesos com o otimizador
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) # clipe de gradiente para 1 norma (evite explosão de gradiente)
        # otimizar o crítico 
        self.critic_optimizer.step()

        # ---------------------------- atualizar o ator ---------------------------- #
        # Calcula a perda do ator
        actions_pred = self.actor_local(states)
        # calcular a perda do ator
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimizar a perda
        self.actor_optimizer.zero_grad()
        # retropropagação
        actor_loss.backward()
        # atualizar os pesos com o otimizador
        self.actor_optimizer.step()

        # ----------------------- atualizar redes de destino ----------------------- #
        # atualizar pesos do ator alvo
        self.soft_update(self.critic_local, self.critic_target, TAU)
        # atualizar pesos do crítico alvo
        self.soft_update(self.actor_local, self.actor_target, TAU) 
        
        # ---------------------------- ruído de atualização ------------------------ #
        # decrementar a exploração
        self.epsilon -= EPSILON_DECAY
        # resetar o ruído
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Parâmetros do modelo de atualização suave.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Parâmetros
        ======
            local_model: modelo PyTorch (pesos serão copiados)
            target_model: modelo PyTorch (os pesos serão copiados para)
            tau (float): parâmetro de interpolação
        """
        # iterar sobre os parâmetros do modelo local e do modelo alvo
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # atualizar os pesos do modelo alvo com a interpolação
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)