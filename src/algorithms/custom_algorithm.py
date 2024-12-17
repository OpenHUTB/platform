@register_algorithm("custom")
class CustomAlgorithm(BaseAlgorithm):
    """自定义算法"""
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # 创建网络
        self.policy_net = self._create_policy_network(config)
        self.value_net = self._create_value_network(config)
        
        # 创建优化器
        self.policy_optimizer = self._create_optimizer(
            self.policy_net.parameters(),
            config['policy_optimizer']
        )
        self.value_optimizer = self._create_optimizer(
            self.value_net.parameters(),
            config['value_optimizer']
        )
        
        # 创建经验回放
        self.replay_buffer = ReplayBuffer(
            config['buffer_size'],
            config['batch_size']
        )
        
    def predict(self, state: Dict) -> np.ndarray:
        """预测动作"""
        with torch.no_grad():
            state_tensor = self._preprocess_state(state)
            action_dist = self.policy_net(state_tensor)
            action = action_dist.sample()
        return action.cpu().numpy()
        
    def update(self, batch: Dict) -> Dict:
        """更新模型"""
        # 处理数据
        states = self._preprocess_states(batch['states'])
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = self._preprocess_states(batch['next_states'])
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        
        # 计算值函数目标
        with torch.no_grad():
            next_values = self.value_net(next_states)
            value_targets = rewards + self.gamma * (1 - dones) * next_values
            
        # 更新值函数
        values = self.value_net(states)
        value_loss = F.mse_loss(values, value_targets)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 更新策略
        action_dists = self.policy_net(states)
        log_probs = action_dists.log_prob(actions)
        advantages = (value_targets - values).detach()
        
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'mean_value': values.mean().item(),
            'mean_advantage': advantages.mean().item()
        } 