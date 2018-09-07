import brica
import tensorflow as tf
import numpy as np
import random
import os

STEP_THRE = 10000
ESP_START = 0.7
ESP_END = 0.0
#--Env---------------------------------------------------------------------------
class Environment(object):
    def __init__(self, fef_data, action_space):
        self.fef_data = fef_data
        #env = gym.make(name)
        #self.env = env
        self.n_states = len(self.fef_data) * 3
        self.n_actions = action_space
        self.a_low_bound = 0.0
        self.a_high_bound = 1.0

    def reset(self):
        self.name = None
        self.n_states = None #len(self.name)
        self.n_actions = None #action_space
        self.a_low_bound = None #0.0
        self.a_high_bound = None #1.0
        
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) cartpole
        #self.steps_beyond_done = None
        #return np.array(self.state)
        return #self.env.reset()

    def step(self, a):
        return dict(to_pfc=None, to_fef=None, to_sc=a)
        #return np.array(self.state), reward, done, {}
        #return self.env.step(a)

#--Agent---------------------------------------------------------------------------
class Agent(object):
    def __init__(self, number, sess, gamma=0.9, max_epochs=100, max_ep_steps=100, params_update_iter=10):
        self.number = number
        self.name = 'agent_' + str(number)
        #self.env = env
        self.agent_brain = Brain(self.name, sess)  # create ACNet for each worker
        self.sess = sess
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.max_ep_steps = max_ep_steps
        self.params_update_iter = params_update_iter

    def _discounted_reward(self, v_, r):
        buffer_r = np.zeros_like(r)
        for i in reversed(range(len(r))):
            v_ = r[i] + v_ * self.gamma
            buffer_r[i] = v_
        return buffer_r

    def choose_action(self, s):
        action = self.agent_brain.choose_action(s)
        return action

    def predict_value(self, s):
        v = self.agent_brain.predict_value(s)
        return v

    def learn(self, buffer_s, buffer_a, buffer_v_target):
        buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
        self.agent_brain.update_global_params(buffer_s, buffer_a, buffer_v_target)  # actual training step, update global ACNet

    def work(self):
        buffer_s, buffer_a, buffer_r = [], [], []
        total_r = 0
        avg_epoch_r_hist = []
        #for epoch in range(self.max_epochs):
            #if epoch == self.max_epochs - 1:
            #    self.env.reset()
            #    self.env = gym.wrappers.Monitor(self.env, './A3C_continuous') # 動画保存
        s = self.env.fef_data#.reset()
        ep_step = 0
        #print('epoch:', epoch)
        while True:
            #if epoch == self.max_epochs - 1:
                #   self.env.render()
            a = self.choose_action(s)
            print('a:', a)
            inputs = self.env.step(a)
            #phase = inputs['from_pfc']
            
            s_ = inputs['from_fef']
            r, done = inputs['from_environment']
            
            done = True if ep_step == self.max_ep_steps - 1 else False

            total_r += r

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append((r + 8) / 8)

            if ep_step % self.params_update_iter == 0 or done:
                if done:
                    v_s_ = 0
                else:
                    v_s_ = self.predict_value(s_)

                discounted_r = self._discounted_reward(v_s_, buffer_r)

                self.learn(buffer_s, buffer_a, discounted_r)

                buffer_s, buffer_a, buffer_r = [], [], []
                self.agent_brain.update_local_params()

            s = s_
            ep_step += 1
            print('ep_step:', ep_step)
            if done:
                break
            
        #if self.number == 0:
        #avg_ep_r = total_r/(epoch+1)
        #avg_epoch_r_hist.append(avg_ep_r)
            #if epoch >= 50:
                #plt.cla()
                #plt.plot(avg_epoch_r_hist)
                #plt.pause(0.001)
        #plt.show()
        #plt.savefig('./A3C_continuous/figure.png')
        #plt.ioff()

#--Network for the Actor Critic----------------------------------------------------------------
class Brain(object):
    def __init__(self, scope, sess, action_scale=1.0, actor_lr=0.001, critic_lr=0.001):
        self.sess = sess
        #self.env = env
        self.low_action_bound = 0.0 #env.a_low_bound
        self.high_action_bound = 1.0 #env.a_high_bound
        self.n_states = 128 * 3 #env.n_states
        self.n_actions = 128 #env.n_actions
        self.action_scale = action_scale
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.entropy_beta = 0.01

        self.build_graph(scope)

    def build_graph(self, scope):
        self.s = tf.placeholder(tf.float32, [None, self.n_states], 's')
        self.a = tf.placeholder(tf.float32, [None, self.n_actions], 'a')
        self.q_target = tf.placeholder(tf.float32, [None, 1], 'q_target')
        if scope == 'global':
            self._build_net(scope=scope)
        else:
            mu, sigma, self.critic_net = self._build_net(scope=scope)

            # workerのparameterセット
            la_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
            lc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

            # globalのparameterセット
            ga_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global/actor')
            gc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='global/critic')

            with tf.name_scope('c_loss'):
                td = tf.subtract(self.q_target, self.critic_net, name='TD_error') # td = R(t) - V(s)
                self.critic_loss_op = tf.reduce_mean(tf.square(td))  # td**2

            with tf.name_scope('a_loss'):
                sigma = sigma + 1e-4
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                log_prob_action_adv = normal_dist.log_prob(self.a) * td  #log( π(a|s,θ) ) * td
                entropy = normal_dist.entropy()
                self.policy_loss = self.entropy_beta * entropy + log_prob_action_adv #おっきく間違わないためのエントロピー項
                self.policy_loss_op = tf.reduce_mean(-self.policy_loss)

            with tf.variable_scope('train'):
                self.actor_optimizer = tf.train.RMSPropOptimizer(self.actor_lr, name='RMSPropA')
                self.critic_optimizer = tf.train.RMSPropOptimizer(self.critic_lr, name='RMSPropC')

            with tf.name_scope('choose_a'):
                self.actor_net = tf.squeeze(normal_dist.sample(1), axis=0) # normal_distから要素を一つ取り出し1なら削除
                self.actor_net = tf.clip_by_value(self.actor_net, self.low_action_bound, self.high_action_bound) # actor_netの値の内上限下限を超えるものは上限下限になる

            with tf.name_scope('local_grad'):
                self.actor_grads = tf.gradients(self.policy_loss_op, la_params, name='actor_grads')
                self.critic_grads = tf.gradients(self.critic_loss_op, lc_params, name='critic_grads')

            with tf.name_scope('pull'):
                # workerにglobalのパラメータをコピー
                self.update_la_params_op = [la.assign(ga) for la, ga in zip(la_params, ga_params)]
                self.update_lc_params_op = [lc.assign(gc) for lc, gc in zip(lc_params, gc_params)]
            with tf.name_scope('push'):
                # globalにworkerのパラメータをコピー
                self.update_ga_params_op = self.actor_optimizer.apply_gradients(zip(self.actor_grads, ga_params))
                self.update_gc_params_op = self.critic_optimizer.apply_gradients(zip(self.critic_grads, gc_params))

    def _build_net(self, scope):
        with tf.variable_scope(scope):
            k_init, b_init = tf.random_normal_initializer(0.0, 0.1), tf.constant_initializer(0.1)

            #actor net 活性化関数relu6 tanh softplus
            with tf.variable_scope('actor'):
                actor_hidden = tf.layers.dense(inputs=self.s, units=128,
                                               activation=tf.nn.relu6,
                                               kernel_initializer=k_init,
                                               bias_initializer=b_init,
                                               name='actor_hidden')
                mu = tf.layers.dense(inputs=actor_hidden,
                                     units=self.n_actions,
                                     activation=tf.nn.tanh,
                                     kernel_initializer=k_init,
                                     bias_initializer=b_init,
                                     name='mu')
                sigma = tf.layers.dense(inputs=actor_hidden,
                                        units=self.n_actions,
                                        activation=tf.nn.softplus,
                                        kernel_initializer=k_init,
                                        bias_initializer=b_init,
                                        name='sigma')
                mu = self.action_scale*mu

            #critic net 活性化関数relu6 
            with tf.variable_scope('critic'):
                critic_hidden = tf.layers.dense(inputs=self.s, units=128,
                                                activation=tf.nn.relu6,
                                                kernel_initializer=k_init,
                                                bias_initializer=b_init,
                                                name='critic_hidden')
                critic_net = tf.layers.dense(inputs=critic_hidden,
                                             units=1,
                                             kernel_initializer=k_init,
                                             bias_initializer=b_init,
                                             name='critic_net')  # state value

        return mu, sigma, critic_net

    def update_global_params(self, s, a, dr):
        s = np.reshape(s,(10, self.n_states))
        feed_dict = {self.s: s, self.a: a, self.q_target: dr}
        self.sess.run([self.update_ga_params_op, self.update_gc_params_op], feed_dict)

    def update_local_params(self):
        self.sess.run([self.update_la_params_op, self.update_lc_params_op])

    def choose_action(self, s):
        #random いれる
        '''if steps > STEP_THRE:
            esp = ESP_END
        else:
            esp = ESP_START + steps * (ESP_END - ESP_START) / STEP_THRE
        
        if random.random() < esp:
            s = np.random.rand(self.n_states)
        '''
        s = np.reshape(s,(1, self.n_states)) #[np.newaxis, :]
        return self.sess.run(self.actor_net, {self.s: s})[0]

    def predict_value(self, s):
        s = np.reshape(s, (1, self.n_states)) # [np.newaxis, :]
        v = self.sess.run(self.critic_net, {self.s: s})[0, 0]
        return v

#--run-------------------------------------------------------------------------------------------------
def run(fef_data, action_space, reward, done):
    sess = tf.Session()
    #plt.ion()

    with tf.device("/cpu:0"):
        env = Environment(fef_data, action_space)
        Brain('global', sess, env=env)
        env = Environment(fef_data, action_space)
        worker = Agent(1, sess, env)

    sess.run(tf.global_variables_initializer())
    job = worker.work()
    
#--BG--------------------------------------------------------------------------------------------------
class BG(object):
    def __init__(self):
        self.timing = brica.Timing(5, 1, 0)
        #plt.ion()
        self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
        self.total_r = []
        self.avg_epoch_r_hist = []
        self.steps = 1
        self.a = []

        self.sess = tf.Session()
        with tf.device("/cpu:0"):
            #env = Environment(fef_data, action_space)
            Brain('global', self.sess)
            #env = Environment(fef_data, action_space)
            self.worker = Agent(1, self.sess)
            self.sess.run(tf.global_variables_initializer())
        
        
    def doaction(self, a):
        return dict(to_pfc=None, to_fef=None, to_sc=a)
    
    def __call__(self, inputs):
        if 'from_environment' not in inputs:
            raise Exception('BG did not recieve from Environment')
        if 'from_pfc' not in inputs:
            raise Exception('BG did not recieve from PFC')
        if 'from_fef' not in inputs:
            raise Exception('BG did not recieve from FEF')
        
        reward, done = inputs['from_environment']
        phase = inputs['from_pfc']
        fef_data = inputs['from_fef']
        action_space = len(fef_data)

        print('step:', self.steps)
        print('reward:', reward)
       
        #run(fef_data, action_space, reward, done)
        #return dict(to_pfc=None, to_fef=None, to_sc=None)
        
        if self.steps == 1:
            self.fef_data = fef_data
            self.n_states = len(self.fef_data) * 3
            self.n_actions = action_space
            self.a_low_bound = 0.0
            self.a_high_bound = 1.0

        s = fef_data

        if self.steps == 1:
            if self.steps > STEP_THRE:
                esp = ESP_END
            else:
                esp = ESP_START + self.steps * (ESP_END - ESP_START) / STEP_THRE
            
            if random.random() < esp:
                self.a = np.random.rand(self.n_actions)
            else:
                self.a = self.worker.choose_action(s)



        if self.steps % self.worker.params_update_iter == 0 :
            if self.steps > STEP_THRE:
                esp = ESP_END
            else:
                esp = ESP_START + self.steps * (ESP_END - ESP_START) / STEP_THRE
            
            if random.random() < esp:
                self.a = np.random.rand(self.n_actions)
            else:
                self.a = self.worker.choose_action(s)



        print('a:', self.a)

        self.total_r.append(reward)
        self.buffer_s.append(s)
        self.buffer_a.append(self.a)
        self.buffer_r.append((reward + 8) / 8)

        if self.steps % self.worker.params_update_iter == 0 or done:
            if done:
                v_s_ = 0
            else:
                #v_s_ = worker.predict_value(s_)
                v_s_ = self.worker.predict_value(s)

            discounted_r = self.worker._discounted_reward(v_s_, self.buffer_r)
            print('discounted_r:', discounted_r)
            self.worker.learn(self.buffer_s, self.buffer_a, discounted_r)
            
            self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
            self.worker.agent_brain.update_local_params()

        #s = s_
        #print('s:', s)
        if self.steps == 500:
            f = open('./list/reward', 'w')
            for i in range(self.steps):
                f.write(str(i)+", "+str(self.total_r[i])+"\n")
            f.close()
        self.steps += 1
        #if done:
        #    break
        do = self.doaction(self.a)
        return do
        #phase = inputs['from_pfc']
        
        #s_ = inputs['from_fef']
        #r, done = inputs['from_environment']
        
        #done = True if ep_step == self.max_ep_steps - 1 else False

        
        
        #if self.number == 0:
        #avg_ep_r = total_r/(epoch+1)
        #avg_epoch_r_hist.append(avg_ep_r)
            #if epoch >= 50:
                #plt.cla()
                #plt.plot(avg_epoch_r_hist)
                #plt.pause(0.001)
        #plt.show()
        #plt.savefig('./A3C_continuous/figure.png')
        #plt.ioff()




