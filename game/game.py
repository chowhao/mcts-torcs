from config.config import *
from utils.game_util import *


class Game:
    def __init__(self, sess, avp_net, vsp_net):
        self.avp_net = avp_net
        self.vsp_net = vsp_net
        self.max_depth = FLAGS.max_depth
        self.action_num = FLAGS.action_num
        self.sess = sess
        self.saver = tf.train.Saver()
        # initialize the network
        self.sess.run(tf.global_variables_initializer())

    def get_game_ended(self, level, img):
        # not the leaf node
        if level != self.max_depth:
            return 0
        # is the leaf node, return the value of this leaf node
        # avp_net is default policy
        else:
            _, v = self.avp_net.predict(img)
            return v

    # predict the next state of this node
    def get_next_state(self, s, a):

        return self.vsp_net.predict(s, a)

    @staticmethod
    def format_steer_angle(w):
        w = int((w + 1) / 0.2)
        # ???
        # if w == 10:
        #     w = 9
        return w

    def train_net(self, train_examples):
        train_times = int(FLAGS.train_interval / FLAGS.vsp_batch_size)

        # train avp and vsp network
        for _ in range(train_times):

            sample_index = np.random.choice(FLAGS.train_interval - 1,
                                            FLAGS.avp_batch_size)

            state_batch = [] # current state of car
            next_state_batch = [] # next state of car
            a_batch = [] # the action of car

            obs_batch = [] # current state of car
            p_batch = [] # possibility of car action
            v_batch = [] # value of car action

            for i in range(FLAGS.avp_batch_size):
                # coach.last_state, coach.last_pi, r, coach.last_a
                obs_batch.append(train_examples[sample_index[i]][0])
                p_batch.append(train_examples[sample_index[i]][1])
                next_state = train_examples[sample_index[i] + 1][0]
                _, tmp_v = self.avp_net.predict(next_state)
                v_batch.append(train_examples[sample_index[i]][2]+FLAGS.gmma*tmp_v)

                state_batch.append(train_examples[sample_index[i]][0])
                tmp = np.zeros((1,))
                tmp[0] = train_examples[sample_index[i]][-1]
                a_batch.append(tmp)
                next_state_batch.append(train_examples[sample_index[i] + 1][0])

            loss_vsp = self.vsp_net.learn(state_batch, a_batch, next_state_batch)
            loss_avp = self.avp_net.learn(p_batch, v_batch, obs_batch)
            # print('loss_vsp:', round(loss_vsp, 5), 'loss_avp:', round(loss_avp, 5))
            print("loss_vsp: ", round(loss_vsp, 5), "loss_avp: ", round(loss_avp, 5))

    def save_ckpt(self):
        self.saver.save(self.sess, 'ckpt/model.ckpt')
