import glob
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    steps = []
    eval_avg_return,  eval_std_return = [], []
    train_avg_return, train_std_return = [], []
    # print([e for e in tf.train.summary_iterator(file)])
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                steps.append(v.simple_value)

            if v.tag == 'Train_AverageReturn':
                train_avg_return.append(v.simple_value)
            # else:
            #     train_avg_return.append(0)

            if v.tag == 'Train_BestReturn':
                train_std_return.append(v.simple_value)
            # else:
            #     train_std_return.append(0)
    return np.array(steps), np.array(train_avg_return), np.array(train_std_return)


# def plot_train(eventfile, name, step_size=5, color_dict=None):
#     steps, eval_avg_return, eval_std_return, train_avg_return, train_std_return = get_section_results(eventfile)
#     batch_steps = np.arange(0, len(eval_avg_return)) * step_size
#     if color_dict is None:
#         plt.plot(batch_steps, eval_avg_return, label=name)
#         plt.fill_between(batch_steps, eval_avg_return - eval_std_return, eval_avg_return + eval_std_return, alpha=0.2)
#     else:
#         print(name)
#         color = color_dict[name]
#         plt.plot(batch_steps, eval_avg_return, label=name, color=color)
#         plt.fill_between(batch_steps, eval_avg_return - eval_std_return, eval_avg_return + eval_std_return, alpha=0.2,
#                          color=color)



def q2(eventfiles):
    eventfiles = [e for e in eventfiles if "q2" in e]
    eventfiles_dqn = [e for e in eventfiles if "_dqn_" in e]
    eventfiles_doubledqn = [e for e in eventfiles if "_doubledqn_" in e]
    print(eventfiles)
    def get_stats(ef):
        steps,  train_avg_return, train_max_return = [], [], []
        for eventfile in ef:
            s, r, b =get_section_results(eventfile)
            steps.append(s)
            train_avg_return.append(np.pad(r, (len(s) - len(r), 0), 'constant', constant_values=(-np.infty, 0)))
            train_max_return.append(np.pad(b, (len(s) - len(b), 0), 'constant', constant_values=(-np.infty, 0)))
        steps = np.stack(steps) / 1000
        train_avg_return = np.vstack(train_avg_return)
        train_max_return = np.vstack(train_max_return)
        return steps, train_avg_return, train_max_return

    steps_dqn, train_avg_return_dqn, train_max_return_dqn = get_stats(eventfiles_dqn)
    steps_ddqn, train_avg_return_ddqn, train_max_return_ddqn = get_stats(eventfiles_doubledqn)
    def p(s, v, name):
        plt.plot(s[0], np.median(v, 0), label=name)
        plt.fill_between(s[0], v.min(0), v.max(0), alpha=0.2)


    p(steps_dqn, train_avg_return_dqn, 'dqn')
    p(steps_ddqn, train_avg_return_ddqn, 'double dqn')

    plt.title("Experiment 2 - Median Reward")
    plt.legend(loc="best")
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -300, 150))
    plt.savefig(f'q2-median.png')
    plt.show()

    p(steps_dqn, train_max_return_dqn, 'dqn')
    p(steps_ddqn, train_max_return_ddqn, 'double dqn')

    plt.title("Experiment 2 - Best Reward")
    plt.legend(loc="best")
    plt.savefig(f'q2-best.png')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -150, 150))
    plt.show()




if __name__ == '__main__':
    logdir = 'data/*/events*'
    eventfiles = glob.glob(logdir)
    # print(eventfiles)

    # q1(eventfiles)
    q2(eventfiles)
    # q3(eventfiles)
    # q4(eventfiles)
    # q5(eventfiles)
    # q6(eventfiles)
    # q7(eventfiles)
    # q8(eventfiles)
