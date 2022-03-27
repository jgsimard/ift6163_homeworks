import glob
import numpy as np

import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt



# remove the tensorflow warnings, I dont care
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False



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


def get_section_results2(file):
    steps = []
    eval_avg_return,  eval_std_return = [], []
    train_avg_return, train_std_return = [], []
    # print([e for e in tf.train.summary_iterator(file)])
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                steps.append(v.simple_value)

            if v.tag == 'Eval_AverageReturn':
                eval_avg_return.append(v.simple_value)

            if v.tag == 'Eval_StdReturn':
                eval_std_return.append(v.simple_value)

    return np.array(steps), np.array(eval_avg_return), np.array(eval_std_return)



def pad(base, x):
    return np.pad(x, (len(base) - len(x), 0), 'constant', constant_values=(-np.infty, 0))

def q1(eventfiles):
    eventfile = [e for e in eventfiles if "q1" in e][0]
    print(eventfile)

    steps, train_avg_return, train_max_return = get_section_results(eventfile)
    plt.plot(steps /1e6, pad(steps, train_avg_return))

    plt.title("Experiment 1")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Million steps")
    # plt.axhline(y=1500, linestyle='-', label='150')
    # plt.legend(loc="lower right")
    plt.savefig(f'q1.png')
    plt.show()


def get_stats(ef):
    steps, train_avg_return, train_max_return = [], [], []
    for eventfile in ef:
        s, r, b = get_section_results2(eventfile)
        steps.append(s)
        train_avg_return.append(pad(s, r))
        train_max_return.append(pad(s, b))
    steps = np.stack(steps) / 1e6
    train_avg_return = np.vstack(train_avg_return)
    train_max_return = np.vstack(train_max_return)
    return steps, train_avg_return, train_max_return

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
            train_avg_return.append(pad(s, r))
            train_max_return.append(pad(s, b))
        steps = np.stack(steps) / 1e6
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
    plt.ylabel("Average Episode Return")
    plt.xlabel("Million steps")
    plt.legend(loc="best")
    plt.grid()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -300, 150))
    plt.savefig(f'q2-median.png')
    plt.show()

    p(steps_dqn, train_max_return_dqn, 'dqn')
    p(steps_ddqn, train_max_return_ddqn, 'double dqn')

    plt.title("Experiment 2 - Best Reward")
    plt.ylabel("Best Episode Return")
    plt.xlabel("Million steps")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f'q2-best.png')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -150, 150))
    plt.show()


def q3(eventfiles):
    def get_stats(ef):
        steps, train_avg_return, train_max_return = [], [], []
        for eventfile in ef:
            s, r, b = get_section_results(eventfile)
            steps.append(s)
            train_avg_return.append(pad(s, r))
            train_max_return.append(pad(s, b))
        steps = np.stack(steps) / 1e6
        train_avg_return = np.vstack(train_avg_return)
        train_max_return = np.vstack(train_max_return)
        return steps, train_avg_return, train_max_return


    eventfiles = [e for e in eventfiles if "q3" in e]
    print(f"len(eventfiles)={len(eventfiles)}")
    # tested_lr = [0.0002, 0.0005, 0.001, 0.002]
    tested_lr = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
    for lr in tested_lr:
        eventfiles_lr = [e for e in eventfiles if f"q3_lr{lr}_seed" in e]
        print(f"len(eventfiles_lr)={len(eventfiles_lr)}, lr={lr}")
        steps, train_avg_return, train_max_return = get_stats(eventfiles_lr)
        steps = steps[0]
        print(steps.shape, train_avg_return.shape, train_max_return.shape)
        # print(train_avg_return)
        # plt.plot(steps, train_avg_return[0])
        # N = 10
        # steps, train_avg_return, train_max_return = steps[1::N], train_avg_return[:, 1::N], train_max_return[:, 1::N]
        def p(s, v, name):
            s = s * 1000
            plt.plot(s, np.median(v, 0), label=name)
            plt.fill_between(s, np.min(v, 0), np.max(v, 0), alpha=0.2)

        p(steps, train_avg_return, f"{lr}")

    plt.title("Experiment 3 - Mean Reward ")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Thousands steps")
    plt.legend(loc="best")
    plt.savefig(f'q3-mean.png')
    plt.show()

    for lr in tested_lr:
        eventfiles_lr = [e for e in eventfiles if f"q3_lr{lr}_seed" in e]
        print(f"len(eventfiles_lr)={len(eventfiles_lr)}, lr={lr}")
        steps, train_avg_return, train_max_return = get_stats(eventfiles_lr)
        steps = steps[0]
        print(steps.shape, train_avg_return.shape, train_max_return.shape)
        # print(train_avg_return)
        # plt.plot(steps, train_avg_return[0])
        # N = 10
        # steps, train_avg_return, train_max_return = steps[1::N], train_avg_return[:, 1::N], train_max_return[:, 1::N]
        def p(s, v, name):
            s = s * 1000
            plt.plot(s, np.median(v, 0), label=name)
            plt.fill_between(s, np.min(v, 0), np.max(v, 0), alpha=0.2)

        p(steps, train_max_return, f"{lr}")

    plt.title("Experiment 3 - Best Reward")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Thousands steps")
    plt.legend(loc="best")
    plt.savefig(f'q3-best.png')
    plt.show()

def q4(eventfiles):
    eventfiles = [e for e in eventfiles if "q4" in e]
    tested_lr = [0.0002, 0.0005, 0.001, 0.002]
    # tested_lr = [0.002]
    for lr in tested_lr:
        eventfiles_lr = [e for e in eventfiles if f"q4_ddpg_lr{lr}_seed" in e]
        # print(f"{lr} ================ {eventfiles}")
        steps, train_avg_return, train_max_return = get_stats(eventfiles_lr)
        steps = steps[0]
        N = 10
        steps, train_avg_return, train_max_return = steps[1::N], train_avg_return[:, 1::N], train_max_return[:, 1::N]
        def p(s, v, name):
            s = s * 1000
            plt.plot(s, np.median(v, 0), label=name)
            plt.fill_between(s, np.min(v, 0), np.max(v, 0), alpha=0.2)

        p(steps, train_avg_return, f"{lr}")

    plt.title("Experiment 4 - Learning Rate")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Thousands steps")
    plt.legend(loc="best")
    plt.savefig(f'q4-lr.png')
    plt.show()

    tested_up = [1, 2, 4]
    for up in tested_up:
        eventfiles_up = [e for e in eventfiles if f"q4_ddpg_up{up}_seed" in e]
        steps, train_avg_return, train_max_return = get_stats(eventfiles_up)
        steps = steps[0]
        if len(steps) > 200:
            N = 10
            steps, train_avg_return, train_max_return = steps[1::N], train_avg_return[:, 1::N], train_max_return[:, 1::N]

        def p(s, v, name):
            s = s * 1000
            plt.plot(s, np.median(v, 0), label=name)
            plt.fill_between(s, np.min(v, 0), np.max(v, 0), alpha=0.2)

        p(steps, train_avg_return, f"{up}")

    plt.title("Experiment 4 - Policy Update Rate")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Thousands steps")
    plt.legend(loc="best")
    plt.savefig(f'q4-up.png')
    plt.show()

def q5(eventfiles):
    def p(s, avg, std, name):
        s = s/1000
        plt.plot(s, avg, label=name)
        plt.fill_between(s, avg - std, avg + std, alpha=0.2)

    eventfile_ddpg = [e for e in eventfiles if "q5" in e][0]
    s, avg, std = get_section_results2(eventfile_ddpg)
    p(s, avg, std, 'ddpg')


    plt.title("Experiment 5 - DDPG")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Thousands steps")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f'q5.png')
    plt.show()

def q6(eventfiles):
    eventfiles = [e for e in eventfiles if "q6" in e]
    tested_rho = [0.1, 0.2, 0.5]
    for rho in tested_rho:
        eventfiles_rho = [e for e in eventfiles if f"q6_td3_rho{rho}_seed" in e]
        steps, train_avg_return, train_max_return = get_stats(eventfiles_rho)
        steps = steps[0]
        N = 10
        steps, train_avg_return, train_max_return = steps[1::N], train_avg_return[:, 1::N], train_max_return[:, 1::N]
        def p(s, v, name):
            s = s * 1000
            plt.plot(s, np.median(v, 0), label=name)
            plt.fill_between(s, np.min(v, 0), np.max(v, 0), alpha=0.2)

        p(steps, train_avg_return, f"{rho}")

    plt.title("Experiment 6 - Rho")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Thousands steps")
    plt.legend(loc="best")
    plt.savefig(f'q6-rho.png')
    plt.show()

    tested_size = [64, 128, 256]
    for size in tested_size:
        eventfiles_up = [e for e in eventfiles if f"q6_td3_size{size}_seed" in e]
        steps, train_avg_return, train_max_return = get_stats(eventfiles_up)
        steps = steps[0]
        if len(steps) > 200:
            N = 10
            steps, train_avg_return, train_max_return = steps[1::N], train_avg_return[:, 1::N], train_max_return[:, 1::N]

        def p(s, v, name):
            s = s * 1000
            plt.plot(s, np.median(v, 0), label=name)
            plt.fill_between(s, np.min(v, 0), np.max(v, 0), alpha=0.2)

        p(steps, train_avg_return, f"{size}")

    plt.title("Experiment 6 - Size")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Thousands steps")
    plt.legend(loc="best")
    plt.savefig(f'q6-size.png')
    plt.show()

def q7(eventfiles):
    def p(s, avg, std, name):
        s = s/1000
        plt.plot(s, avg, label=name)
        plt.fill_between(s, avg - std, avg + std, alpha=0.2)

    eventfile_ddpg = [e for e in eventfiles if "q5" in e][0]
    s, avg, std = get_section_results2(eventfile_ddpg)
    p(s, avg, std, 'ddpg')

    # eventfile_td3 = [e for e in eventfiles if "q7_td3" in e][0]
    # s, avg, std = get_section_results2(eventfile_td3)
    # p(s, avg, std, 'td3')

    eventfile_td3 = [e for e in eventfiles if "q7_2_td3" in e][0]
    s, avg, std = get_section_results2(eventfile_td3)
    p(s, avg, std, 'td3')

    plt.title("Experiment 7 - DDPG vs TD3")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Thousands steps")
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -150, y2))
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f'q7.png')
    plt.show()

def bonus(eventfiles):
    eventfiles_ = [e for e in eventfiles if "q2" in e]
    eventfiles_dqn = [e for e in eventfiles_ if "_dqn_" in e]
    eventfiles_doubledqn = [e for e in eventfiles_ if "_doubledqn_" in e]
    # print(eventfiles)
    def get_stats(ef):
        steps,  train_avg_return, train_max_return = [], [], []
        for eventfile in ef:
            s, r, b =get_section_results(eventfile)
            steps.append(s)
            train_avg_return.append(pad(s, r))
            train_max_return.append(pad(s, b))
        # print([s.shape for s in steps])
        steps = np.stack(steps) / 1e6
        train_avg_return = np.vstack(train_avg_return)
        train_max_return = np.vstack(train_max_return)
        print(steps.shape)
        return steps, train_avg_return, train_max_return

    eventfiles_ = [e for e in eventfiles if "bonus" in e]
    eventfiles_dqn_double_dueling = [e for e in eventfiles_ if "_dqn_double_dueling" in e]

    steps_dqn, train_avg_return_dqn, train_max_return_dqn = get_stats(eventfiles_dqn)
    steps_ddqn, train_avg_return_ddqn, train_max_return_ddqn = get_stats(eventfiles_doubledqn)
    steps_ddqn_dueling, train_avg_return_ddqn_dueling, train_max_return_ddqn_dueling = get_stats(eventfiles_dqn_double_dueling)


    def p(s, v, name):
        plt.plot(s[0], np.median(v, 0), label=name)
        plt.fill_between(s[0], v.min(0), v.max(0), alpha=0.2)


    p(steps_dqn, train_avg_return_dqn, 'dqn')
    p(steps_ddqn, train_avg_return_ddqn, '+double')
    p(steps_ddqn_dueling, train_avg_return_ddqn_dueling, '+double+dueling')

    plt.title("Experiment Bonus - Median Reward")
    plt.ylabel("Average Episode Return")
    plt.xlabel("Million steps")
    plt.legend(loc="best")
    plt.grid()
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -300, y2))
    plt.savefig(f'bonus-median.png')
    plt.show()

    p(steps_dqn, train_max_return_dqn, 'dqn')
    p(steps_ddqn, train_max_return_ddqn, 'double dqn')
    p(steps_ddqn_dueling, train_max_return_ddqn_dueling, '+double+dueling')

    plt.title("Experiment Bonus - Best Reward")
    plt.ylabel("Best Episode Return")
    plt.xlabel("Million steps")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(f'bonus-best.png')
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, -150, 150))
    plt.show()

if __name__ == '__main__':
    logdir = 'data/*/events*'
    eventfiles = glob.glob(logdir)
    # print(eventfiles)

    # q1(eventfiles)
    # q2(eventfiles)
    # q3(eventfiles)
    # q4(eventfiles)
    # q5(eventfiles)
    # q6(eventfiles)
    # q7(eventfiles)
    bonus(eventfiles)
