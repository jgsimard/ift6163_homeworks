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
            elif v.tag == 'Eval_AverageReturn':
                eval_avg_return.append(v.simple_value)
            elif v.tag == 'Eval_StdReturn':
                eval_std_return.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                train_avg_return.append(v.simple_value)
            elif v.tag == 'Train_StdReturn':
                train_std_return.append(v.simple_value)
    return np.array(steps), np.array(eval_avg_return), np.array(eval_std_return), np.array(train_avg_return), np.array(train_std_return)


def plot_train(eventfile, name, step_size=5, color_dict=None):
    steps, eval_avg_return, eval_std_return, train_avg_return, train_std_return = get_section_results(eventfile)
    batch_steps = np.arange(0, len(eval_avg_return)) * step_size
    if color_dict is None:
        plt.plot(batch_steps, eval_avg_return, label=name)
        plt.fill_between(batch_steps, eval_avg_return - eval_std_return, eval_avg_return + eval_std_return, alpha=0.2)
    else:
        print(name)
        color = color_dict[name]
        plt.plot(batch_steps, eval_avg_return, label=name, color=color)
        plt.fill_between(batch_steps, eval_avg_return - eval_std_return, eval_avg_return + eval_std_return, alpha=0.2,
                         color=color)


def q1(eventfiles):
    eventfiles = [e for e in eventfiles if "q1" in e]
    eventfiles_lb = [e for e in eventfiles if "lb" in e]
    eventfiles_sb = [e for e in eventfiles if "sb" in e]
    print(eventfiles)
    color_dict = {'rtg_dsa': 'b', 'rtg_na': 'r', 'no_rtg_dsa': 'g'}
    for eventfile in eventfiles_lb :
        name = eventfile.split('/')[1]
        exp_name = name[10:][:-32]
        plot_train(eventfile, exp_name, color_dict=color_dict)

    plt.title("Experiment 1 : Large Batch")
    plt.legend(loc="lower right")
    plt.savefig(f'q1_large.png')
    plt.show()

    for eventfile in eventfiles_sb :
        name = eventfile.split('/')[1]
        exp_name = name[10:][:-32]
        plot_train(eventfile, exp_name, color_dict=color_dict)

    plt.title("Experiment 1 : Small Batch")
    plt.legend(loc="lower right")
    plt.savefig(f'q1_small.png')
    plt.show()


def q2(eventfiles):
    eventfiles = [e for e in eventfiles if "q2" in e]
    print(eventfiles)
    for eventfile in eventfiles:
        name = eventfile.split('/')[1]
        exp_name = name[7:][:-40]
        print(name, exp_name)
        step_size = 2 if "b1000_lr0.01" in eventfile else 5
        plot_train(eventfile, exp_name, step_size=step_size)

    plt.title("Experiment 2")
    plt.legend(loc="best")
    plt.savefig(f'q2.png')
    plt.show()


def q3(eventfiles):
    eventfiles = [e for e in eventfiles if "q3" in e]
    print(eventfiles)
    for eventfile in eventfiles :
        name = eventfile.split('/')[1]
        exp_name = name.split('_')[4]
        print(name, exp_name)
        plot_train(eventfile, exp_name)

    plt.title("Experiment 3")
    plt.axhline(y=180, linestyle='-', label='180')
    plt.legend(loc="lower right")
    plt.savefig(f'q3.png')
    plt.show()


def q4(eventfiles):
    eventfiles = [e for e in eventfiles if "q4" in e]
    color_dict = {'0.005': 'b', '0.01': 'r', '0.02': 'g'}
    for bs in ["10000", "30000", "50000"]:
        eventfiles_b = [e for e in eventfiles if f"b{bs}" in e and "search" in e]
        print(eventfiles)
        for eventfile in eventfiles_b :
            name = eventfile.split('/')[1]
            exp_name = name.split('_')[4][2:]
            print(name, exp_name)
            plot_train(eventfile, exp_name, color_dict=color_dict)

        plt.title(f"Experiment 4 : Batch Size = {bs}")
        plt.legend(loc="lower right")
        plt.savefig(f'q4_{bs}.png')
        plt.show()

    eventfiles_ = [e for e in eventfiles if "q4_b50000" in e]
    print(eventfiles)
    for eventfile in eventfiles_:
        name = eventfile.split('/')[1]
        exp_name = name[21:][:-35]
        print(name, exp_name)
        plot_train(eventfile, exp_name)
    e = [e for e in eventfiles if "q4_search_b50000_lr0.02" in e][0]
    name = e.split('/')[1]
    exp_name = name[28:][:-35]
    print(name, exp_name)
    plot_train(e, exp_name)

    plt.title(f"Experiment 4, Batch Size = 50000, Learning Rate = 0.02")
    plt.legend(loc="lower right")
    plt.savefig(f'q4_opt.png')
    plt.show()


def q5(eventfiles):
    eventfiles = [e for e in eventfiles if "q5" in e]
    print(eventfiles)
    for eventfile in eventfiles :
        name = eventfile.split('/')[1]
        exp_name = name.split('_')[4][6:]
        print(name, exp_name)
        plot_train(eventfile, exp_name)

    plt.title("Experiment 5")
    plt.legend(loc="lower right")
    plt.savefig(f'q5.png')
    plt.show()


def q6(eventfiles):
    eventfiles = [e for e in eventfiles if "q6" in e]
    print(eventfiles)
    for eventfile in eventfiles :
        name = eventfile.split('/')[1]
        split = name.split('_')
        ntu = split[2]
        ngsptu = split[3]
        exp_name = f"ntu={ntu}, ngsptu={ngsptu}"
        print(name, exp_name)
        plot_train(eventfile, exp_name)

    plt.title("Experiment 6")
    plt.legend(loc="lower right")
    plt.savefig(f'q6.png')
    plt.show()


def q7(eventfiles):
    eventfiles = [e for e in eventfiles if "q7" in e]
    print(eventfiles)
    for eventfile in eventfiles :
        name = eventfile.split('/')[1]
        exp_name = name.split('_')[2]
        print(name, exp_name)
        plot_train(eventfile, exp_name)

    plt.title("Experiment 7")
    plt.axhline(y=150, linestyle='-', label='150')
    plt.legend(loc="lower right")
    plt.savefig(f'q7.png')
    plt.show()

def q8(eventfiles):
    eventfiles = [e for e in eventfiles if "q8" in e]
    print(eventfiles)
    for eventfile in eventfiles :
        name = eventfile.split('/')[1]
        exp_name = f"{name.split('_')[2]}_{name.split('_')[3]}"
        print(name, exp_name)
        plot_train(eventfile, exp_name)

    plt.title("Experiment 8")
    plt.legend(loc="lower right")
    plt.savefig(f'q8.png')
    plt.show()


if __name__ == '__main__':
    logdir = 'run logs/*/events*'
    eventfiles = glob.glob(logdir)
    # print(eventfiles)

    q1(eventfiles)
    q2(eventfiles)
    q3(eventfiles)
    q4(eventfiles)
    q5(eventfiles)
    q6(eventfiles)
    q7(eventfiles)
    q8(eventfiles)
