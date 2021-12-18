import matplotlib.pyplot as plt


def save_avg_graph(episodes,good_averages,bad_averages):
    plt.clf()
    plt.plot(episodes,good_averages, label="Good")
    plt.plot(episodes,bad_averages, label="Bad")
    plt.title("Good and Bad Rewards")
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward')
    plt.legend(loc='best')
    plt.savefig(f'Images/Episode{episodes[len(episodes)-1]}.png')