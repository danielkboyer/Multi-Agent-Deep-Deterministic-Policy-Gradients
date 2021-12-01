import matplotlib.pyplot as plt

plt.plot([0, 1, 2, 3, 4], [0, 3, 5, 9, 11])

def save_avg_graph(episodes,averages):
    plt.plot(episodes,averages)
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward')
    plt.savefig(f'Images/Episode{episodes[len(episodes)-1]}.png')