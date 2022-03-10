from matplotlib import pyplot as plt

def plot_model_chart(H, output_file, metrics=["accuracy", "val_accuracy"]):
    plt.figure()
    axis = [plt.subplot(2, 3, idx) for idx in range(1, len(metrics) + 1)]

    for i, m in enumerate(metrics):
        axis[i].set_title(m)
        axis[i].plot(range(1, len(H.history[m]) + 1), H.history[m])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc="lower left")

    plt.savefig(output_file)
    plt.close()
