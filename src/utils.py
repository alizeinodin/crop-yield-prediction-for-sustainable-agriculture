import matplotlib.pyplot as plt

def plot_actual_vs_pred(y_true, y_pred, title="Actual vs Predicted"):
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()
