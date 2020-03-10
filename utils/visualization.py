import io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def reachability_plot(env, patients, reach_maps):
    single_env = isinstance(patients, (int, np.int64))
    envs = 1 if single_env else len(patients)
    goals = env.goals
    goal_rows = []
    goal_cols = []
    numerator = np.zeros([envs, env.num_rows * 2 + 1, env.num_cols * 2 + 1])
    denominator = np.zeros_like(numerator)

    plot_size = 5
    fig, ax = plt.subplots(1, 1, figsize=(plot_size, plot_size))
    for i in range(envs):
        goal = goals[patients] if single_env else goals[patients[i]]
        goal_row, goal_col = env.val_to_coords(goal[-1]) if isinstance(goal, list) else env.val_to_coords(goal)
        goal_rows.append(goal_row)
        goal_cols.append(goal_col)
        reach_map = reach_maps[i, :, :]
        x_shift = env.num_cols - goal_col
        y_shift = env.num_rows - goal_row
        numerator[i, :, :] = np.pad(reach_map, [[y_shift, env.num_rows - y_shift + 1], [x_shift, env.num_cols - x_shift + 1]], mode="constant")
        denominator[i, :, :] = np.pad(np.ones_like(reach_map), [[y_shift, env.num_rows - y_shift + 1], [x_shift, env.num_cols - x_shift + 1]], mode="constant")

    denominator = np.sum(denominator, axis=0)
    rows = np.any(denominator, axis=1)
    cols = np.any(denominator, axis=0)
    first_row, last_row = np.where(rows)[0][[0, -1]]
    first_col, last_col = np.where(cols)[0][[0, -1]]

    denominator += (denominator == 0) * 1
    im = ax.matshow(np.sum(numerator, axis=0)[first_row:last_row + 1, first_col:last_col + 1] / denominator[first_row:last_row + 1, first_col:last_col + 1], cmap='Greens')
    ax.scatter(np.max(goal_cols), np.max(goal_rows), marker='s', c='red', s=100)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    ax.set_title("Average reachability: {}".format(np.sum(reach_maps)/np.prod(reach_maps.shape)))

    return fig

def plot2fig(fig):
    """Create a pyplot plot and save to buffer."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    buf.close()
    return image
