#Code written by AI to help me understand it better
import matplotlib.pyplot as plt
import numpy as np


# 1. Define the mathematical log loss functions
def log_loss_y1(p):
    return -np.log(p)


def log_loss_y0(p):
    return -np.log(1 - p)


# 2. Generate 100 evenly spaced probabilities between 0.01 and 0.99
# We avoid exactly 0 and 1 because log(0) explodes to infinity
probs = np.linspace(0.01, 0.99, 100)

# 3. Compute the loss values using our functions
loss_if_actual_is_1 = log_loss_y1(probs)
loss_if_actual_is_0 = log_loss_y0(probs)

# 4. Create the plot using matplotlib
plt.figure(figsize=(9, 5))

# Plot both lines
plt.plot(
    probs,
    loss_if_actual_is_1,
    label="True Label $y = 1$ (Loss = $-\log(p)$)",
    color="darkgreen",
    linewidth=2.5,
)
plt.plot(
    probs,
    loss_if_actual_is_0,
    label="True Label $y = 0$ (Loss = $-\log(1-p)$)",
    color="crimson",
    linewidth=2.5,
)

# Style and annotate the graph
plt.title("Visualizing Binary Log Loss Penalties", fontsize=14, pad=15)
plt.xlabel("Model's Predicted Probability ($p$)", fontsize=11)
plt.ylabel("Log Loss Penalty", fontsize=11)
plt.xlim(0, 1)
plt.ylim(0, 5)  # Constrain y-axis since loss climbs towards infinity near the edges
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11, loc="upper center")

# Display the final graph
plt.show()
