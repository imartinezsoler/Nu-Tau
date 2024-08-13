import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('IC_tracks.csv', sep=", ")
plt.plot(df['gamma'], df['norm'], label='IceCube tracks')  # Change 'x' and 'y' to your actual column names

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Line Plot')

# Display a legend
plt.legend()

# Show the plot
plt.show()
