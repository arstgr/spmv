import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import numpy as np

# File name
#file_name = "HV15R/HV15R.mtx"  # Replace with your file name
file_name = "PR02R/PR02R.mtx"  # Replace with your file name
output_file = "PR02R_visualization.png"  # Output image file

# Read file and construct sparse matrix
rows, cols, values = [], [], []

with open(file_name, 'r') as f:
    for line in f:
        if line.startswith('%'):  # Skip comment lines
            continue
        parts = line.split()
        rows.append(int(parts[0]))  # X index
        cols.append(int(parts[1]))  # Y index
        values.append(float(parts[2]))  # Value

# Determine matrix dimensions
n_rows = max(rows) + 1
n_cols = max(cols) + 1

# Create a sparse matrix
matrix = coo_matrix((values, (rows, cols)), shape=(n_rows, n_cols))

min_val = np.min(matrix.data)
max_val = np.max(matrix.data)
normalized_values = (matrix.data - min_val) / (max_val - min_val)

# Visualize and save the plot
plt.figure(figsize=(10, 10))
plt.spy(matrix, markersize=1)
plt.title("Sparse Matrix Visualization")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save the plot
plt.close()  # Close the plot to free memory


#plt.figure(figsize=(12, 12))
#plt.scatter(matrix.col, matrix.row, c=normalized_values, cmap='viridis', s=1)  # Use color to represent values
#plt.colorbar(label='Normalized Values')
#plt.title("Sparse Matrix Visualization")
#plt.xlabel("Columns")
#plt.ylabel("Rows")
#plt.gca().invert_yaxis()  # Match matrix row order in the visualization
#plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Save the plot as a PNG
#plt.close()  # Close the plot to free memory

print(f"Visualization saved to {output_file}")

