import numpy as np
import csv
import os

def calculate_cep(filepath: str) -> float:
    """
    Calculates the Circular Error Probable (CEP) for a dataset of 3D coordinates.

    CEP in 3D represents the radius of a sphere that contains 50% of the data points.
    This function uses a constant derived from the chi-squared distribution.

    Args:
        filepath (str): The path to a CSV file containing columns 'x', 'y', 'z'.

    Returns:
        float: The calculated CEP value.

    Raises:
        FileNotFoundError: If the file at filepath is not found.
        KeyError: If the CSV file is missing 'x', 'y', or 'z' columns.
        ValueError: If the coordinates are not numeric or there are not enough data points.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")

    try:
        coords = []
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # Check for required columns
            if not all(col in reader.fieldnames for col in ['x', 'y', 'z']):
                raise KeyError("CSV file must have 'x', 'y', and 'z' columns.")

            for row in reader:
                coords.append([float(row['x']), float(row['y']), float(row['z'])])
    except (ValueError, KeyError) as e:
        raise e
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Ensure there are enough data points
    if len(coords) < 2:
        raise ValueError("At least two data points are required for calculation.")

    # Convert to a NumPy array for efficient calculations
    coords_np = np.array(coords)

    # Calculate the mean (centroid) of the data
    mean_coords = np.mean(coords_np, axis=0)

    # Calculate the variance for each axis
    variance = np.var(coords_np, axis=0)

    # The spherical standard deviation (sigma_s) is the square root of the sum of variances
    sigma_s = np.sqrt(np.sum(variance))

    # For a spherical normal distribution, 50% of the points fall within a radius
    # of approximately 1.5382 times the spherical standard deviation.
    # This constant is derived from the chi-squared distribution with 3 degrees of freedom.
    cep_value = 1.5382 * sigma_s

    return cep_value

if __name__ == "__main__":
    # --- Example Usage ---
    # To run this, you will need a file named 'data/sample_data.csv' in the data directory.
    # Create a sample CSV file for demonstration
    sample_data_path = 'data/sample_data.csv'
    os.makedirs('data', exist_ok=True)
    sample_data_content = [
        ['x', 'y', 'z'],
        ['1.5', '2.1', '0.5'],
        ['-0.8', '-1.0', '1.2'],
        ['2.1', '1.8', '-2.5'],
        ['0.3', '-2.5', '0.0'],
        ['-1.5', '0.9', '-1.5'],
    ]

    with open(sample_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sample_data_content)
    
    print("Sample 'data/sample_data.csv' created.")
    
    try:
        # Calculate CEP from the sample data
        cep_result = calculate_cep(sample_data_path)
        print(f"Calculated 3D CEP: {cep_result:.4f}")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
