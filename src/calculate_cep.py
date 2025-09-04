import numpy as np
import csv

def calculate_cep(filepath: str) -> float:
    """
    Calculates the Circular Error Probable (CEP) for a dataset of 3D coordinates.
    
    The CEP is defined as the radius of a circle that contains 50% of the
    data points. This function calculates it by finding the median of the
    radial distances from the mean impact point.

    Args:
        filepath (str): The path to a CSV file containing columns 'x', 'y', 'z'.
                        The file must have a header row.

    Returns:
        float: The calculated CEP value. Returns 0.0 if there is an error
               reading the file or if the dataset is empty.
    """
    try:
        # Load data from the CSV file
        coords = []
        with open(filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                coords.append([
                    float(row['x']),
                    float(row['y']),
                    float(row['z'])
                ])
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return 0.0
    except KeyError:
        print(f"Error: CSV file must contain 'x', 'y', and 'z' columns.")
        return 0.0
    except (ValueError, TypeError) as e:
        print(f"Error parsing data: {e}. Ensure coordinates are numeric.")
        return 0.0

    if not coords:
        print("Warning: The dataset is empty. CEP cannot be calculated.")
        return 0.0

    coords_array = np.array(coords)

    # 1. Calculate the mean impact point
    mean_point = np.mean(coords_array, axis=0)

    # 2. Calculate the radial distances of all points from the mean
    radial_distances = np.linalg.norm(coords_array - mean_point, axis=1)

    # 3. Calculate the CEP as the median of the radial distances
    cep_value = np.median(radial_distances)
    
    return cep_value

if __name__ == '__main__':
    # This block provides a simple example of how to use the function.
    print("Running CEP calculation on sample data...")
    sample_file_path = "data/sample_data.csv"
    
    try:
        # Create a dummy data file for the example if it doesn't exist
        # In a real project, this would be handled externally
        with open(sample_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z'])
            writer.writerow([10.5, -5.2, 1.0])
            writer.writerow([12.1, -4.8, 0.9])
            writer.writerow([9.9, -6.1, 1.2])
            writer.writerow([11.8, -5.5, 0.8])
            writer.writerow([10.2, -5.8, 1.1])
        
        calculated_cep = calculate_cep(sample_file_path)
        print(f"Calculated CEP: {calculated_cep:.4f} units")
    except Exception as e:
        print(f"An error occurred during the example run: {e}")