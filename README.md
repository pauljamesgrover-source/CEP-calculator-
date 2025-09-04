# CEP-calculator-
CEP Calculator: Calculates Circular Error Probable for 2D coordinate data, displaying mean, standard deviation, and CEP value.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CEP Calculator</title>
    <!-- Use Tailwind CSS for modern styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">

    <!-- Main application container -->
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-lg w-full text-center">
        <h1 class="text-3xl sm:text-4xl font-bold text-gray-800 mb-4">
            Circular Error Probable (CEP) Calculator
        </h1>
        <p class="text-gray-600 mb-6">
            Enter a list of impact coordinates (X, Z) to calculate the CEP.
            The values should be comma-separated, with each pair on a new line.
        </p>
        
        <!-- Coordinates input area -->
        <div class="mb-6">
            <textarea
                id="coordinates-input"
                class="w-full h-40 p-4 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-blue-500 transition-colors"
                placeholder="Example:&#10;20010.5, -5.2&#10;19998.1, 2.7&#10;20005.3, 1.1"
            ></textarea>
        </div>

        <!-- Calculation button -->
        <button
            id="calculate-button"
            class="w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-full hover:bg-blue-700 transition-transform transform hover:scale-105 active:scale-95 shadow-lg"
        >
            Calculate CEP
        </button>

        <!-- Result display area -->
        <div id="cep-output" class="mt-8 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <p class="text-gray-800 font-medium">CEP will be displayed here.</p>
        </div>
    </div>

    <!-- JavaScript for calculation logic -->
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // Get references to all necessary DOM elements
            const calculateButton = document.getElementById('calculate-button');
            const coordinatesInput = document.getElementById('coordinates-input');
            const cepOutput = document.getElementById('cep-output');

            // Add a click event listener to the button
            calculateButton.addEventListener('click', () => {
                // Get the raw text input from the textarea
                const rawCoordinates = coordinatesInput.value;

                // Split the input into an array of lines
                const lines = rawCoordinates.split('\n').filter(line => line.trim() !== '');
                
                // If there are less than 2 points, we can't calculate CEP
                if (lines.length < 2) {
                    cepOutput.innerHTML = '<p class="text-red-500">Please enter at least two coordinate pairs.</p>';
                    return;
                }

                // Process each line to get a list of [x, z] coordinate pairs
                const points = lines.map(line => {
                    const parts = line.split(',').map(part => parseFloat(part.trim()));
                    // Basic validation to ensure the line has two valid numbers
                    if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
                        return { x: parts[0], z: parts[1] };
                    }
                    return null;
                }).filter(point => point !== null); // Remove any invalid lines

                // Check if any valid points were found
                if (points.length < 2) {
                    cepOutput.innerHTML = '<p class="text-red-500">Invalid coordinate format. Please use "x,z".</p>';
                    return;
                }

                // Calculate the centroid (mean of all x and z values)
                const sumX = points.reduce((acc, point) => acc + point.x, 0);
                const sumZ = points.reduce((acc, point) => acc + point.z, 0);
                const centroidX = sumX / points.length;
                const centroidZ = sumZ / points.length;

                // Calculate the radial distance of each point from the centroid
                const radialErrors = points.map(point => {
                    const dx = point.x - centroidX;
                    const dz = point.z - centroidZ;
                    return Math.sqrt(dx * dx + dz * dz);
                });

                // Sort the radial errors to find the median (50th percentile)
                radialErrors.sort((a, b) => a - b);
                
                // Calculate the CEP (median of radial errors)
                const mid = Math.floor(radialErrors.length / 2);
                let cep;
                if (radialErrors.length % 2 === 0) {
                    // For an even number of points, take the average of the two middle values
                    cep = (radialErrors[mid - 1] + radialErrors[mid]) / 2;
                } else {
                    // For an odd number, take the single middle value
                    cep = radialErrors[mid];
                }

                // Display the final calculated CEP with two decimal places
                cepOutput.innerHTML = `
                    <p class="text-gray-800 font-semibold text-lg">
                        Calculated CEP: <span class="text-blue-600 font-bold">${cep.toFixed(2)} m</span>
                    </p>
                    <p class="text-sm text-gray-500 mt-2">
                        Based on ${points.length} valid impact points.
                    </p>
                `;
            });
        });
    </script>
</body>
</html>
