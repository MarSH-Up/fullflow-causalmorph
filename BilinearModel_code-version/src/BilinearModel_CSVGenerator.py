import csv


def write_to_csv(Y, timestamps, filename="output.csv"):
    """
    Write the results of Y and timestamps to a CSV file.

    Parameters:
    - Y: A 2D numpy array of shape (2*nRegions, N)
    - timestamps: A 1D numpy array of shape (N,)
    - filename: The name of the CSV file to save the results to
    """

    # Check if the shapes are compatible
    if Y.shape[1] != len(timestamps):
        raise ValueError(
            "The number of columns in Y must match the length of timestamps."
        )

    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)

        # Write the header
        header = ["Timestamps"] + [f"Channel{i+1}" for i in range(Y.shape[0])]
        writer.writerow(header)

        # Write the data
        for j in range(Y.shape[1]):
            row = [timestamps[j]] + list(Y[:, j])
            writer.writerow(row)

    print(f"Data written to {filename}")
