import numpy


def main():
    samples = 3
    features_file_name = "dummy_features.npy"
    targets_file_name = "dummy_targets.npy"

    # Generate and save features
    with open(features_file_name, "wb") as f:
        features = numpy.array([
            [float(i) for i in range(6)] for _ in range(samples)
        ]).reshape(samples, 6)
        numpy.save(f, features)

    # Generate and save targets
    with open(targets_file_name, "wb") as f:
        targets = numpy.array([3.14 for _ in range(samples)]).reshape(samples)
        numpy.save(f, targets)

    print("Generated targets and features successfully")
    print()

    # Load and print features
    with open(features_file_name, "rb") as f:
        print("features:\n", numpy.load(f))

    print()

    # Load and print targets
    with open(targets_file_name, "rb") as f:
        print("targets:\n", numpy.load(f))


if __name__ == "__main__":
    main()
