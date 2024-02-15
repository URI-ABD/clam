import inspect

import abd_distances


def main():
    # Use the `inspect` module to print the members of the `abd_distances` module
    list(map(print, inspect.getmembers(abd_distances.vectors)))

    print(abd_distances.vectors.euclidean_f32([1, 2, 3], [4, 5, 6]))


if __name__ == "__main__":
    main()
