# CLAM: Contributor Guidelines

Pull requests and bug reports are welcome.
For major changes, please open an issue to discuss what you would like to change.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Prerequisites

- [`docker`](https://docs.docker.com/engine/install/)
- [`hermit`](https://cashapp.github.io/hermit/usage/get-started/)

## Getting Started

1. Fork the repository to your own GitHub account. You should make changes in your own fork and contribute back to the base repository (under URI-ABD) via pull requests.
2. Clone the repo from your fork.
   1. `git clone ...`
3. Initialize submodules.
   1. `git submodule update --init`
4. Test that things work.
   1. `cargo test --release`
5. Install pre-commit hooks
   1. `pre-commit install`
6. Make a new branch.
   1. Make sure to branch from the head of the `master` branch.
   2. Have a plan and scope in mind for your changes.
   3. You may not have merge commits in your branch because we wish to keep a linear history on the `master` branch. Use `git rebase` to keep your branch up-to-date with the `master` branch.
7. Make your changes.
   1. Remember to add tests and documentation.
8. Bump the version.
   1. If you need help with this step, please ask.
9. Commit and push your changes.
10. Open a pull request.

### Python-Specific Work

1. Install [`poetry`](https://python-poetry.org/docs/#installation).
   1. `python -m pip install poetry`
2. Install the project dependencies, using `poetry`.
   - Use poetry to install the project dependencies: `poetry install`
