# CLAM: Contributor Guidelines

Pull requests and bug reports are welcome.
For major changes, please open an issue to discuss what you would like to change.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Prerequisites

- [`rust`](https://www.rust-lang.org/tools/install)
- [`docker`](https://docs.docker.com/engine/install/)
  - You will need this to test the CI/CD pipelines locally or to use Earthly to run various project commands.
- [`hermit`](https://cashapp.github.io/hermit/usage/get-started/)
  - This tool provides binaries you may want to have at hand to work on this repo.
  - You can see the full list of tools in the `./bin` directory in the repo root.
  - If you do not want to use hermit, you can install each tool manually.
  - If you wish to use some tools from hermit, you may execute them directly with `./bin/<tool>`.
  - (Recommended) If you wish to use all tools provided by earthly, we recommend installing the shell hooks (see [here](https://docs.earthly.dev/guides/shell-hooks) for more information).

## Getting Started

1. Fork the repository to your own GitHub account. You should make changes in your own fork and contribute back to the base repository (under URI-ABD) via pull requests.
2. Clone the repo from your fork.
   1. `git clone ...`
3. Test that things work.
   1. `cargo test --release`
   2. `earthly +test`
4. Install pre-commit hooks
   1. `pre-commit install`
   2. `pre-commit run --all-files`
5. Make a new branch.
   1. Make sure to branch from the head of the `master` branch.
   2. Have a plan and scope in mind for your changes.
   3. You may not have merge commits in your branch because we wish to keep a linear history on the `master` branch. Use `git rebase` to keep your branch up-to-date with the `master` branch.
6. Make your changes.
   1. Remember to add tests and documentation.
7. Bump the version.
   1. If you need help with this step, please ask.
8.  Commit and push your changes.
9.  Open a pull request.

### Python-Specific Work

1. Install [`poetry`](https://python-poetry.org/docs/#installation).
   1. `python -m pip install poetry`
2. Install the project dependencies, using `poetry`.
   - Use poetry to install the project dependencies: `poetry install`
