# CLAM: Contributor Guidelines

Pull requests and bug reports are welcome.
For major changes, please open an issue to discuss what you would like to change.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Pull Requests

1. Fork the repository to your own GitHub account. You should make changes in your own fork and contribute back to the base repository (under URI-ABD) via pull requests.
2. Install [poetry](https://python-poetry.org/docs/#installation) on your system.
   - `curl -sSL https://install.python-poetry.org | python3 -`
3. Make a python virtual environment with python 3.9.x.
   - We recommend using [pyenv](https://github.com/pyenv/pyenv) to install the version of python you need: `pyenv install 3.9.16`
   - Use the python version you installed with pyenv to create a virtual environment: `~/.pyenv/verisons/3.9.16/bin/python -m venv .venv`
   - Activate the virtual environment: `source .venv/bin/activate`
   - Use poetry to install the project dependencies: `poetry install`
   - Install the pre-commit hooks: `pre-commit install`
4. Make a new branch.
   - Make sure to branch from the head of the `master` branch.
   - Have a plan and scope in mind for your changes.
   - You may not have merge commits in your branch because we wish to keep a linear history on the `master` branch. Use `git rebase` to keep your branch up-to-date with the `master` branch.
5. Make your changes.
   - Remember to add tests and documentation.
6. Bump the version number using `bump2version`.
   - `bump2version patch` for bug fixes, performance improvements and refactoring.
   - `bump2version minor` for feature additions and breaking changes.
   - `bump2version dev` for work-in-progress changes.
   - `bump2version release` when the work-in-progress changes are ready for release.
   - `bump2version --dry-run --verbose --allow-dirty release|minor|patch|dev` to see what will happen without actually changing anything.
7. Commit and push your changes.
   - Work through the pre-commit checks.
   - If you wish to save your work before the pre-commit checks are complete, use `git commit --no-verify`.
   - Try to use commit messages that are descriptive and follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
8. Open a pull request.
   - You must rebase your branch to the head of the `master` branch before opening a pull request.
   - Pre-commit checks and tests will run again on GitHub and must pass before someone will even review the PR.
   - Once a PR is approved, it will be merged into the `master` branch and will automatically trigger a release to `PyPI` and `crates.io`.
