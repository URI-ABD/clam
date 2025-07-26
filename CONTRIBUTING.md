# CLAM: Contributor Guidelines

Pull requests and bug reports are welcome.
For major changes, please open an issue to discuss what you would like to change.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Prerequisites

- `rust`: Use the shell script from [`rustup`](https://www.rust-lang.org/tools/install).
- `uv`: Use the shell script from [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

## Getting Started

1. Fork the repository to your own GitHub account. You should make changes in your own fork and contribute back to the base repository (under URI-ABD) via pull requests.
2. Clone the repo from your fork to your local machine:
   1. `git clone <insert-link-here>`
3. Build the workspace:
   1. `cargo build --release --workspace`
   2. `uv sync --all-packages`
4. Run tests to make sure that things are working:
   1. `cargo test --release --workspace`
   2. `uv run pytest`
5. Install and pre-commit hooks
   1. `uv run pre-commit install`
   2. `uv run pre-commit run --all-files`
6. Make a new branch.
   1. Have a plan and scope in mind for your changes.
   2. Make sure to branch from the head of the `master` branch.
   3. You **may not have merge commits** in your branch because we wish to keep a linear history on the `master` branch. Use `git rebase` to keep your branch up-to-date with the `master` branch.
7. Make your changes.
   1. Remember to add tests and documentation.
8. Commit and push your changes.
9. Open a pull request.
10. Wait for a review.
11. ???
12. Profit.
