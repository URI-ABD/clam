# Contribution guidelines

Contributions are welcome, encouraged, and appreciated!

## Reporting bugs

If you found a bug, please report it as an issue.

## Suggesting enhancements

If you have an idea how to improve the project, please open an issue.
After discussion, you can submit a pull request.

## Submitting pull requests

1. Fork the repository to your own GitHub account.
  - You should make changes in your own fork and contribute back to the base repository (under URI-ABD) via pull requests.
  - You should keep your fork in sync with the base repository. See [here](https://help.github.com/articles/syncing-a-fork/) for instructions.
  - Optionally, enable [GitHub actions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository) for your fork to run tests automatically.
2. Set up the `pre-commit` hooks.
  - Install [`pre-commit`](https://pre-commit.com/#install).
  - Run `pre-commit install` in the repository.
  - These hooks will run automatically when you commit changes.
  - You can run the hooks manually with `pre-commit run --all-files`.
  - These are in place to ensure a basic level of code quality.
3. Create a new branch for your changes (remember to checkout your branch).
  - The branch name should be descriptive of the changes.
  - The branch name should start with the issue number, if applicable.
4. Make your changes.
  - Make sure to add tests and documentation for your changes.
  - Run `cargo test --release` to make sure all tests pass.
5. Update the version number in the project.
  - Use [bump2version](https://github.com/c4urself/bump2version) for keeping the version number in sync across the project.
  - The version number should follow [Semantic Versioning](https://semver.org/).
6. Commit and push your changes.
  - Try to use descriptive commit messages.
  - See [here](https://www.conventionalcommits.org/en/v1.0.0/) for a specification for commit messages.
  - If you use the conventional commit format, we can automatically generate a changelog.
  - See [here](.hooks/commit-msg), under `TYPES`, for our commit message formats.
7. Open a pull request and assign a reviewer.
  - The pull request title should be descriptive of the changes.
  - The pull request description should:
    - reference the issue(s) it fixes.
    - include a summary of the changes.
8. If the GitHub actions pass, your reviewer will look at your changes.
  - If the reviewer requests changes, make the changes, push them to your branch, and request another review.
  - If the reviewer approves the changes, they will merge the pull request.
  - After the pull request is merged, you should delete your branch.
