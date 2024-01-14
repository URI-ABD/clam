# CLAM: Contributor Guidelines

Pull requests and bug reports are welcome.
For major changes, please open an issue to discuss what you would like to change.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Prerequisites

- [`rust`](https://www.rust-lang.org/tools/install)
- [`docker`](https://docs.docker.com/engine/install/)
  - You will need this to test the CI/CD pipelines locally or to use Earthly to run various project commands.
  - You may need to start the docker daemon before running any commands. You can do this with `sudo systemctl start docker`.
  - You may need to add your user to the `docker` group to run docker commands without `sudo`. You can do this with `sudo usermod -aG docker $USER`. You need to re-login or reboot after doing this.
- [`hermit`](https://cashapp.github.io/hermit/usage/get-started/)
  - Use the curl command in the hermit link above to install it
  - You may need to add `/home/username/bin` to your $PATH if it's not already there.
  - This tool provides binaries you may want to have at hand to work on this repo.
  - You can see the full list of tools in the `./bin` directory in the repo root.
  - Do one of the following options:
    - Install each tool manually (do this if you do not want to use hermit).
    - Execute the tools you want directly with `./bin/<tool>` (do this if you want to use *some* tools from hermit).
    - (Recommended) Install shell hooks (see [here](https://cashapp.github.io/hermit/usage/shell/) for more information) (do this if you want to use *all* tools from hermit). Installing shell-hooks will eliminate the need to reactivate your hermit environment every time you open this repository.

### Windows Users
- [`wsl`](https://learn.microsoft.com/en-us/windows/wsl/install)
  - You will need to enable systemd for docker to run
    - Start your Ubuntu (or other Systemd) distribution under WSL
    - Run command 'sudo -e /etc/wsl.conf'
    - Add the following to the file:<br>
     [boot]<br>
     systemd=true
    - Restart WSL

### Things included with `hermit`

> Here are some of the tools we include by default with hermit that you may want to install on your own if you want all of the functionality of this repo.

- [`earthly`](https://earthly.dev/get-earthly)
  - This is the build tool used by the project.
  - You can use it to run various commands, such as `cargo test` or `cargo fmt`.
  - You can also use it to build the project, run the project, or build the documentation.
  - Example commands:
    - `earthly +test`
    - `earthly +fmt`
    - You can see all of the current targets available with `earthly ls`
- [`make`](https://www.gnu.org/software/make/)
  - This is a build tool that is used by the individual crates under the `crates/` directory.
  - Earthly does not support wildcard imports, so we use `make` to access earthlfile targets dynamically.
  - You can use it to run various commands, such as `make test` or `make publish`.
  - Example commands:
    - `make test`
    - `make publish`
    - All available make targets can be seen in the root makefile under `crates/Makefile`. Additionally, crates may override these targets or define new ones, so be sure to check the individual crate makefile before calling targets.
- [`python`](https://www.python.org/)
  - Some of our crates will offer python bindings, so we include python in hermit.
  - You can use it to run various commands, such as `python -m pytest` or `python -m pip install -r requirements.txt`.
  - Example commands:
    - `python -m pytest`
    - `python -m pip install -r requirements.txt`

## Getting Started

1. Fork the repository to your own GitHub account. You should make changes in your own fork and contribute back to the base repository (under URI-ABD) via pull requests.
2. Clone the repo from your fork.
   1. `git clone ...` or `gh repo clone ...`
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
   <!-- TODO: Add steps for setting up Bump2version and when to use each type (e.g., patch) -->
8.  Commit and push your changes.
9.  Open a pull request.

### Python-Specific Work

1. Install [`poetry`](https://python-poetry.org/docs/#installation).
   1. `python -m pip install poetry`
2. Install the project dependencies, using `poetry`.
   - Use poetry to install the project dependencies: `poetry install`
