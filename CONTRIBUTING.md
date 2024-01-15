# CLAM: Contributor Guidelines

Pull requests and bug reports are welcome.
For major changes, please open an issue to discuss what you would like to change.

As of writing this document, the project is still in a pre-1.0 state.
This means that the API is not yet stable and breaking changes may occur frequently.

## Prerequisites

### Mandatory

The following tools are mandatory.

- [`rust`](https://www.rust-lang.org/tools/install) for working on the project at all.
- [`git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) for version control.
- [`pre-commit`](https://pre-commit.com/#install) for running pre-commit hooks.
  - You will need this to run the pre-commit hooks, which will run various checks on your code when you commit to `git`.
  - The `pre-commit` hooks will try to automatically fix any issues they can, but some issues will require manual intervention.
  - If your code does not pass these checks, your PR will not even be reviewed or considered for merging until you fix the issues.
  - `pre-commit` is available via `pip` (`python -m pip install pre-commit`), `brew` (`brew install pre-commit`) on macOS, or via [`snap`](https://snapcraft.io/install/pre-commit/ubuntu) for Linux users.

#### Python Specific Work

If you are working on the Python bindings, you will need:

- [`python`](https://www.python.org/downloads/) 3.11.
- [`poetry`](https://python-poetry.org/docs/#installation) for managing python dependencies.
  - `python -m pip install poetry` will install poetry in the local python environment. You could also install it globally by following the instructions in the link above.
  - To install the dependencies for a python package, `cd` into the package directory and run `poetry install`.

### Optional

The following tools are optional, but recommended.

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
    - Run the command `sudo -e /etc/wsl.conf`
    - Add the following to the file:
         ```
         [boot]
         systemd=true
         ```
    - In powershell, restart WSL by running the commands `wsl --shutdown` and `wsl`.
    - In your WSL terminal, run the command `sudo systemctl start docker` to start the docker daemon.
    - You can have docker start automatically with WSL by running the command `sudo systemctl enable docker`.
    - If you run into any issues, see [`here`](https://askubuntu.com/questions/1379425/system-has-not-been-booted-with-systemd-as-init-system-pid-1-cant-operate) for help

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
7. Commit and push your changes.
8. Open a pull request.
9. Bump the version.
   1. We use [`bump2version`](https://github.com/c4urself/bump2version) to manage versioning and to keep the version numbers in sync.
   2. You should ask someone to review your changes before bumping the version. You should also ask which `part` to bump.
