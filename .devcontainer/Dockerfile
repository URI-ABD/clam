FROM archlinux/base

# Dependancies.
RUN pacman -Syu --noconfirm -y && pacman -S --noconfirm \
    python \
    git \
    zsh \
    base-devel \
    openssh \
    neovim

# Rust.
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="${HOME}/.cargo/bin:${PATH}"

# Oh My Zsh.
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

## Helpful Aliases.
RUN echo 'alias wip="git commit --no-verify -m WIP"' >> ~/.zshrc
RUN echo 'alias vim=nvim' >> ~/.zshrc

# Git extras.
RUN curl -sSL https://git.io/git-extras-setup | sudo bash /dev/stdin
