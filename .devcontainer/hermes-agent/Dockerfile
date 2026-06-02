# Containerised environment for Hermes Agent on ARC-AGI-3 evaluation.
# Mirrors .devcontainer/hermes-agent in sethkarten/continual-harness,
# adapted for the hermes-agent main branch (no mini-swe-agent submodule).

FROM python:3.11-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    iptables \
    iproute2 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

ARG USERNAME=hermes-agent
ARG USER_UID=1001
ARG USER_GID=1001

# Create non-root user that matches the host UID/GID to avoid permission issues
# on bind-mounted volumes.
RUN if getent group $USER_GID > /dev/null 2>&1; then \
        groupdel $(getent group $USER_GID | cut -d: -f1); \
    fi && \
    if getent passwd $USER_UID > /dev/null 2>&1; then \
        userdel -f $(getent passwd $USER_UID | cut -d: -f1); \
    fi && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Clone and install Hermes from main branch.
# Pass --build-arg HERMES_COMMIT=<sha> to pin a known-good commit if main drifts.
ARG HERMES_COMMIT=""
RUN git clone https://github.com/NousResearch/hermes-agent.git /opt/hermes-agent && \
    cd /opt/hermes-agent && \
    if [ -n "$HERMES_COMMIT" ]; then git checkout "$HERMES_COMMIT"; fi && \
    pip install --no-cache-dir ".[mcp]"

COPY init-firewall.sh /usr/local/bin/init-firewall.sh
RUN chmod +x /usr/local/bin/init-firewall.sh

USER $USERNAME
WORKDIR /workspace

# init-firewall.sh runs as root (via sudo -E), sets up iptables, then execs
# the provided CMD as the hermes-agent user.
ENTRYPOINT ["sudo", "-E", "/usr/local/bin/init-firewall.sh"]
