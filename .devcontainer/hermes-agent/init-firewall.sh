#!/bin/bash
# Firewall initialisation for ARC-AGI-3 Hermes evaluation container.
# Mirrors .devcontainer/hermes-agent/init-firewall.sh from sethkarten/continual-harness.
#
# Egress policy:
#   ALLOW  loopback
#   ALLOW  established/related
#   ALLOW  DNS (53 udp/tcp)
#   ALLOW  HTTPS (443) — Gemini LLM API
#   ALLOW  MCP_PORT → host.docker.internal  (the only game access)
#   DROP   GAME_SERVER_PORT                 (blocks direct ARC env access)
#   DROP   everything else

set -e

MCP_PORT="${MCP_PORT:-8002}"
GAME_SERVER_PORT="${GAME_SERVER_PORT:-8000}"

# Create a small wrapper that sets HOME/HERMES_HOME/PYTHONPATH before exec.
# Hermes must come first in PYTHONPATH so its own utils.py is found before
# any same-named module in /opt/arc-src.
cat > /tmp/run_hermes.sh << 'WRAPPER_EOF'
#!/bin/sh
export HOME=/home/hermes-agent
export HERMES_HOME=/home/hermes-agent/.hermes
export PYTHONPATH="/opt/hermes-agent:/opt/arc-src${PYTHONPATH:+:$PYTHONPATH}"
exec "$@"
WRAPPER_EOF
chmod +x /tmp/run_hermes.sh

if [ "${SKIP_FIREWALL:-0}" = "1" ]; then
    echo "⏭️  Skipping firewall (SKIP_FIREWALL=1)"
    exec su hermes-agent -c "/tmp/run_hermes.sh $*"
fi

echo "🔒 Initialising container firewall..."

iptables -F && iptables -X
iptables -P INPUT  DROP
iptables -P FORWARD DROP
iptables -P OUTPUT DROP

# Loopback
iptables -A INPUT  -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Established connections
iptables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT

# DNS
iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
iptables -A OUTPUT -p tcp --dport 53 -j ACCEPT

# HTTPS — Gemini LLM API.
# NOTE: ARC's online API also uses 443, so port-based isolation is not
# possible; protection comes from the container never holding ARC_API_KEY.
iptables -A OUTPUT -p tcp --dport 443 -j ACCEPT

# MCP server on the host — the only approved game access channel.
HOST_IP=$(getent hosts host.docker.internal 2>/dev/null | awk '{print $1}')
if [ -z "$HOST_IP" ]; then
    HOST_IP=$(ip route | awk '/default/ {print $3; exit}')
fi
iptables -A OUTPUT -p tcp -d "$HOST_IP" --dport "$MCP_PORT" -j ACCEPT
echo "   ✓ MCP server allowed: $HOST_IP:$MCP_PORT"

# Block direct access to the ARC game server (defence-in-depth).
iptables -A OUTPUT -p tcp --dport "$GAME_SERVER_PORT" -j DROP
echo "   ✓ Game server blocked: port $GAME_SERVER_PORT"

# IPv6 — mirror the same policy.
ip6tables -F 2>/dev/null && ip6tables -X 2>/dev/null || true
ip6tables -P INPUT   DROP 2>/dev/null || true
ip6tables -P FORWARD DROP 2>/dev/null || true
ip6tables -P OUTPUT  DROP 2>/dev/null || true
ip6tables -A INPUT  -i lo -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true
ip6tables -A INPUT  -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -p udp --dport 53 -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -p tcp --dport 53 -j ACCEPT 2>/dev/null || true
ip6tables -A OUTPUT -p tcp --dport 443 -j ACCEPT 2>/dev/null || true

echo "✅ Firewall initialised"

exec su hermes-agent -c "/tmp/run_hermes.sh $*"
