#!/bin/bash
set -e

# Nous installer — one-line install for Linux servers
# Usage: curl -sSL https://raw.githubusercontent.com/artaeon/nous/main/install.sh | bash

INSTALL_DIR="/opt/nous"
BINARY="nous"
VERSION="0.9.0"

echo "  Installing Nous v${VERSION}..."
echo ""

# Detect architecture
ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

echo "  Platform: ${OS}/${ARCH}"

# Check for Ollama
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "  Ollama not found. Installing..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Pull default model
echo "  Pulling qwen2.5:1.5b..."
ollama pull qwen2.5:1.5b 2>/dev/null || true

# Pull fast model for perception
echo "  Pulling tinyllama..."
ollama pull tinyllama 2>/dev/null || true

# Build from source (if in repo) or download binary
if [ -f "go.mod" ] && grep -q "artaeon/nous" go.mod 2>/dev/null; then
    echo "  Building from source..."
    go build -o ${BINARY} ./cmd/nous
    sudo mkdir -p ${INSTALL_DIR}
    sudo cp ${BINARY} ${INSTALL_DIR}/
else
    echo "  Downloading binary..."
    sudo mkdir -p ${INSTALL_DIR}
    # For now, build from source requires cloning
    echo "  Please clone and build:"
    echo "    git clone https://github.com/artaeon/nous.git"
    echo "    cd nous && go build -o nous ./cmd/nous"
    echo "    sudo cp nous ${INSTALL_DIR}/"
fi

# Create user
if ! id -u nous &>/dev/null; then
    sudo useradd -r -s /bin/false -d ${INSTALL_DIR} nous 2>/dev/null || true
fi
sudo chown -R nous:nous ${INSTALL_DIR} 2>/dev/null || true

# Install systemd service
if [ -f "nous.service" ]; then
    sudo cp nous.service /etc/systemd/system/
    sudo systemctl daemon-reload
    echo "  Systemd service installed."
    echo ""
    echo "  Start with:  sudo systemctl start nous"
    echo "  Enable with: sudo systemctl enable nous"
fi

echo ""
echo "  Installation complete!"
echo ""
echo "  Usage:"
echo "    nous                        # Interactive REPL"
echo "    nous --serve --port 3333    # HTTP server mode"
echo "    nous --allow-shell          # Enable shell commands"
echo "    nous --trust                # Skip confirmation prompts"
echo "    nous --model qwen2.5:3b    # Use a different model"
echo ""
echo "  Web UI: http://localhost:3333 (in server mode)"
echo ""
