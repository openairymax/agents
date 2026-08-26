#!/bin/bash
# Copyright (c) 2026 SPHARX. All Rights Reserved.
# "From data intelligence emerges."
#
# Airymax Orchestration Quick Run Script
# ========================
# This script runs the Architect Agent demonstration.
#
# Usage:
#   ./run.sh              # Run default demo
#   ./run.sh architect    # Run architect agent
#   ./run.sh test         # Run tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Airymax Orchestration Kernel${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 || python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ -z "$PYTHON_VERSION" ]; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Python version: $PYTHON_VERSION${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo ""
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r orchestration/requirements.txt
echo -e "${GREEN}Dependencies installed${NC}"

# Run the demo
echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Running Single-Agent End-to-End Demo${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# 默认演示：单 Agent 端到端（Mock 模式自动启用，无 Key 亦可运行）。
# 历史版本的 "Architect Agent Demo"（orchestration.agents.architect.agent）模块
# 已随架构演进移除，此处指向真实可运行的 examples 入口。
DEMO_SCRIPT="$SCRIPT_DIR/../examples/minimal/01_single_agent.py"
if [ -f "$DEMO_SCRIPT" ]; then
    python "$DEMO_SCRIPT"
else
    echo -e "${RED}Error: demo script not found: $DEMO_SCRIPT${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Demo Complete${NC}"
echo -e "${GREEN}======================================${NC}"
