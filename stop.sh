#!/bin/bash

# =======================================================================
# Stop all Engagement Analyzer services (Linux)
# =======================================================================

echo -e "\e[33mStopping services...\e[0m"

function free_port() {
    local port=$1
    if fuser -s "$port/tcp" 2>/dev/null; then
        fuser -k -9 "$port/tcp" 2>/dev/null
    fi
}

free_port 8000
free_port 8501
sleep 1

echo -e "\e[32mAll services stopped.\e[0m"
