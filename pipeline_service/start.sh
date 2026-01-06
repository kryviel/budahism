#!/bin/bash
set -e

echo "-----------------------------------------------------"
echo "🚀 STARTING MAIN FASTAPI SERVICE (Base Env)"
echo "-----------------------------------------------------"

uvicorn serve:app --host 0.0.0.0 --port 10006 --reload
