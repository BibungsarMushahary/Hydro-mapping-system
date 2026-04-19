#!/usr/bin/env python3
"""
HydroML System — Entry Point
============================
Starts the Flask web server.

Usage:
    python run.py [--port 5000] [--host 0.0.0.0] [--debug]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Run HydroML Web Server')
    parser.add_argument('--port',  type=int,  default=5000)
    parser.add_argument('--host',  type=str,  default='0.0.0.0')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Ensure output directories exist
    for d in ['data/uploads', 'data/outputs', 'data/models']:
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("  🌊  HydroML — Drainage & Flood Risk System")
    print("=" * 60)
    print(f"  Server  : http://{args.host}:{args.port}")
    print(f"  Debug   : {args.debug}")
    print("=" * 60)

    from backend.app import app
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == '__main__':
    main()
