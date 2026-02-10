#!/bin/bash
# Simple Hello World shell script that takes a name parameter

# Default greeting
GREETING="Hello"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --greeting)
            GREETING="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--greeting GREETING] NAME"
            echo "  --greeting GREETING   Greeting to use (default: Hello)"
            echo "  NAME                  Name to greet"
            exit 0
            ;;
        *)
            NAME="$1"
            shift
            ;;
    esac
done

# Check if name is provided
if [ -z "$NAME" ]; then
    echo "Error: Please provide a name to greet"
    echo "Usage: $0 [--greeting GREETING] NAME"
    exit 1
fi

echo "$GREETING, $NAME!"
