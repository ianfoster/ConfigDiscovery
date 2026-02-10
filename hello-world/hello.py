#!/usr/bin/env python3
"""
Simple Hello World program that takes a name parameter
"""
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Hello World program')
    parser.add_argument('name', help='Name to greet')
    parser.add_argument('--greeting', default='Hello', help='Greeting to use (default: Hello)')
    
    args = parser.parse_args()
    
    print(f"{args.greeting}, {args.name}!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
