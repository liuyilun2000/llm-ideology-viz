import argparse

def main(args=None):
    parser = argparse.ArgumentParser(description="Generate synthetic data with LLMs.")
    # Add arguments as needed
    if args is None:
        args = parser.parse_args()
    print('TODO: Implement prompt/response generation') 