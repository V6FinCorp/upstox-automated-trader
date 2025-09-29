try:
    # When installed or run from project root using package imports
    from .cli.scan_once import main as scan_once_main  # type: ignore
except ImportError:
    # When executed directly from the src folder
    from cli.scan_once import main as scan_once_main

if __name__ == "__main__":
    scan_once_main()
