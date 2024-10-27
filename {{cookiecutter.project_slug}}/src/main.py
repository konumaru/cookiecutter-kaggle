import os

from utils import timer


def main() -> None:
    print("Hello Workd!")


if __name__ == "__main__":
    with timer(os.path.basename(__file__)):
        main()
