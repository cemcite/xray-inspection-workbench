from urllib.request import urlopen


def main() -> None:
    with urlopen("http://localhost:8000/api/v1/health", timeout=3) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
