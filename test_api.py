import requests
import random
import json


def run_api_test():
    print("generating mock tensors for api payload...")

    # generate dimensions: static(5), past(30 time steps, 3 features), future(14 time steps, 2 features)
    payload = {
        "ticker": "AAPL_API_TEST",
        "static_features": [random.random() for _ in range(5)],
        "past_features": [[random.random() for _ in range(3)] for _ in range(30)],
        "future_features": [[random.random() for _ in range(2)] for _ in range(14)]
    }

    response = None  # fix for the unboundlocalerror

    print("sending post request to local fast api server...")
    try:
        # use localhost to allow macos to route ipv4/ipv6 correctly, add 10s timeout
        response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
        response.raise_for_status()

        print("\n[SUCCESS] api response:")
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.RequestException as e:
        print(f"\n[FAIL] api request failed: {e}")
        if response is not None:
            print(response.text)


if __name__ == "__main__":
    run_api_test()