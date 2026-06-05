import requests

API_KEY = "bQKs7wOAnp5FiESwRJx8LFZ7KM56SZyn"

def test_coordinate(name, lat, lon, zoom=18):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/relative/{zoom}/json"
    params = {
        "key": API_KEY,
        "point": f"{lat},{lon}",
        "unit": "KMPH"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            flow = data.get("flowSegmentData", {})
            current_speed = flow.get("currentSpeed")
            free_flow_speed = flow.get("freeFlowSpeed")
            frc = flow.get("frc")
            ratio = (current_speed / free_flow_speed) if free_flow_speed and free_flow_speed > 0 else None
            ratio_str = f"{ratio:.2f}" if ratio is not None else "N/A"
            print(f"{name} ({lat}, {lon}) -> Speed: {current_speed} / {free_flow_speed} km/h | Ratio: {ratio_str} | FRC: {frc}")
        else:
            print(f"{name} -> Error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"{name} -> Exception: {e}")

if __name__ == "__main__":
    test_coordinate("Campus", 10.875286, 106.795604, 18)
    test_coordinate("Street Original", 10.8715, 106.8025, 18)
    test_coordinate("Hanoi Highway Center", 10.8715, 106.8076, 18)
    test_coordinate("Hanoi Highway Center (zoom 12)", 10.8715, 106.8076, 12)
