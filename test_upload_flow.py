import requests

BASE_URL = "http://127.0.0.1:8080"
UPLOAD_ENDPOINT = f"{BASE_URL}/upload"
PREVIEW_ENDPOINT = f"{BASE_URL}/api/data/preview"  # or /api/data_preview if that's your main endpoint

TEST_FILE_PATH = "test_data.csv"

def test_upload_and_preview():
    session = requests.Session()

    # Step 1: Get the upload page (to get cookies/session)
    resp = session.get(UPLOAD_ENDPOINT)
    print("GET /upload status:", resp.status_code)

    # Step 2: Upload the file
    with open(TEST_FILE_PATH, "rb") as f:
        files = {'file': (TEST_FILE_PATH, f, 'text/csv')}
        resp = session.post(UPLOAD_ENDPOINT, files=files, allow_redirects=True)
        print("POST /upload status:", resp.status_code)
        if resp.history:
            print("Redirected to:", resp.url)
        else:
            print("No redirect after upload.")

    # Step 3: Call the preview endpoint
    resp = session.get(PREVIEW_ENDPOINT)
    print(f"GET {PREVIEW_ENDPOINT} status:", resp.status_code)
    try:
        print("Preview response:", resp.json())
    except Exception as e:
        print("Failed to parse JSON:", e)
        print("Raw response:", resp.text)

if __name__ == "__main__":
    test_upload_and_preview()
