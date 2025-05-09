import requests
import json
from typing import Dict, Any

class HttpClient:

    def __init__(self,base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        try:
            response.raise_for_status()
            return response.json()
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}
        except requests.HTTPError as e:
            return {"error": f"HTTP error: {str(e)}"}
        
    def get(self, endpoint: str) -> Dict[str,Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url)
            return self._handle_response(response)
        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=data)
            return self._handle_response(response)
        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

