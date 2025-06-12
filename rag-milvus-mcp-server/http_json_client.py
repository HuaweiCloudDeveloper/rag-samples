from typing import Dict,Any
import requests


class HTTPJSONClient:

    def __init__(self,base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        base_url = self.base_url
        if endpoint.startswith("/"):
            endpoint = endpoint[1:]
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        url = f"{base_url}/{endpoint}"
        try:
            response = self.session.post(url, json=data)
            return self._handle_response(response)
        except requests.RequestException as e:
            raise ValueError(f"Request failed: {str(e)}")

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        response.raise_for_status()
        return response.json()