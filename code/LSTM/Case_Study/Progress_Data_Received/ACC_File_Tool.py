"""
ACC File Tool - Forge Client for BIM360/ACC
===========================================
Client for Autodesk Forge API (APS) operations with ACC/BIM360 integration.
Features:
- Direct upload to ACC folders (no OSS buckets)
- Automatic version control
- Model derivative translation
- Network retry mechanism
- Improved error handling
"""

import os
import base64
import logging
from urllib.parse import quote
import requests
import json, time
from typing import Dict, Any, Optional
from pathlib import Path

# Network retry imports
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import configuration
try:
    from forge_config import load_forge_config, get_env_variable
except ImportError:
    # Fallback if forge_config not available
    from dotenv import load_dotenv

    load_dotenv()

# Set up logging
logger = logging.getLogger("ForgeClient")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variable keys for compatibility
PROJECT_ENV_KEYS = ["PROJECT_ID", "project_id"]
FOLDER_ENV_KEYS = ["FOLDER_ID", "folder_id"]
FORGE_USER_ID = os.getenv("FORGE_USER_ID")

def create_session_with_retry(
        total_retries: int = 3,
        backoff_factor: float = 1.0,
        status_forcelist: tuple = (429, 500, 502, 503, 504)
) -> requests.Session:
    """
    Create a requests session with automatic retry capability

    Args:
        total_retries: Maximum number of retry attempts
        backoff_factor: Delay multiplier between retries (exponential backoff)
        status_forcelist: HTTP status codes that trigger a retry

    Returns:
        Configured requests Session
    """
    session = requests.Session()
    retry = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "PUT", "POST", "DELETE", "OPTIONS", "TRACE"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


class ForgeClient:
    """
    Client for Autodesk Forge API (APS) operations without using OSS buckets.
    Instead uploads directly to BIM360/ACC with version control support.
    """

    def __init__(self, client_id: str = None, client_secret: str = None):
        """
        Initialize Forge client

        Args:
            client_id: Forge application client ID (optional, reads from env)
            client_secret: Forge application client secret (optional, reads from env)
        """
        self.client_id = client_id or os.getenv("FORGE_CLIENT_ID") or os.getenv("CLIENT_ID")
        self.client_secret = client_secret or os.getenv("FORGE_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")
        self.access_token = None
        self.token_expiry = 0
        self.session = create_session_with_retry()
        self._3leg_tokens_path = "forge_tokens.json"
        self._3leg = None

        if not self.client_id or not self.client_secret:
            raise ValueError("FORGE_CLIENT_ID and FORGE_CLIENT_SECRET must be provided")

        logger.info("ForgeClient initialized")

    def _load_3leg(self):
        if self._3leg is not None:
            return self._3leg
        try:
            with open(self._3leg_tokens_path, "r", encoding="utf-8") as f:
                self._3leg = json.load(f)
            return self._3leg
        except FileNotFoundError:
            return None

    def _save_3leg(self, t):
        self._3leg = t
        with open(self._3leg_tokens_path, "w", encoding="utf-8") as f:
            json.dump(t, f, indent=2)

    def _3leg_valid(self):
        t = self._load_3leg()
        if not t: return False
        # access_token
        exp = t.get("expires_in", 0)
        obtained = t.get("obtained_at", 0)
        return time.time() < obtained + exp - 60

    def _3leg_access_token(self):
        t = self._load_3leg()
        return t.get("access_token") if t else None

    def _refresh_3leg(self):
        t = self._load_3leg()
        if not t or not t.get("refresh_token"):
            return False
        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": t["refresh_token"],
        }
        url = "https://developer.api.autodesk.com/authentication/v2/token"
        r = self.session.post(url, data=data, timeout=30)
        if not r.ok:
            return False
        newt = r.json()
        newt["obtained_at"] = int(time.time())
        # 有的返回不重复带 refresh_token，保留旧的
        if "refresh_token" not in newt and "refresh_token" in t:
            newt["refresh_token"] = t["refresh_token"]
        self._save_3leg(newt)
        return True

    def authenticate(self, scope="data:read", prefer_3leg_for_write=False):
        """
        - Read: default 2-legged (scope can be data:read)
        - Write: when prefer_3leg_for_write=True, prefer 3-legged, fallback to 2-legged on failure
        """

        if prefer_3leg_for_write:
            if self._3leg_valid():
                return self._3leg_access_token()
            elif self._refresh_3leg():
                return self._3leg_access_token()
            # if no 3-legged,can continue with 2-legged(also can throw error here suggesting execute first oauth_login.py)
            # print("No 3-legged token. Falling back to 2-legged with x-user-id...")

            # —— the following is 2-legged obtain(keep your original logic, but scope passed in keeping literal space separation)——
        if self.access_token and time.time() < self.token_expiry - 60 and getattr(self, "token_scope", ""):
            have = set(self.token_scope.split());
            need = set(scope.split())
            if need.issubset(have):
                return self.access_token

        url = "https://developer.api.autodesk.com/authentication/v2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scope": scope,
        }
        r = self.session.post(url, data=data, timeout=30)
        r.raise_for_status()
        tok = r.json()
        self.access_token = tok["access_token"]
        self.token_expiry = time.time() + tok["expires_in"]
        self.token_scope = scope
        return self.access_token


    def find_existing_item(self, project_id: str, folder_id: str, file_name: str) -> Optional[Dict[str, Any]]:
        """
        Find existing item in the folder by name

        Args:
            project_id: The project ID
            folder_id: The folder ID to search in
            file_name: The file name to search for

        Returns:
            Dict containing item data if found, None otherwise
        """
        self.authenticate(scope="data:read")

        logger.info(f"Searching for existing item '{file_name}' in folder {folder_id}")

        fid = quote(folder_id, safe='')  # ← 必须编码
        url = f"https://developer.api.autodesk.com/data/v1/projects/{project_id}/folders/{fid}/contents"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/vnd.api+json",
            "x-user-id": FORGE_USER_ID,
        }

        try:
            resp = self.session.get(url, headers=headers)
            resp.raise_for_status()
            contents = resp.json()

            # Search for matching file name
            for item in contents.get("data", []):
                if item.get("type") == "items":
                    display_name = item.get("attributes", {}).get("displayName", "")
                    if display_name == file_name:
                        logger.info(f"Found existing item: {item['id']}")
                        return item

            logger.info(f"No existing item found for '{file_name}'")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching for existing item: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return None

    def get_next_version_number(self, project_id: str, item_id: str) -> int:
        """
        Get the next version number for an item

        Args:
            project_id: The project ID
            item_id: The item ID

        Returns:
            Next version number
        """
        self.authenticate(scope="data:read")

        logger.info(f"Getting version count for item {item_id}")

        url = f"https://developer.api.autodesk.com/data/v1/projects/{project_id}/items/{item_id}/versions"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/vnd.api+json",
            "x-user-id": FORGE_USER_ID
        }

        try:
            resp = self.session.get(url, headers=headers)
            resp.raise_for_status()
            versions = resp.json()

            version_count = len(versions.get("data", []))
            next_version = version_count + 1
            logger.info(f"Next version number: {next_version}")
            return next_version

        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting version count: {str(e)}")
            # If we can't get version count, default to version 2
            return 2

    def create_new_version(self, file_path: str, project_id: str, item_id: str,
                           version_number: int, folder_id: str) -> Dict[str, Any]:
        """
        Create a new version for an existing item
        """
        self.authenticate(scope="data:write data:create")

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        logger.info(f"Creating new version {version_number} for item {item_id}")

        try:
            # Step 1: Create storage for the new version (JSON:API + specify targetfolder)
            storage_url = f"https://developer.api.autodesk.com/data/v1/projects/{project_id}/storage"
            storage_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "x-user-id": FORGE_USER_ID,
                "Content-Type": "application/vnd.api+json",
                "Accept": "application/vnd.api+json",
            }
            storage_data = {
                "data": {
                    "type": "objects",
                    "attributes": {"name": file_name},
                    "relationships": {
                        "target": {"data": {"type": "folders", "id": folder_id}}
                    }
                }
            }
            storage_resp = self.session.post(storage_url, headers=storage_headers, json=storage_data)
            storage_resp.raise_for_status()
            storage_result = storage_resp.json()
            object_id = storage_result["data"]["id"]
            logger.info(f"Storage created. Object ID: {object_id}")

            # Extract bucket and object keys
            urn_parts = object_id.split(":")
            bucket_key, object_key = urn_parts[-1].split("/")

            # Step 2: Get upload parameters and upload file
            upload_params_url = f"https://developer.api.autodesk.com/oss/v2/buckets/{bucket_key}/objects/{object_key}/signeds3upload"
            upload_params_headers = {"Authorization": f"Bearer {self.access_token}"}
            upload_params_resp = self.session.get(upload_params_url, headers=upload_params_headers)
            upload_params_resp.raise_for_status()
            upload_params_data = upload_params_resp.json()

            upload_key = upload_params_data["uploadKey"]
            upload_url = upload_params_data["urls"][0]

            # Upload the file
            logger.info(f"Uploading file ({file_size / 1024 / 1024:.2f} MB)")
            with open(file_path, 'rb') as file_content:
                upload_headers = {"Content-Type": "application/octet-stream"}
                upload_resp = self.session.put(upload_url, headers=upload_headers, data=file_content)
                upload_resp.raise_for_status()

            # Complete the upload
            complete_upload_url = f"https://developer.api.autodesk.com/oss/v2/buckets/{bucket_key}/objects/{object_key}/signeds3upload"
            complete_upload_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "x-user-id": FORGE_USER_ID
            }
            complete_upload_data = {"uploadKey": upload_key}
            complete_upload_resp = self.session.post(
                complete_upload_url,
                json=complete_upload_data,
                headers=complete_upload_headers
            )
            complete_upload_resp.raise_for_status()

            logger.info("File uploaded successfully")

            # Step 3: Create new version
            version_url = f"https://developer.api.autodesk.com/data/v1/projects/{project_id}/versions"
            version_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/vnd.api+json",
                "Accept": "application/vnd.api+json",
                "x-user-id": FORGE_USER_ID
            }

            version_data = {
                "jsonapi": {"version": "1.0"},
                "data": {
                    "type": "versions",
                    "attributes": {
                        "name": file_name,
                        "extension": {
                            "type": "versions:autodesk.bim360:File",
                            "version": "1.0"
                        }
                    },
                    "relationships": {
                        "item": {
                            "data": {
                                "type": "items",
                                "id": item_id
                            }
                        },
                        "storage": {
                            "data": {
                                "type": "objects",
                                "id": object_id
                            }
                        }
                    }
                }
            }

            version_resp = self.session.post(version_url, headers=version_headers, json=version_data)
            version_resp.raise_for_status()
            version_result = version_resp.json()

            version_id = version_result["data"]["id"]
            logger.info(f"Version {version_number} created. Version ID: {version_id}")

            return {
                "version_id": version_id,
                "item_id": item_id,
                "object_id": object_id,
                "version_number": version_number
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating new version: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise

    def upload_file_with_version_control(self, local_path: str, project_id: str, folder_id: str) -> dict:
        """
        minimal version:determinewhether/ifalready exists -> (create new item / create newversion)-> return {urn, item_id, version_id}
        onlyusecorrect JSON:API workflow/process;notwill for /versions/{versionId} do/perform POST/PUT.
        """
        assert os.path.isfile(local_path), f"File not found: {local_path}"
        file_name = os.path.basename(local_path)

        # use read firstCheck permission if file with same name exists
        existing = self.find_existing_item(project_id, folder_id, file_name)

        if not existing:
            result = self.upload_file_to_bim360(local_path, project_id, folder_id)
            return {
                "urn": result.get("urn"),
                "item_id": result.get("item_id"),
                "version_id": result.get("version_id"),
            }

        # already exists:calculate nextversionnumber and create newversion
        item_id = existing["id"]  # items 的 lineage id
        next_ver = self.get_next_version_number(project_id, item_id)
        ver_info = self.create_new_version(local_path, project_id, item_id, next_ver, folder_id)

        urn = ver_info["object_id"]
        version_id = ver_info["version_id"]

        return {
            "urn": urn,
            "item_id": item_id,
            "version_id": version_id,
        }



    def upload_file_to_bim360(self, file_path: str, project_id: str,
                              folder_id: str) -> Dict[str, Any]:
        """
        Upload a new file to BIM360/ACC folder

        Args:
            file_path: Path to the file to upload
            project_id: The project ID
            folder_id: The folder ID

        Returns:
            Dict containing upload result
        """
        self.authenticate(scope="data:write data:create")

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        logger.info(f"Uploading new file '{file_name}' ({file_size / 1024 / 1024:.2f} MB)")

        try:
            # Step 1: Create storage
            storage_url = f"https://developer.api.autodesk.com/data/v1/projects/{project_id}/storage"
            storage_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "x-user-id": FORGE_USER_ID,
                "Content-Type": "application/vnd.api+json",
                "Accept": "application/vnd.api+json",
            }
            storage_data = {
                "jsonapi": {"version": "1.0"},
                "data": {
                    "type": "objects",
                    "attributes": {
                        "name": file_name
                    },
                    "relationships": {
                        "target": {
                            "data": {
                                "type": "folders",
                                "id": folder_id
                            }
                        }
                    }
                }
            }

            storage_resp = self.session.post(storage_url, headers=storage_headers, json=storage_data)
            storage_resp.raise_for_status()
            storage_result = storage_resp.json()
            object_id = storage_result["data"]["id"]

            logger.info(f"Storage created. Object ID: {object_id}")

            # Extract bucket and object keys
            urn_parts = object_id.split(":")
            bucket_key, object_key = urn_parts[-1].split("/")

            # Step 2: Get signed upload URL
            upload_params_url = f"https://developer.api.autodesk.com/oss/v2/buckets/{bucket_key}/objects/{object_key}/signeds3upload"
            upload_params_headers = {"Authorization": f"Bearer {self.access_token}"}
            upload_params_resp = self.session.get(upload_params_url, headers=upload_params_headers)
            upload_params_resp.raise_for_status()
            upload_params_data = upload_params_resp.json()

            upload_key = upload_params_data["uploadKey"]
            upload_url = upload_params_data["urls"][0]

            # Step 3: Upload file
            logger.info("Uploading file content...")
            with open(file_path, 'rb') as file_content:
                upload_headers = {"Content-Type": "application/octet-stream"}
                upload_resp = self.session.put(upload_url, headers=upload_headers, data=file_content)
                upload_resp.raise_for_status()

            # Step 4: Complete upload
            complete_url = f"https://developer.api.autodesk.com/oss/v2/buckets/{bucket_key}/objects/{object_key}/signeds3upload"
            complete_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "x-user-id": FORGE_USER_ID
            }
            complete_data = {"uploadKey": upload_key}
            complete_resp = self.session.post(complete_url, json=complete_data, headers=complete_headers)
            complete_resp.raise_for_status()

            logger.info("File content uploaded successfully")

            # Step 5: Create item
            item_url = f"https://developer.api.autodesk.com/data/v1/projects/{project_id}/items"
            item_headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/vnd.api+json",
                "Accept": "application/vnd.api+json",
                "x-user-id": FORGE_USER_ID
            }

            item_data = {
                "jsonapi": {"version": "1.0"},
                "data": {
                    "type": "items",
                    "attributes": {
                        "displayName": file_name,
                        "extension": {
                            "type": "items:autodesk.bim360:File",
                            "version": "1.0"
                        }
                    },
                    "relationships": {
                        "tip": {
                            "data": {
                                "type": "versions",
                                "id": "1"
                            }
                        },
                        "parent": {
                            "data": {
                                "type": "folders",
                                "id": folder_id
                            }
                        }
                    }
                },
                "included": [
                    {
                        "type": "versions",
                        "id": "1",
                        "attributes": {
                            "name": file_name,
                            "extension": {
                                "type": "versions:autodesk.bim360:File",
                                "version": "1.0"
                            }
                        },
                        "relationships": {
                            "storage": {
                                "data": {
                                    "type": "objects",
                                    "id": object_id
                                }
                            }
                        }
                    }
                ]
            }

            item_resp = self.session.post(item_url, headers=item_headers, json=item_data)
            item_resp.raise_for_status()
            item_result = item_resp.json()

            # Extract IDs
            if "included" not in item_result or len(item_result["included"]) == 0:
                raise ValueError(f"Invalid item creation response")

            version_id = item_result["included"][0]["id"]
            item_id = item_result["data"]["id"]

            logger.info(f"Item created. Item ID: {item_id}, Version ID: {version_id}")

            # Use object_id as URN for Model Derivative API
            # No need for additional GET request which may cause 403
            urn = object_id

            logger.info(f"Upload complete. URN: {urn[-40:]}")

            return {
                "urn": urn,
                "item_id": item_id,
                "version_id": version_id,
                "object_id": object_id
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Upload failed: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise

    import base64

    def start_model_derivative_job(self, urn: str) -> Dict:
        self.authenticate(scope="data:read data:write")

        # URL-safe base64 encode the complete URN (including 'urn:' prefix)
        # Remove padding '=' characters as required by Forge API
        base64_urn = base64.urlsafe_b64encode(urn.encode()).decode().rstrip('=')

        logger.info(f"Starting derivative job for base64 URN: {base64_urn[:40]}...")

        url = "https://developer.api.autodesk.com/modelderivative/v2/designdata/job"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
            "x-ads-force": "true"
        }
        data = {
            "input": {"urn": base64_urn},
            "output": {
                "formats": [{
                    "type": "svf",
                    "views": ["2d", "3d"]
                }]
            }
        }

        try:
            resp = self.session.post(url, headers=headers, json=data)
            resp.raise_for_status()
            job_data = resp.json()
            logger.info(f"Derivative job started successfully")
            return job_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to start derivative job: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise

    def check_derivative_job(self, urn: str) -> Dict:
        self.authenticate(scope="data:read")

        # URL-safe base64 encode the complete URN
        base64_urn = base64.urlsafe_b64encode(urn.encode()).decode().rstrip('=')

        url = f"https://developer.api.autodesk.com/modelderivative/v2/designdata/{base64_urn}/manifest"
        headers = {"Authorization": f"Bearer {self.access_token}"}

        try:
            resp = self.session.get(url, headers=headers)
            resp.raise_for_status()
            manifest = resp.json()
            status = manifest.get('status', 'unknown')
            progress = manifest.get('progress', 'unknown')
            logger.debug(f"Derivative status: {status}, progress: {progress}")
            return manifest
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check derivative job: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            raise

    def wait_for_derivative_job(self, urn: str, timeout: int = 600,
                                check_interval: int = 5) -> Dict:
        """
        Wait for a Model Derivative job to complete

        Args:
            urn: The URN of the file
            timeout: Maximum time to wait in seconds
            check_interval: Time between status checks in seconds

        Returns:
            Dict containing the final manifest

        Raises:
            TimeoutError: If job doesn't complete within timeout
            Exception: If job fails
        """
        logger.info(f"Waiting for derivative job to complete (timeout: {timeout}s)")
        start_time = time.time()

        while time.time() - start_time < timeout:
            manifest = self.check_derivative_job(urn)
            status = manifest.get("status")

            if status == "success":
                logger.info("Derivative job completed successfully")
                return manifest
            elif status in ["failed", "timeout"]:
                error_msg = manifest.get("errors", [{"message": "Unknown error"}])[0].get("message")
                logger.error(f"Derivative job failed: {error_msg}")
                raise Exception(f"Derivative job failed: {error_msg}")

            progress = manifest.get("progress", "unknown")
            logger.info(f"Derivative job in progress: {progress}%")
            time.sleep(check_interval)

        raise TimeoutError(f"Derivative job timed out after {timeout} seconds")


# For backward compatibility and standalone use
def scan_and_upload_loop(watch_dir: str,
                         project_id: str,
                         folder_id: str,
                         client: ForgeClient,
                         poll_sec: int = 10,
                         processed_db: str = "processed.json",
                         run_derivative: bool = True):
    """
    Monitor directory and upload new/modified files

    NOTE: This function is kept for backward compatibility.
    Consider using Auto_Manual_Upload_Fixed.py instead for production use.

    Args:
        watch_dir: Directory to monitor
        project_id: ACC project ID
        folder_id: ACC folder ID
        client: ForgeClient instance
        poll_sec: Polling interval in seconds
        processed_db: Path to processed files database
        run_derivative: Whether to run derivative translation
    """
    from pathlib import Path
    import json

    # Load processed database
    if os.path.exists(processed_db):
        try:
            with open(processed_db, "r", encoding="utf-8") as f:
                processed = json.load(f)
        except Exception:
            processed = {}
    else:
        processed = {}

    logger.info(f"[Watcher] Monitoring: {watch_dir}")
    logger.info(f"[Watcher] Project: {project_id}")
    logger.info(f"[Watcher] Folder: {folder_id}")

    while True:
        try:
            for name in os.listdir(watch_dir):
                if not name.lower().endswith(".rvt"):
                    continue
                file_path = os.path.join(watch_dir, name)
                if not os.path.isfile(file_path):
                    continue

                mtime = os.path.getmtime(file_path)
                key = os.path.abspath(file_path)

                # New or updated file
                if key not in processed or processed[key] < mtime:
                    logger.info(f"[Watcher] Detected new/updated file: {file_path}")

                    # Upload
                    result = client.upload_file_with_version_control(file_path, project_id, folder_id)
                    urn = result["urn"]

                    # Optional derivative translation
                    if run_derivative:
                        logger.info(f"[Derivative] Starting job for URN: {urn[-20:]}")
                        client.start_model_derivative_job(urn)
                        client.wait_for_derivative_job(urn, timeout=900, check_interval=10)

                    # Record processed timestamp
                    processed[key] = mtime
                    with open(processed_db, "w", encoding="utf-8") as f:
                        json.dump(processed, f, ensure_ascii=False, indent=2)

                    # Display viewer URL
                    if not urn.startswith("urn:"):
                        b64 = base64.b64encode(urn.encode()).decode()
                    else:
                        b64 = base64.b64encode(urn[4:].encode()).decode()
                    logger.info(f"[Viewer] https://viewer.autodesk.com/?urn={b64}")

        except Exception as e:
            logger.error(f"[Watcher] Error: {e}")

        time.sleep(poll_sec)


if __name__ == "__main__":
    """
    Standalone execution mode
    """
    try:
        config = load_forge_config()
    except:
        # Fallback to environment variables
        config = {
            'client_id': os.getenv("FORGE_CLIENT_ID"),
            'client_secret': os.getenv("FORGE_CLIENT_SECRET"),
            'project_id': os.getenv("PROJECT_ID"),
            'folder_id': os.getenv("FOLDER_ID"),
        }

    watch_dir = os.getenv("WATCH_FOLDER", "input_revit_data")

    if not all([config['client_id'], config['client_secret'],
                config['project_id'], config['folder_id'], watch_dir]):
        logger.error("Missing required environment variables")
        logger.error("Required: FORGE_CLIENT_ID, FORGE_CLIENT_SECRET, PROJECT_ID, FOLDER_ID, WATCH_FOLDER")
        exit(1)

    if not os.path.isdir(watch_dir):
        logger.error(f"Watch directory not found: {watch_dir}")
        exit(1)

    client = ForgeClient(config['client_id'], config['client_secret'])

    logger.info("Starting file watcher...")
    scan_and_upload_loop(
        watch_dir=watch_dir,
        project_id=config['project_id'],
        folder_id=config['folder_id'],
        client=client,
        poll_sec=10,
        processed_db="processed.json",
        run_derivative=True
    )