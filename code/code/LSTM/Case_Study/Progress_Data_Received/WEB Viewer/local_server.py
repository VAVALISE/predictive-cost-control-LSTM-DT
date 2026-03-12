"""
Simple Local Web Server for Forge Viewer
=========================================
Serves viewer.html and progress_history.json for local testing.
"""

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json
import importlib.util
import cgi
from urllib.parse import urlparse, parse_qs


# Add parent directory to path to import forge modules
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


def import_from_parent(module_name):
    """Dynamically import module from parent directory"""
    module_path = parent_dir / f"{module_name}.py"
    if not module_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return None


def get_forge_token():
    """Get Forge access token from parent directory's ForgeClient"""
    try:
        # Import modules dynamically to avoid PyCharm warnings
        forge_config = import_from_parent('forge_config')
        acc_file_tool = import_from_parent('ACC_File_Tool')

        if not forge_config or not acc_file_tool:
            print("[ERROR] Could not import Forge modules from parent directory")
            return ''

        # Initialize ForgeClient（内部会用环境变量读取 FORGE_CLIENT_ID / SECRET）
        ForgeClient = getattr(acc_file_tool, 'ForgeClient', None)
        if ForgeClient is None:
            print("[ERROR] ACC_File_Tool has no class ForgeClient")
            return ''

        client = ForgeClient()

        # Viewer only needs read permissions, here we use data:read + viewables:read
        token = client.authenticate(scope="data:read viewables:read")

        if token:
            print(f"[INFO] Successfully retrieved Forge token (length: {len(token)})")
            return token
        else:
            print("[WARNING] Forge token is empty")
            return ''

    except Exception as e:
        print(f"[ERROR] Failed to get Forge token: {e}")
        print("[INFO] Make sure forge_config.py and ACC_File_Tool.py are in parent directory")
        print(f"[INFO] Parent directory: {parent_dir}")
        return ''


class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve files with proper CORS headers"""

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        """Handle GET requests"""
        # Special handling for API endpoints
        if self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode('utf-8'))
            return

        if self.path == '/api/token':
            # Return Forge access token
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            token = get_forge_token()

            self.wfile.write(json.dumps({
                'access_token': token,
                'expires_in': 3600
            }).encode('utf-8'))
            return

        # ---- New: Model Derivative polling status ----
        if self.path.startswith('/api/status'):
            #  Parse URL parameters ?urn=...
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            urn_vals = query.get('urn')
            if not urn_vals:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Missing urn parameter'}).encode('utf-8'))
                return

            urn = urn_vals[0]

            try:
                acc_file_tool = import_from_parent('ACC_File_Tool')
                ForgeClient = getattr(acc_file_tool, 'ForgeClient', None)
                if ForgeClient is None:
                    raise RuntimeError("ACC_File_Tool.ForgeClient not available")

                client = ForgeClient()

                # Call Forge's manifest API
                manifest = client.check_derivative_job(urn)
                status = manifest.get('status', 'unknown')
                progress = manifest.get('progress', '0%')

                # Only mark when translation is complete viewer_ready = True
                viewer_ready = (isinstance(status, str) and status.lower() == 'success')

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': status.lower() if isinstance(status, str) else status,
                    'progress': progress,
                    'viewer_ready': viewer_ready
                }).encode('utf-8'))
                return

            except Exception as e:
                msg = str(e)

                # If Forge manifest not yet generated (usually 404 Not Found),
                # Frontend logic continues polling, so return pending instead of 500
                if '404' in msg or 'Not Found' in msg:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        'status': 'pending',
                        'progress': '0%',
                        'viewer_ready': False
                    }).encode('utf-8'))
                    return

                # Other errors treated as actual failures
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': msg}).encode('utf-8'))
                return


        # Special handling for progress_history.json
        if self.path == '/progress_history.json':
            history_path = Path(__file__).parent / 'progress_history.json'

            if not history_path.exists():
                self.send_error(404, 'progress_history.json not found')
                return

            try:
                with open(history_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(data).encode('utf-8'))
                return
            except Exception as e:
                self.send_error(500, f'Error reading progress_history.json: {e}')
                return

        # Default handling for other files
        super().do_GET()

    def do_POST(self):
        """Handle POST requests (file upload)"""
        if self.path == '/api/upload':
            return self.handle_upload()
        else:
            self.send_error(501, "Unsupported POST path")
            return

    def handle_upload(self):
        """Handle model upload and start translation via Forge/ACC"""
        try:
            # 1) Parse multipart/form-data
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self.send_error(400, "Expected multipart/form-data")
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type,
                }
            )

            file_item = form['file'] if 'file' in form else None
            if file_item is None or not getattr(file_item, "filename", ""):
                self.send_error(400, "No file uploaded")
                return

            filename = os.path.basename(file_item.filename)

            # 2) Save to local watch directory (consistent with ACC_File_Tool)
            #    Default uses WATCH_FOLDER environment variable, otherwise use parent_dir/input_revit_data
            watch_dir = os.getenv("WATCH_FOLDER") or "input_revit_data"
            watch_path = parent_dir / watch_dir if not os.path.isabs(watch_dir) else Path(watch_dir)
            watch_path.mkdir(parents=True, exist_ok=True)

            local_path = watch_path / filename
            with open(local_path, 'wb') as f:
                f.write(file_item.file.read())

            print(f"[INFO] Saved uploaded file to: {local_path}")

            # 3) Read project/folder config (forge_config priority, then environment variables)
            forge_config = import_from_parent('forge_config')
            if forge_config and hasattr(forge_config, 'load_forge_config'):
                cfg = forge_config.load_forge_config()
                project_id = cfg.get('project_id')
                folder_id = cfg.get('folder_id')
                client_id = cfg.get('client_id')
                client_secret = cfg.get('client_secret')
            else:
                project_id = os.getenv("PROJECT_ID") or os.getenv("project_id")
                folder_id = os.getenv("FOLDER_ID") or os.getenv("folder_id")
                client_id = os.getenv("FORGE_CLIENT_ID") or os.getenv("CLIENT_ID")
                client_secret = os.getenv("FORGE_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")

            if not project_id or not folder_id:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "PROJECT_ID or FOLDER_ID not configured"
                }).encode('utf-8'))
                return

            # 4) Call ACC_File_Tool upload + version control
            acc_file_tool = import_from_parent('ACC_File_Tool')
            if acc_file_tool is None or not hasattr(acc_file_tool, 'ForgeClient'):
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "ACC_File_Tool.ForgeClient not available"
                }).encode('utf-8'))
                return

            ForgeClient = acc_file_tool.ForgeClient
            client = ForgeClient(client_id, client_secret)

            # Upload + version control
            upload_result = client.upload_file_with_version_control(
                str(local_path),
                project_id=project_id,
                folder_id=folder_id
            )

            urn = upload_result.get("urn")
            item_id = upload_result.get("item_id")
            version_id = upload_result.get("version_id")

            if not urn:
                raise RuntimeError("Upload succeeded but URN not found in response")

            print(f"[INFO] Upload OK, URN: {urn}")

            # 5) Start Model Derivative translation job
            client.start_model_derivative_job(urn)

            # 6) Return JSON to frontend
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "urn": urn,
                "item_id": item_id,
                "version_id": version_id
            }).encode('utf-8'))

        except Exception as e:
            # On error return 500 + error message, frontend will display it
            print(f"[ERROR] Upload/translation failed: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": str(e)
            }).encode('utf-8'))


    def log_message(self, format, *args):
        """Custom log message format"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    # Change to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    print("=" * 70)
    print("Forge Viewer Local Server")
    print("=" * 70)
    print(f"Server directory: {script_dir}")
    print(f"Parent directory: {script_dir.parent}")
    print()

    # Check required files
    required_files = ['viewer.html', 'progress_history.json', 'urn_mapping.json']
    missing_files = []

    for filename in required_files:
        filepath = script_dir / filename
        if filepath.exists():
            print(f"✓ Found: {filename}")
        else:
            print(f"✗ Missing: {filename}")
            missing_files.append(filename)

    print()

    # Check parent directory for Forge modules
    print("Checking Forge configuration...")
    parent_dir = script_dir.parent
    forge_files = ['forge_config.py', 'ACC_File_Tool.py']
    forge_ok = True

    for filename in forge_files:
        filepath = parent_dir / filename
        if filepath.exists():
            print(f"✓ Found in parent: {filename}")
        else:
            print(f"✗ Missing in parent: {filename}")
            forge_ok = False

    print()

    if not forge_ok:
        print("WARNING: Forge configuration files not found in parent directory!")
        print(f"  Expected location: {parent_dir}")
        print("  This may cause issues when loading models.")
        print()

    if missing_files:
        print("WARNING: Missing required files!")
        if 'progress_history.json' in missing_files:
            print("  → Run 'python Progress_Viewer.py' (option 4) to generate history")
            print(f"  → Then copy from: {parent_dir / 'progress_history.json'}")
        if 'urn_mapping.json' in missing_files:
            print("  → Copy from parent directory:")
            print(f"    copy {parent_dir / 'urn_mapping.json'} .")
        if 'viewer.html' in missing_files:
            print("  → Copy viewer.html to this directory")
        print()

    # Start server
    port = 8000
    server_address = ('', port)

    print(f"Starting server on http://localhost:{port}")
    print()
    print("Open your browser and navigate to:")
    print(f"  → http://localhost:{port}/model_links.html (recommended)")
    print(f"  → http://localhost:{port}/viewer.html")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    print()

    try:
        httpd = HTTPServer(server_address, CustomHTTPRequestHandler)
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)


if __name__ == '__main__':
    main()