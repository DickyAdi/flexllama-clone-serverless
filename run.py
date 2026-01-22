import os
import sys
import json
import signal
import asyncio
import uvicorn
import argparse
import threading
from aiohttp import web
from pathlib import Path
from datetime import datetime
from app.check_validate_config import validate_config_file


def get_config_path() -> str:
    """
    Get config path with priority: CLI argument > ENV variable > default.

    Usage:
        python run.py --config configOriginal.json
        python run.py -c /path/to/config.json
        CONFIG_PATH=config.json python run.py
        python run.py  # uses default config.json
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python run.py --config configOriginal.json
        python run.py -c /path/to/custom_config.json
        CONFIG_PATH=myconfig.json python run.py
        """
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default=None,
        help='Path to config JSON file (default: config.json or $CONFIG_PATH env var)'
    )

    args = parser.parse_args()

    # Priority: CLI arg > ENV var > default
    if args.config:
        return args.config
    return os.getenv("CONFIG_PATH", "config.json")


CONFIG_PATH = get_config_path()
# STATUS_SERVER_PORT = 8001
STATUS_SERVER_HOST = os.getenv('STATUS_SERVER_HOST', "0.0.0.0")
STATUS_SERVER_PORT = os.getenv('STATUS_SERVER_PORT', 80)


def init_status_file(config_data: dict):
    """
    Initialize status file sebelum FastAPI dimulai.
    Ini memungkinkan status bisa dibaca sebelum server ready.
    """
    project_root = Path(__file__).parent
    status_file = project_root / "logs" / "model_status.json"

    # Ensure logs directory exists
    status_file.parent.mkdir(parents=True, exist_ok=True)

    # Get model aliases dari config
    model_aliases = list(config_data.get("models", {}).keys())

    # Create initial status
    status_data = {
        "server": {
            "status": "starting",
            "started_at": None,
            "updated_at": datetime.now().isoformat()
        },
        "models": {
            alias: {
                "alias": alias,
                "status": "off",
                "port": None,
                "started_at": None,
                "last_used_at": None,
                "load_progress": None,
                "error_message": None,
                "vram_used_mb": None,
                "updated_at": datetime.now().isoformat()
            }
            for alias in model_aliases
        }
    }

    # Write to file
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)

    print(f"Status file initialized: {status_file}")
    return status_file


def run_status_server_thread(host: str, port: int, stop_event: threading.Event):
    """
    Run lightweight status server di thread terpisah.
    Server ini berjalan lebih cepat dari FastAPI dan bisa diakses segera.
    """
    try:
        project_root = Path(__file__).parent
        status_file = project_root / "logs" / "model_status.json"

        # Ensure logs directory exists
        status_file.parent.mkdir(parents=True, exist_ok=True)

        async def get_status(request):
            """Get current status from file."""
            try:
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        data = json.load(f)
                    return web.json_response(data)
                return web.json_response({
                    "server": {"status": "starting"},
                    "models": {},
                    "message": "Status file not yet created"
                })
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        async def get_model_status(request):
            """Get status untuk satu model."""
            model_alias = request.match_info.get('alias', '')
            try:
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        data = json.load(f)
                    models = data.get('models', {})
                    if model_alias in models:
                        return web.json_response(models[model_alias])
                return web.json_response(
                    {"error": f"Model '{model_alias}' not found"},
                    status=404
                )
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        async def health_check(request):
            """Simple health check."""
            return web.json_response({
                "status": "ok",
                "service": "status-server",
                "timestamp": datetime.now().isoformat()
            })

        async def sse_stream(request):
            """
            SSE stream untuk status updates via file polling.

            Event types yang dikirim:
            - full_status: Status lengkap saat pertama connect
            - server_update: Perubahan status server
            - model_update: Perubahan status satu model
            - models_update: Perubahan status multiple models sekaligus
            - heartbeat: Keep-alive setiap 30 detik
            """
            response = web.StreamResponse()
            response.headers['Content-Type'] = 'text/event-stream'
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['X-Accel-Buffering'] = 'no'

            await response.prepare(request)

            last_server_status = None
            last_models_status = {}
            poll_interval = 0.5  # Poll lebih cepat untuk responsif
            heartbeat_interval = 30
            last_heartbeat = asyncio.get_event_loop().time()

            try:
                # Send initial full status
                if status_file.exists():
                    with open(status_file, 'r') as f:
                        initial_data = json.load(f)

                    # Kirim full status
                    await response.write(
                        f"event: full_status\ndata: {json.dumps(initial_data)}\n\n".encode(
                        )
                    )

                    # Simpan state untuk tracking perubahan
                    last_server_status = initial_data.get("server", {})
                    last_models_status = initial_data.get("models", {})

                while not stop_event.is_set():
                    await asyncio.sleep(poll_interval)

                    current_time = asyncio.get_event_loop().time()

                    # Check for file changes
                    if status_file.exists():
                        try:
                            with open(status_file, 'r') as f:
                                current_data = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            continue

                        current_server = current_data.get("server", {})
                        current_models = current_data.get("models", {})

                        # Check server status change
                        if current_server != last_server_status:
                            server_event = {
                                "server": current_server,
                                "changed_fields": [
                                    k for k in current_server
                                    if last_server_status.get(k) != current_server.get(k)
                                ] if last_server_status else ["all"]
                            }
                            await response.write(
                                f"event: server_update\ndata: {json.dumps(server_event)}\n\n".encode(
                                )
                            )
                            last_server_status = current_server.copy()
                            last_heartbeat = current_time

                        # Check individual model changes
                        changed_models = []
                        for alias, model_data in current_models.items():
                            old_model_data = last_models_status.get(alias, {})

                            # Compare status dan field penting
                            if model_data != old_model_data:
                                # Tentukan apa yang berubah
                                changes = []
                                for key in ["status", "port", "load_progress", "vram_used_mb", "error_message"]:
                                    if model_data.get(key) != old_model_data.get(key):
                                        changes.append(key)

                                model_event = {
                                    "alias": alias,
                                    "previous_status": old_model_data.get("status"),
                                    "current": model_data,
                                    "changed_fields": changes
                                }
                                changed_models.append(model_event)

                        # Kirim model updates
                        if changed_models:
                            if len(changed_models) == 1:
                                # Single model update
                                await response.write(
                                    f"event: model_update\ndata: {json.dumps(changed_models[0])}\n\n".encode(
                                    )
                                )
                            else:
                                # Multiple models update
                                await response.write(
                                    f"event: models_update\ndata: {json.dumps({'models': changed_models, 'count': len(changed_models)})}\n\n".encode(
                                    )
                                )

                            last_models_status = {k: v.copy()
                                                  for k, v in current_models.items()}
                            last_heartbeat = current_time

                        # Check for removed models
                        for alias in list(last_models_status.keys()):
                            if alias not in current_models:
                                remove_event = {
                                    "alias": alias,
                                    "event": "removed"
                                }
                                await response.write(
                                    f"event: model_removed\ndata: {json.dumps(remove_event)}\n\n".encode(
                                    )
                                )
                                del last_models_status[alias]

                    # Send heartbeat
                    if current_time - last_heartbeat >= heartbeat_interval:
                        # Include summary in heartbeat
                        summary = {
                            "timestamp": datetime.now().isoformat(),
                            "type": "heartbeat",
                            "server_status": last_server_status.get("status") if last_server_status else "unknown",
                            "models_summary": {}
                        }

                        # Count models by status
                        for alias, model in last_models_status.items():
                            status = model.get("status", "unknown")
                            summary["models_summary"][status] = summary["models_summary"].get(
                                status, 0) + 1

                        await response.write(
                            f"event: heartbeat\ndata: {json.dumps(summary)}\n\n".encode(
                            )
                        )
                        last_heartbeat = current_time

            except (asyncio.CancelledError, ConnectionResetError):
                pass
            except Exception as e:
                print(f"SSE stream error: {e}")
                import traceback
                traceback.print_exc()

            return response

        # Create app dengan CORS
        async def cors_middleware(app, handler):
            async def middleware_handler(request):
                if request.method == 'OPTIONS':
                    response = web.Response()
                else:
                    response = await handler(request)
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = '*'
                return response
            return middleware_handler

        app = web.Application(middlewares=[cors_middleware])
        app.router.add_get('/status', get_status)
        app.router.add_get('/status/stream', sse_stream)
        app.router.add_get('/ping', health_check)

        async def run_server():
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            print(f"Status server running on http://{host}:{port}")

            # Keep running until stop event
            while not stop_event.is_set():
                await asyncio.sleep(0.5)

            await runner.cleanup()
            print("Status server stopped")

        # Run in new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server())
        loop.close()

    except ImportError:
        print("Warning: aiohttp not installed. Status server disabled.")
        print("Install with: pip install aiohttp")
    except Exception as e:
        print(f"Status server error: {e}")


class Server:
    """Wrapper untuk uvicorn server dengan proper shutdown handling."""

    def __init__(self, app_path: str, host: str, port: int):
        self.app_path = app_path
        self.host = host
        self.port = port
        self.server = None
        self.should_exit = False
        self.status_server_thread = None
        self.stop_event = threading.Event()

    def handle_signal(self, sig, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {sig}. Shutting down gracefully.")
        self.should_exit = True

        # Stop status server
        self.stop_event.set()

        if self.server:
            # Trigger server shutdown
            self.server.should_exit = True

    def run(self):
        """Run uvicorn server dengan signal handling."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        # Create config
        config = uvicorn.Config(
            app=self.app_path,
            host=self.host,
            port=self.port,
            reload=False,
            workers=1,
            log_level="info"
        )

        # Create server
        self.server = uvicorn.Server(config)

        # Run server
        try:
            self.server.run()
        except KeyboardInterrupt:
            print("\nShutdown complete.")
        finally:
            # Ensure status server stops
            self.stop_event.set()
            if self.status_server_thread and self.status_server_thread.is_alive():
                self.status_server_thread.join(timeout=5)
            print("Server stopped.")


if __name__ == "__main__":
    try:
        # Validasi config
        if not validate_config_file(CONFIG_PATH):
            print("\nFATAL: Config validation failed.")
            sys.exit(1)

        with open(CONFIG_PATH, 'r') as f:
            config_data = json.load(f)

        # Validasi struktur
        required_keys = ["api", "system", "models"]
        missing_keys = [key for key in required_keys if key not in config_data]

        if missing_keys:
            print(f"FATAL: Config tidak lengkap. Missing keys: {missing_keys}")
            sys.exit(1)

        if not config_data.get("models"):
            print("FATAL: Tidak ada model yang terdefinisi di config.json")
            sys.exit(1)

        API_HOST = config_data.get("api", {}).get("host", "0.0.0.0")
        API_PORT = config_data.get("api", {}).get("port", 8000)

        # Initialize status file SEBELUM server dimulai
        init_status_file(config_data)

        # Start status server di background thread
        # Ini akan tersedia SEGERA, tidak perlu menunggu FastAPI
        stop_event = threading.Event()
        status_thread = threading.Thread(
            target=run_status_server_thread,
            args=(STATUS_SERVER_HOST, STATUS_SERVER_PORT, stop_event),
            daemon=True
        )
        status_thread.start()

        # Beri waktu status server untuk start
        import time
        time.sleep(0.5)

        print(f"Starting API Gateway at http://{API_HOST}:{API_PORT}.")
        print(
            f"Status server available at http://{STATUS_SERVER_HOST}:{STATUS_SERVER_PORT}")

        # Run server dengan proper signal handling
        server = Server("app.main:app", API_HOST, API_PORT)
        server.status_server_thread = status_thread
        server.stop_event = stop_event
        server.run()

    except FileNotFoundError as e:
        print(f"{e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"{e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Gagal menjalankan server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
