import sys
import json
import signal
import uvicorn
from app.check_validate_config import validate_config_file

CONFIG_PATH = "configOriginal.json"


class Server:
    """Wrapper untuk uvicorn server dengan proper shutdown handling."""

    def __init__(self, app_path: str, host: str, port: int):
        self.app_path = app_path
        self.host = host
        self.port = port
        self.server = None
        self.should_exit = False

    def handle_signal(self, sig, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {sig}. Shutting down gracefully.")
        self.should_exit = True

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

        print(f"Starting API Gateway at http://{API_HOST}:{API_PORT}.")

        # Run server dengan proper signal handling
        server = Server("app.main:app", API_HOST, API_PORT)
        server.run()

    except FileNotFoundError:
        print("FATAL: config.json tidak ditemukan.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"FATAL: config.json tidak valid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL: Gagal menjalankan server: {e}")
        sys.exit(1)
