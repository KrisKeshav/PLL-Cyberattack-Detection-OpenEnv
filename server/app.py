"""
server/app.py — Server entry point for openenv validate compatibility.
"""
import uvicorn
from src.api import app  # noqa: F401


def main():
    """Start the FastAPI server."""
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
