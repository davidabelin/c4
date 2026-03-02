"""Local entrypoint for the c4 Flask app scaffold."""

from c4_web import create_app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
