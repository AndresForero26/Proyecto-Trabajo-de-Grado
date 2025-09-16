from flask import Flask
from src.config.config import Config, init_dirs
from src.models.db import db, init_db
from src.routes.index import bp as index_bp

def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['SQLALCHEMY_DATABASE_URI'] = Config.SQLALCHEMY_DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    app.config['MODEL_PATH'] = Config.MODEL_PATH
    app.config['CLASS_INDICES_PATH'] = Config.CLASS_INDICES_PATH

    db.init_app(app)
    with app.app_context():
        init_dirs()
        init_db()

    app.register_blueprint(index_bp)
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
