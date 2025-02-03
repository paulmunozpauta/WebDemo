from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()  # Initialize SQLAlchemy outside create_app
migrate = Migrate()  # Initialize Flask-Migrate outside create_app

def create_app(config_class=None):
    app = Flask(__name__)
    
    # Configure the app
    app.config.from_object(config_class or 'config.DefaultConfig')

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)

    # Import and register blueprints (if applicable)
    from .routes import main as main_routes
    app.register_blueprint(main_routes)

    return app
