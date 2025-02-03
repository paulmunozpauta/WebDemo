import logging
from logging.config import fileConfig

from flask import current_app

from alembic import context

# This is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers.
fileConfig(config.config_file_name)
logger = logging.getLogger('alembic.env')


def get_users_engine():
    """Get the engine specifically for the `users` database."""
    try:
        return current_app.extensions['migrate'].db.get_engine(bind='users')
    except (TypeError, AttributeError):
        return current_app.extensions['migrate'].db.engines['users']


def get_users_metadata():
    """Retrieve the metadata of the `users` database."""
    from wikimpacts_web.app import User  # Import your `User` model
    return User.metadata  # Use the `metadata` of your `User` model


def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = get_users_engine().url.render_as_string(hide_password=False).replace('%', '%%')
    context.configure(
        url=url,
        target_metadata=get_users_metadata(),
        literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode."""

    def process_revision_directives(context, revision, directives):
        if getattr(context.config.cmd_opts, 'autogenerate', False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info('No changes in schema detected.')

    connectable = get_users_engine()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=get_users_metadata(),
            process_revision_directives=process_revision_directives,
            **current_app.extensions['migrate'].configure_args
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
