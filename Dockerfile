FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /repo

# COPY pixi.lock /repo/pixi.lock
COPY pyproject.toml /repo/pyproject.toml

RUN apt-get update && apt-get install -y git
RUN /usr/local/bin/pixi install --manifest-path pyproject.toml --environment cuda

# create a shell-hook so commands passed tot he container are run in the env
RUN pixi shell-hook -s bash > /shell-hook
RUN echo "#!/bin/bash" > /repo/entrypoint.sh
RUN cat /shell-hook >> /repo/entrypoint.sh
# make python available in the container
RUN echo 'cd /repo && pixi shell' >> /repo/entrypoint.sh
# extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /repo/entrypoint.sh

# add libgl deps to run properly on cluster
RUN apt-get -y update && apt-get install -y libgl1-mesa-dev

# Entrypoint shell script ensures that any commands we run start with `pixi shell`,
# which in turn ensures that we have the environment activated
# when running any commands.
RUN chmod 700 /repo/entrypoint.sh
ENTRYPOINT [ "/repo/entrypoint.sh" ]
