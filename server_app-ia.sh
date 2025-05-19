#!/bin/bash
source "$HOME/miniforge3/bin/activate" cc6409 && \
cd "$HOME/twitter-censor-plugin/app-ia" && \
mod_wsgi-express start-server application.wsgi --port 8905 \
--server-root "$HOME/twitter-censor-plugin/apache-app-ia" \
--access-log --log-to-terminal \
2>&1 | /usr/bin/cronolog "$HOME/twitter-censor-plugin/apache-app-ia/logs/apache.%Y-%m-%d.log"