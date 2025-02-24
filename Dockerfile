FROM python:3.11-slim-buster

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1) Install uv package manager
#    By default, uv installs to ~/.local/bin. We update PATH so uv is recognized.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# 2) Copy the repository content
COPY . /app

# 3) Expose the port that LangGraph dev server uses (default: 2024)
EXPOSE 2024

# 4) Launch the assistant with the LangGraph dev server:
#    Equivalent to the quickstart: uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
CMD ["uvx", \
     "--refresh", \
     "--from", "langgraph-cli[inmem]", \
     "--with-editable", ".", \
     "--python", "3.11", \
     "langgraph", \
     "dev", \
     "--host", "0.0.0.0"]