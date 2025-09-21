FROM nvcr.io/nvidia/pytorch:25.06-py3

# DLI setup
COPY composer/bashrc /root/.bashrc
WORKDIR /workspace/exercises

# Environment variables
ENV HDF5_USE_FILE_LOCKING=FALSE
ENV LOGURU_LEVEL=INFO
ENV EARTH2STUDIO_CACHE=/workspace/data/earth2cache
ENV EARTH2STUDIO_PACKAGE_TIMEOUT=600
ENV NGC_CLI_ORG=no-org
ENV NGC_CLI_TEAM=no-team

# Dependencies
COPY --from=ghcr.io/astral-sh/uv:0.7.3 /uv /uvx /bin/
RUN pip install --upgrade ipywidgets jupyterlab>=4.3.6 jupyterlab-widgets jupyter-archive jupyter-resource-usage
RUN pip install --upgrade cartopy seaborn windpowerlib==0.2.2 zarr>=3
RUN pip install "makani[all] @ git+https://github.com/NickGeneva/modulus-makani.git@3da09f9e52a6393839d73d44262779ac7279bc2f"
RUN uv pip install --system --break-system-packages "earth2studio[data,corrdiff,perturbation,sfno] @ git+https://github.com/swbg/earth2studio.git@stefan/earth2dli-25.07"

# Download plotting assets
COPY composer/fetch_data.py ..
RUN python3 ../fetch_data.py

# Create entrypoint
EXPOSE 8888
COPY composer/entrypoint.sh ..
ENTRYPOINT ["/workspace/entrypoint.sh"]
