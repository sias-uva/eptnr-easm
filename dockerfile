# Use the official Miniconda3 base image
FROM mambaorg/micromamba:1.4.3

# Set the working directory inside the container
WORKDIR /app

# Copy file structure to the container
COPY . .

# Adjust the permissions of the directory
USER root
RUN chmod -R 777 /app
USER $MAMBA_USER

# Create and activate the Conda environment
COPY --chown=$MAMBA_USER:$MAMBA_USER ./conda_env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# Run optimal run
RUN python optimal_run.py ./atlanta_data/resulting_graph/Atlanta_problem_graph_2023-05-17.gml ./atlanta_data/atlanta_census.parquet

# Set the command to run when the container starts
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
