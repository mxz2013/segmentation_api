A sementic segmentation model using PyTorch and torchvision. The API is build with FastAPI.
server_main.py is the main file to run the server.
client_main.py is the main file to run the client.

# run without a api server
python main.py --image tests/images/cat_3.jpg

# run with a api server
# start the server
python server_main.py 

# in another terminal, run the client
python client_main.py --image tests/images/cat_3.jpg 

# Requirements
- Python 3.10+
pip install -r requirements.txt
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Test concurrency
You can test the concurrency of the API server using the provided script `produce_stats_concurrent_users.py`.
python tests/produce_stats_concurrent_users.py

# Docker
You can build and run the Docker container using the provided Dockerfile.

docker build -t spore-segmentation-api:cuda .
docker run -p 8080:8000 --name spore_cpu_inference spore-segmentation-api:cuda
possible arguments:
- add -e MODEL=MODEL_NAME to set the model name
- add -e DEVICE=cuda/cpu to set the device for the server
- add -e WORKERS=NUM_WORKERS to set the number of workers for the server
- add -e PORT=PORT_NUMBER to set the port number for the server
- add -e HOST=HOST_ADDRESS to set the host address for the server

pull my docker image from docker hub
docker pull skyzhou323/spore-segmentation-api:1.0