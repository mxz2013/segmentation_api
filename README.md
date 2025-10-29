A semantic segmentation model using PyTorch and torchvision. The API is build with FastAPI, with RQ for task queue management. The key improvement is to handel requests asynchronously, allowing multiple clients to submit requests simultaneously without blocking. 
The heavy lifting of model inference is handled by RQ workers, which process tasks sequentially from a Redis queue, either by CPU or GPU.
The code is designed to cache model loading correctly to avoid reloading the model for each request in the same worker process 

# Requirements
- poetry is recommended for dependency management.
- poetry install will do all the job
- The code has been tested on Ubuntu 24.04 with Python 3.10 with CPU inference.
- Python 3.10.6 

# How to run
1. Start a Redis server (make sure redis-server is installed), `redis-server --port 6379`
2. Start RQ worker(s): `python segmentation_api/api/worker.py`
3. Start FastAPI server: `python segmentation_api/api/server.py`
4. Send requests to the server `python segmentation_api/api/client.py`

# How fastapi works with RQ and Redis 
Suppose that we are having 3 clients sending requests simultaneously to the server for image segmentation.

## 1. Client -> Server (request submission)
```
# 3 threads simultaneously send these requests:
response1 = session.post("/predict", json=request1.dict())  # Thread 1
response2 = session.post("/predict", json=request2.dict())  # Thread 2  
response3 = session.post("/predict", json=request3.dict())  # Thread 3`
```
After sending all requests, the clients start to wait for responses.

## 2. Server processing (FastAPI - Asynchronous handling)
```
# FastAPI handles these CONCURRENTLY using async
async def predict_single_image(request: PredictionRequest):
    # 1. Validate request (fast)
    if not Path(request.image_path).exists():
        raise HTTPException(404, "Image not found")
    
    # 2. Create task data (fast)
    task_data = ServerTaskData(
        image_path=request.image_path,
        target_class_ids=request.target_class_ids,
        threshold=request.threshold,
        model_name=self.model_name,
        device=self.device,
    )
    
    # 3. Enqueue to Redis (fast - just metadata)
    job = inference_queue.enqueue(perform_segmentation_task, task_data.dict())
    
    # 4. Return immediate response
    return ServerQueueStatus(
        success=True,
        job_id=job.id,
        status=f"/job_status/{job.id}",
        position_in_queue=job.get_position(),  # 0, 1, 2, etc.
    )
```
## 3. Redis Queue Management (Redis + RQ Workers)
in Redis, the jobs are queued as they arrive. RQ workers will pick them up one by one for processing.
```
Time T0: Request1 → Job1 created → Queue: [Job1]
Time T0: Request2 → Job2 created → Queue: [Job1, Job2]  
Time T0: Request3 → Job3 created → Queue: [Job1, Job2, Job3]

# What's actually in Redis:
Job1 = {
    "id": "uuid-1", 
    "data": {"image_path": "...", "threshold": 0.5, ...},
    "status": "queued",
    "created_at": "2024-01-01T10:00:00"
}

Job2 = {
    "id": "uuid-2",
    "data": {"image_path": "...", "threshold": 0.5, ...}, 
    "status": "queued",
    "created_at": "2024-01-01T10:00:01"
}
...
```

## 4. RQ Worker Processing (sequential processing, i.e., synchronous)
The heavy lifting of model inference is done here. Each worker picks up one job at a time, performs the segmentation task (which is blocking/synchronous), and then stores the result back in Redis.
```
# Worker runs in infinite loop (a piece of pseudo-code):
while True:
    job = queue.dequeue()  # Gets Job1 (blocks until job available)
    result = perform_segmentation_task(job.data)  # SYNC - blocks here
    job.set_result(result)  # Store result in Redis
    # Then gets Job2, then Job3...
```

After processing results, the clients can poll for job status and retrieve results by a unique job id from Redis.
```response = self.session.get(f"{self.base_url}/job_status/{job_id}")```