A semantic segmentation model using PyTorch and torchvision. The API is build with FastAPI, with RQ for task queue management. The key improvement is to handel requests asynchronously, allowing multiple clients to submit requests simultaneously without blocking. 
The heavy lifting of model inference is handled by RQ workers, which process tasks sequentially from a Redis queue, either by CPU or GPU.
The code is designed to cache model loading correctly to avoid reloading the model for each request in the same worker process 

# Requirements
- poetry is recommended for dependency management.
- The code has been tested on Ubuntu 24.04 with Python 3.10 with CPU inference.
- Python 3.10.6 

# How fastapi works with RQ and Redis 
Suppose that we are having 3 clients sending requests simultaneously to the server for image segmentation.

## 1. client -> server (request submission)
```
# 3 threads simultaneously send these requests:
response1 = session.post("/predict", json=request1.dict())  # Thread 1
response2 = session.post("/predict", json=request2.dict())  # Thread 2  
response3 = session.post("/predict", json=request3.dict())  # Thread 3`
```
After sending all requests, the clients start to wait for responses.

## 2. server processing (FastAPI - Asynchronous handling)
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
```

## 4. RQ Worker Processing (sequential processing, i.e., synchronous)
```
# Worker runs in infinite loop (a piece of pseudo-code):
while True:
    job = queue.dequeue()  # Gets Job1 (blocks until job available)
    result = perform_segmentation_task(job.data)  # SYNC - blocks here
    job.set_result(result)  # Store result in Redis
    # Then gets Job2, then Job3...
```

After processing results, the clients can poll for job status and retrieve results by a unique job id.
```response = self.session.get(f"{self.base_url}/job_status/{job_id}")```