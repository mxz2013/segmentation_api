# ====================
# 1. TASK QUEUE MANAGER
# ====================
import redis
import json
import uuid
from typing import Dict, Optional
from datetime import datetime


class TaskQueue:
    """
    A Redis-based task queue for managing inference requests.
    """

    def __init__(self, redis_host="localhost", redis_port=6379):
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, decode_responses=True
        )
        self.task_queue = "inference_tasks"
        self.result_prefix = "result:"

    def submit_task(
        self, image_path: str, target_class_ids: list, threshold: float
    ) -> str:
        """
        Submit a new inference task to the queue.
        Args:
            image_path:
            target_class_ids:
            threshold:

        Returns:

        """
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "image_path": image_path,
            "target_class_ids": target_class_ids,
            "threshold": threshold,
            "status": "pending",
            "submitted_at": datetime.utcnow().isoformat(),
        }

        # Push to queue
        self.redis_client.rpush(self.task_queue, json.dumps(task_data))

        # Store initial status
        self.redis_client.setex(
            f"{self.result_prefix}{task_id}",
            3600,  # Expire after 1 hour
            json.dumps({"status": "pending"}),
        )

        return task_id

    def get_task(self, timeout: int = 0) -> Optional[Dict]:
        """
        Get next task from queue (blocking).
        Args:
            timeout:

        Returns:

        """
        result = self.redis_client.blpop(self.task_queue, timeout=timeout)
        if result:
            _, task_json = result
            return json.loads(task_json)
        return None

    def store_result(self, task_id: str, result: Dict):
        """
        Store task result
        Args:
            task_id:
            result:

        Returns:

        """
        self.redis_client.setex(
            f"{self.result_prefix}{task_id}", 3600, json.dumps(result)
        )

    def get_result(self, task_id: str) -> Optional[Dict]:
        """
        Get task result
        Args:
            task_id:

        Returns:

        """
        result = self.redis_client.get(f"{self.result_prefix}{task_id}")
        return json.loads(result) if result else None
